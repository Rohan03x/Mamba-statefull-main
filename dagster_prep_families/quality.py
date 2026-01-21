from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROXY_PATTERNS: Tuple[str, ...] = (
    "_has_data",
    "has_data",
    "is_etf",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
    "is_month_start",
    "is_month_end",
    "proxy",
    "fallback",
)

EXEMPT_SUBSTRINGS: Tuple[str, ...] = (
    "trust_score",  # frequently constant diagnostic
)


SNAPSHOT_FAMILIES: Set[str] = {
    # Options chain snapshots are point-in-time and may only appear on a few
    # window end-dates (depending on schedule/API availability).
    "options",
}


def _family_columns(panel: pd.DataFrame, family: str) -> List[str]:
    """Return columns that belong to a given family.

    Default convention is prefix-based: "{family}_*".

    Some legacy families emit non-prefixed columns that are still conceptually
    part of the family (notably macro derived/interaction blocks).
    """

    family = str(family)
    cols = [c for c in panel.columns if str(c).startswith(f"{family}_")]

    # Legacy macro_tst_hf emits these without a family prefix.
    if family == "macro_tst_hf":
        cols.extend([c for c in panel.columns if str(c).startswith("derived_")])
        cols.extend([c for c in panel.columns if str(c).startswith("derived_interact")])
        if "l3_real_interest_rate" in panel.columns:
            cols.append("l3_real_interest_rate")

    # De-dupe while preserving order.
    seen: Set[str] = set()
    ordered: List[str] = []
    for c in cols:
        if c in seen:
            continue
        ordered.append(c)
        seen.add(c)
    return ordered


@dataclass
class QualityThresholds:
    min_variance_threshold: float = 1e-10
    max_zero_percentage: float = 0.95
    max_null_percentage: float = 0.90

    # "Real data" checks
    min_has_data_mean: float = 0.05  # fail if a *_has_data column is ~always 0
    max_proxy_flag_mean: float = 0.05  # fail if a *_proxy/*fallback flag is frequently 1

    # Coverage checks are evaluated on a recent tail window (not full history), since
    # many families legitimately only become available later in the sample and some
    # are inherently sparse (e.g., earnings-day signals).
    has_data_tail_rows: int = 252
    min_has_data_nonzero_tail: int = 3  # require at least N active days in tail window

    # Sampling
    max_rows: Optional[int] = 100_000
    random_state: int = 7


@dataclass
class FamilyQualityResult:
    family: str
    passed: bool
    reason: str
    n_cols: int
    n_proxy_cols: int
    n_real_cols: int


def _is_proxy_col(col: str) -> bool:
    c = str(col).lower()
    return any(p in c for p in PROXY_PATTERNS)


def _is_exempt_col(col: str) -> bool:
    c = str(col).lower()
    return any(p in c for p in EXEMPT_SUBSTRINGS)


def _maybe_sample(df: pd.DataFrame, thresholds: QualityThresholds) -> pd.DataFrame:
    if thresholds.max_rows is None:
        return df
    if len(df) <= int(thresholds.max_rows):
        return df
    return df.sample(n=int(thresholds.max_rows), random_state=int(thresholds.random_state))


def _is_binary_like(series: pd.Series) -> bool:
    """Return True if the series looks like a 0/1 flag.

    We intentionally avoid treating continuous "*_proxy" numeric columns as flags,
    since their mean can be arbitrarily large and would create false failures.
    """
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return False
        # Sample to keep checks fast.
        if len(s) > 1000:
            s = s.sample(n=1000, random_state=7)
        # Tolerate float-y flags.
        rounded = s.round(6)
        uniq = set(rounded.unique().tolist())
        return uniq.issubset({0.0, 1.0})
    except Exception:
        return False


def validate_family_quality_in_panel(
    panel: pd.DataFrame,
    family: str,
    thresholds: QualityThresholds,
    allowed_dormant_families: Optional[Sequence[str]] = None,
    fail_on_missing: bool = False,
    allowed_missing_families: Optional[Sequence[str]] = None,
) -> FamilyQualityResult:
    family = str(family)
    allowed: Set[str] = set()
    if allowed_dormant_families:
        allowed = {str(f).strip() for f in allowed_dormant_families if str(f).strip()}

    allowed_missing: Set[str] = set()
    if allowed_missing_families:
        allowed_missing = {str(f).strip() for f in allowed_missing_families if str(f).strip()}

    cols = _family_columns(panel, family)
    if not cols:
        if fail_on_missing and family not in allowed_missing:
            return FamilyQualityResult(
                family=family,
                passed=False,
                reason="missing: no columns present in merged panel",
                n_cols=0,
                n_proxy_cols=0,
                n_real_cols=0,
            )
        return FamilyQualityResult(
            family=family,
            passed=True,
            reason="missing: no columns (ignored)",
            n_cols=0,
            n_proxy_cols=0,
            n_real_cols=0,
        )

    proxy_cols = [c for c in cols if _is_proxy_col(str(c))]
    real_cols = [c for c in cols if c not in proxy_cols and not _is_exempt_col(str(c))]

    if not real_cols:
        if family in allowed:
            return FamilyQualityResult(
                family=family,
                passed=True,
                reason=f"allowed dormant: only proxy/flag/diagnostic columns ({len(proxy_cols)} proxy cols, 0 real features)",
                n_cols=len(cols),
                n_proxy_cols=len(proxy_cols),
                n_real_cols=0,
            )
        return FamilyQualityResult(
            family=family,
            passed=False,
            reason=f"only proxy/flag/diagnostic columns ({len(proxy_cols)} proxy cols, 0 real features)",
            n_cols=len(cols),
            n_proxy_cols=len(proxy_cols),
            n_real_cols=0,
        )

    df = _maybe_sample(panel[real_cols + proxy_cols], thresholds)

    # Real-data coverage: *_has_data columns should not be almost always 0.
    has_data_cols = [c for c in proxy_cols if str(c).lower().endswith("has_data") or "_has_data" in str(c).lower()]
    if family in allowed and has_data_cols:
        # If a family is allowed to be dormant and RECENT coverage is low, do not enforce
        # degenerate-feature checks (zeros/nulls/variance). If it IS active in the tail,
        # we still want to validate it normally.
        tail_means: List[float] = []
        tail_nonzeros: List[int] = []

        required_nonzero_tail = int(thresholds.min_has_data_nonzero_tail)
        if family in SNAPSHOT_FAMILIES:
            required_nonzero_tail = min(required_nonzero_tail, 1)

        for c in has_data_cols:
            s_full = pd.to_numeric(df[c], errors="coerce")
            if not len(s_full):
                continue
            s_tail = s_full.fillna(0.0).tail(int(thresholds.has_data_tail_rows))
            tail_means.append(float(s_tail.mean()) if len(s_tail) else 0.0)
            tail_nonzeros.append(int((s_tail > 0.0).sum()))

        cov_tail = float(max(tail_means)) if tail_means else 0.0
        nz_tail = int(max(tail_nonzeros)) if tail_nonzeros else 0

        if (cov_tail < thresholds.min_has_data_mean) and (nz_tail < required_nonzero_tail):
            return FamilyQualityResult(
                family=family,
                passed=True,
                reason=(
                    f"allowed dormant: tail has_data coverage={cov_tail:.3f} < min_has_data_mean={thresholds.min_has_data_mean:.3f} "
                    f"AND tail_nonzero={nz_tail} < min_has_data_nonzero_tail={required_nonzero_tail}"
                ),
                n_cols=len(cols),
                n_proxy_cols=len(proxy_cols),
                n_real_cols=len(real_cols),
            )

    for c in has_data_cols:
        s_full = pd.to_numeric(df[c], errors="coerce")
        if not len(s_full):
            continue
        s_tail = s_full.fillna(0.0).tail(int(thresholds.has_data_tail_rows))
        mean_tail = float(s_tail.mean()) if len(s_tail) else 0.0
        nonzero_tail = int((s_tail > 0.0).sum())

        required_nonzero_tail = int(thresholds.min_has_data_nonzero_tail)
        if family in SNAPSHOT_FAMILIES:
            required_nonzero_tail = min(required_nonzero_tail, 1)

        if (mean_tail < thresholds.min_has_data_mean) and (nonzero_tail < required_nonzero_tail):
            return FamilyQualityResult(
                family=family,
                passed=False,
                reason=(
                    f"{c} tail_mean={mean_tail:.3f} < min_has_data_mean={thresholds.min_has_data_mean:.3f} AND "
                    f"tail_nonzero={nonzero_tail} < min_has_data_nonzero_tail={required_nonzero_tail} "
                    f"(looks like stub/proxy/missing source)"
                ),
                n_cols=len(cols),
                n_proxy_cols=len(proxy_cols),
                n_real_cols=len(real_cols),
            )

    # Proxy/fallback flags: if present and frequently 1, fail.
    proxy_flag_cols = [c for c in proxy_cols if any(k in str(c).lower() for k in ("proxy", "fallback"))]
    for c in proxy_flag_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if not _is_binary_like(s):
            # Many "*_proxy" columns are continuous proxy values, not boolean flags.
            continue
        mean = float(s.fillna(0.0).mean()) if len(s) else 0.0
        if mean > thresholds.max_proxy_flag_mean:
            return FamilyQualityResult(
                family=family,
                passed=False,
                reason=f"{c} mean={mean:.3f} > max_proxy_flag_mean={thresholds.max_proxy_flag_mean:.3f}",
                n_cols=len(cols),
                n_proxy_cols=len(proxy_cols),
                n_real_cols=len(real_cols),
            )

    # Numeric-only checks.
    numeric_cols = df[real_cols].select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return FamilyQualityResult(
            family=family,
            passed=False,
            reason=f"no numeric real features (real cols: {len(real_cols)})",
            n_cols=len(cols),
            n_proxy_cols=len(proxy_cols),
            n_real_cols=len(real_cols),
        )

    variances = df[numeric_cols].var(ddof=0)
    features_with_variance = int((variances > thresholds.min_variance_threshold).sum())
    if features_with_variance == 0:
        return FamilyQualityResult(
            family=family,
            passed=False,
            reason=f"ALL {len(numeric_cols)} numeric features ~zero variance",
            n_cols=len(cols),
            n_proxy_cols=len(proxy_cols),
            n_real_cols=len(real_cols),
        )

    zero_percentages = (df[numeric_cols] == 0).sum() / max(1, len(df))
    high_zero_features = int((zero_percentages > thresholds.max_zero_percentage).sum())

    null_percentages = df[numeric_cols].isna().sum() / max(1, len(df))
    high_null_features = int((null_percentages > thresholds.max_null_percentage).sum())

    if high_zero_features == len(numeric_cols):
        return FamilyQualityResult(
            family=family,
            passed=False,
            reason=f"ALL {len(numeric_cols)} numeric features are >{int(thresholds.max_zero_percentage * 100)}% zeros",
            n_cols=len(cols),
            n_proxy_cols=len(proxy_cols),
            n_real_cols=len(real_cols),
        )

    if high_null_features == len(numeric_cols):
        return FamilyQualityResult(
            family=family,
            passed=False,
            reason=f"ALL {len(numeric_cols)} numeric features are >{int(thresholds.max_null_percentage * 100)}% null",
            n_cols=len(cols),
            n_proxy_cols=len(proxy_cols),
            n_real_cols=len(real_cols),
        )

    return FamilyQualityResult(
        family=family,
        passed=True,
        reason=f"ok ({features_with_variance}/{len(numeric_cols)} numeric cols have variance)",
        n_cols=len(cols),
        n_proxy_cols=len(proxy_cols),
        n_real_cols=len(real_cols),
    )


def validate_panel_by_known_families(
    panel: pd.DataFrame,
    thresholds: QualityThresholds,
    families: Optional[Sequence[str]] = None,
    allowed_dormant_families: Optional[Sequence[str]] = None,
    fail_on_missing: bool = False,
    allowed_missing_families: Optional[Sequence[str]] = None,
) -> Tuple[bool, List[FamilyQualityResult]]:
    if families is None:
        # Lazy import to avoid heavy module load at dagster startup.
        from src.features.family_spec import FAMILY_SPECS

        families = list(FAMILY_SPECS.keys())

    results: List[FamilyQualityResult] = []
    any_failed = False

    for fam in families:
        r = validate_family_quality_in_panel(
            panel=panel,
            family=str(fam),
            thresholds=thresholds,
            allowed_dormant_families=allowed_dormant_families,
            fail_on_missing=fail_on_missing,
            allowed_missing_families=allowed_missing_families,
        )
        if r.n_cols == 0 and not fail_on_missing:
            # In non-strict mode we ignore completely absent families.
            continue
        results.append(r)
        if not r.passed:
            any_failed = True

    return (not any_failed), results
