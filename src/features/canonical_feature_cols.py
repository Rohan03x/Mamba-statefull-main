from __future__ import annotations

"""Feature schema contracts for drift-prone families.

Institutional policy: Accept-and-Warn + Contract Enforcement.

These contracts are used to validate cached unified panels before Stage B trusts them.
The validation is intentionally *not* an exact allowlist check:
- Extra columns are allowed (and logged as drift).
- We fail only on contract violations:
    - required columns missing
    - forbidden columns present

Column names here should match the *final* feature column names as they appear in
cached parquet outputs (lowercase, without the date/timestamp columns).
"""

from typing import Dict, FrozenSet, List


ML_FRAMEWORK_REQUIRED_COLS: FrozenSet[str] = frozenset(
    {
        # Minimal technical bundle contract (stable subset).
        "ml_framework_ml_sma_20",
        "ml_framework_ml_sma_50",
        "ml_framework_ml_ema_12",
        "ml_framework_ml_ema_26",
        "ml_framework_ml_rsi_14",
        "ml_framework_ml_atr_14",
        "ml_framework_ml_macd",
        "ml_framework_ml_macd_signal",
    }
)

ML_FRAMEWORK_FORBIDDEN_COLS: FrozenSet[str] = frozenset()


CORRELATION_REQUIRED_COLS: FrozenSet[str] = frozenset(
    {
        # Minimal correlation contract: core regime/risk signals.
        "correlation_corr_20_spy",
        "correlation_corr_60_spy",
        "correlation_corr_20_qqq",
        "correlation_corr_20_sector",
        "correlation_corr_decoupling_z",
    }
)

CORRELATION_FORBIDDEN_COLS: FrozenSet[str] = frozenset()


EARNINGS_REQUIRED_COLS: FrozenSet[str] = frozenset(
    {
        # Minimal earnings contract used across engines.
        "earnings_has_data",
        "earnings_eps_surprise_pct",
        "earnings_revenue_surprise_pct",
        "earnings_days_since_earnings",
        "earnings_event_decay",
        "earnings_days_to_next_earnings",
    }
)

EARNINGS_FORBIDDEN_COLS: FrozenSet[str] = frozenset()


FAMILY_REQUIRED_COLS: Dict[str, FrozenSet[str]] = {
    "ml_framework": ML_FRAMEWORK_REQUIRED_COLS,
    "correlation": CORRELATION_REQUIRED_COLS,
    "earnings": EARNINGS_REQUIRED_COLS,
}

FAMILY_FORBIDDEN_COLS: Dict[str, FrozenSet[str]] = {
    "ml_framework": ML_FRAMEWORK_FORBIDDEN_COLS,
    "correlation": CORRELATION_FORBIDDEN_COLS,
    "earnings": EARNINGS_FORBIDDEN_COLS,
}


META_COL_SUFFIXES: FrozenSet[str] = frozenset(
    {
        "_has_data",
        "_activity",
        "_days_since_update",
    }
)


# ---------------------------------------------------------------------------
# Legacy exports (backwards compatible)
# ---------------------------------------------------------------------------
#
# These are used by tools/prep_families.py to write shape-stable caches.
# They are intentionally *not* used for StageB's cached-panel acceptance logic
# anymore (StageB uses FAMILY_REQUIRED_COLS / FAMILY_FORBIDDEN_COLS).

ML_FRAMEWORK_CANONICAL_FEATURE_COLS: List[str] = [
    "ml_framework_ml_sma_20",
    "ml_framework_ml_sma_50",
    "ml_framework_ml_ema_12",
    "ml_framework_ml_ema_26",
    "ml_framework_ml_bbands_upper",
    "ml_framework_ml_bbands_middle",
    "ml_framework_ml_bbands_lower",
    "ml_framework_ml_rsi_14",
    "ml_framework_ml_atr_14",
    "ml_framework_ml_macd",
    "ml_framework_ml_macd_signal",
    "ml_framework_ml_macd_divergence",
    # Governance columns (required for all families).
    "ml_framework_has_data",
    "ml_framework_activity",
    "ml_framework_days_since_update",
]

CORRELATION_CANONICAL_FEATURE_COLS: List[str] = [
    "correlation_corr_20_spy",
    "correlation_corr_60_spy",
    "correlation_corr_20_qqq",
    "correlation_corr_20_vxx",
    "correlation_corr_20_sector",
    "correlation_corr_decoupling_z",
    "correlation_lag_corr_1_spy",
    "correlation_lag_corr_5_spy",
    "correlation_lag_corr_1_vxx",
    "correlation_lag_corr_2_vxx",
    "correlation_corr_20_spy_trend",
    "correlation_corr_20_qqq_trend",
    "correlation_corr_20_vxx_trend",
    "correlation_corr_20_spy_vol",
    "correlation_corr_20_vxx_vol",
    "correlation_corr_spread_20_60_spy",
    "correlation_corr_spread_20_60_qqq",
    "correlation_corr_return_vol_20",
    "correlation_corr_return_range_10",
    "correlation_corr_vol_volatility_20",
    "correlation_corr_20_vix_lag1",
    "correlation_corr_60_vix_lag1",
    "correlation_corr_20_vix_lag2",
    "correlation_corr_60_vix_lag2",
    "correlation_acf_ret_1",
    "correlation_acf_ret_5",
    "correlation_acf_absret_1",
    "correlation_acf_vol_1",
    # Governance columns (required for all families).
    "correlation_has_data",
    "correlation_activity",
    "correlation_days_since_update",
]

EARNINGS_CANONICAL_FEATURE_COLS: List[str] = [
    "earnings_has_data",
    "earnings_eps_surprise_pct",
    "earnings_revenue_surprise_pct",
    "earnings_eps_growth_qoq",
    "earnings_eps_growth_yoy",
    "earnings_revenue_growth_qoq",
    "earnings_revenue_growth_yoy",
    "earnings_beat_streak",
    "earnings_miss_streak",
    "earnings_surprise_volatility",
    "earnings_beat_rate_3y",
    "earnings_revision_breadth",
    "earnings_estimate_dispersion",
    "earnings_days_since_earnings",
    "earnings_event_decay",
    "earnings_days_to_next_earnings",
    # Governance columns (required for all families).
    "earnings_activity",
    "earnings_days_since_update",
]

CANONICAL_FAMILY_COLS: Dict[str, List[str]] = {
    "ml_framework": ML_FRAMEWORK_CANONICAL_FEATURE_COLS,
    "correlation": CORRELATION_CANONICAL_FEATURE_COLS,
    "earnings": EARNINGS_CANONICAL_FEATURE_COLS,
}
