from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import pandas as pd


def _default_snapshot_dir() -> Path:
    """Return the default snapshot directory, using cache_paths if available."""
    try:
        from src.cache_paths import UNIVERSE_CACHE_ROOT
        return UNIVERSE_CACHE_ROOT / "two_tier"
    except ImportError:
        return Path("data/cache/universe/two_tier")


@dataclass(frozen=True)
class EligibilityConfig:
    """Tier-1 (hard gate) config.

    All rules are binary constraints evaluated *as-of* a date.

    Note: This config is intentionally conservative and provider-only.
    """

    exchange_code: str = "US"
    country_allow: Optional[Sequence[str]] = None
    sector_allow: Optional[Sequence[str]] = None
    sector_block: Optional[Sequence[str]] = None

    exclude_etf: bool = True
    exclude_adr: bool = True
    exclude_fund: bool = True

    min_market_cap: Optional[float] = None
    min_avg_volume: Optional[float] = None
    min_price: Optional[float] = None

    # Optional sanity checks (still binary).
    require_not_suspended: bool = True
    require_not_delisted: bool = True


@dataclass(frozen=True)
class ResearchRankingConfig:
    """Tier-2 (research ranking) config.

    Tier 2 consumes internal merged parquets (research-only) and produces
    cross-sectional per-date scores/ranks.
    """

    horizon: int = 63

    # Leak-safety: when selecting a universe for date t, use features up to
    # (t - leakage_safe_shift_sessions). Default 1 session is conservative.
    leakage_safe_shift_sessions: int = 1

    # Which families contribute to the ranking score. If None, uses the
    # provided Stage-A weights keys.
    families: Optional[Sequence[str]] = None

    # Minimum families required per symbol to be rankable.
    min_non_nan_families: int = 1

    # Cross-sectional normalization.
    zscore_clip: float = 6.0


@dataclass(frozen=True)
class UniverseConstructionConfig:
    """Overall two-tier universe construction config."""

    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    ranking: ResearchRankingConfig = field(default_factory=ResearchRankingConfig)

    # Where to persist snapshots (parquet). Only snapshots are persisted.
    snapshot_dir: Path = field(default_factory=_default_snapshot_dir)


# --------------------------
# Dependency interfaces
# --------------------------

# A provider capable of evaluating Tier-1 eligibility constraints.
# In production this is expected to wrap EODHD.
Tier1EligibilityProvider = Any


class MergedParquetStore:
    """Internal-only feature store for merged parquets.

    This interface is intentionally narrow so raw merged feature panels are
    never returned to downstream consumers outside the research layer.
    """

    def load_panel(self, *, symbol: str, horizon: int) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


# Provider for Stage-A family weights.
StageAWeightsProvider = Callable[[pd.Timestamp], Dict[str, float]]


@dataclass(frozen=True)
class UniverseSnapshot:
    """Final persisted output (date×symbol) for the selected universe."""

    frame: pd.DataFrame
    meta: Dict[str, Any]
