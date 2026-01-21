"""Two-tier universe construction (eligibility + research ranking).

This package is intentionally standalone and *not* wired into Stage B / Phase2 yet.
It implements a leak-aware architecture where:
- Tier 1 uses provider (EODHD) constraints to build EligibleUniverse(t)
- Tier 2 uses internal research parquets to rank EligibleUniverse(t)
- Only final universe snapshots are persisted.

See docs: docs/two_tier_universe_construction.md
"""

from .types import (
    EligibilityConfig,
    ResearchRankingConfig,
    UniverseConstructionConfig,
)
from .pipeline import build_universe_snapshot

__all__ = [
    "EligibilityConfig",
    "ResearchRankingConfig",
    "UniverseConstructionConfig",
    "build_universe_snapshot",
]
