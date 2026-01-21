from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, cast

import pandas as pd

from .merged_store import LocalMergedParquetStore
from .snapshot_store import write_snapshot
from .tier1_eligibility import EODHDEligibilityGate
from .tier2_research_ranking import ResearchRankingGate
from .types import (
    MergedParquetStore,
    StageAWeightsProvider,
    UniverseConstructionConfig,
    UniverseSnapshot,
)


def build_universe_snapshot(
    *,
    asof: str | pd.Timestamp,
    candidate_symbols: Sequence[str],
    cfg: UniverseConstructionConfig,
    tier1_provider: Any,
    stage_a_weights: Mapping[str, float] | StageAWeightsProvider,
    merged_store: Optional[MergedParquetStore] = None,
    persist: bool = True,
) -> UniverseSnapshot:
    """Build a single date universe snapshot.

    This is the main entrypoint for the new two-tier architecture.

    Notes on boundaries:
    - Tier 1 touches *only* the provider (EODHD)
    - Tier 2 touches *only* internal merged parquets
    - Only the final snapshot is returned/persisted.

    Parameters
    - stage_a_weights:
        Either a static weights dict, or a callable that returns weights for the given asof.
    """

    asof_ts = pd.Timestamp(asof).normalize()

    # Resolve weights as-of date.
    if callable(stage_a_weights):
        raw: Any = stage_a_weights(asof_ts)
    else:
        raw = stage_a_weights

    raw_map: Mapping[str, Any] = cast(Mapping[str, Any], raw) if isinstance(raw, Mapping) else {}
    weights: Dict[str, float] = {str(k): float(v) for k, v in raw_map.items() if isinstance(v, (int, float))}

    # Tier 1: eligibility gate.
    t1 = EODHDEligibilityGate(provider=tier1_provider, cfg=cfg.eligibility)
    elig_df = t1.evaluate(symbols=candidate_symbols, asof=asof_ts)
    elig_now = elig_df[elig_df["eligible"]].copy()
    eligible = [str(s).upper() for s in elig_now["symbol"].tolist()]

    # Tier 2: research ranking gate.
    store = merged_store or LocalMergedParquetStore()
    t2 = ResearchRankingGate(store=store, cfg=cfg.ranking)
    tier2 = t2.rank(eligible=eligible, asof=asof_ts, stage_a_weights=weights)

    # Merge to final snapshot shape.
    ranked = tier2.ranked.copy()
    ranked["eligible"] = True

    # Attach Tier-1 flags for auditability.
    # This is still safe: it's not merged features, it's provider eligibility.
    keep_elig_cols = [c for c in elig_df.columns if c.startswith("flag_")]
    if keep_elig_cols:
        ranked = ranked.merge(
            elig_df[["symbol", *keep_elig_cols]].copy(),
            how="left",
            on="symbol",
        )

    # Final snapshot schema: minimal + auditable.
    snapshot = ranked.copy()
    snapshot["date"] = pd.Timestamp(asof_ts)
    snapshot = snapshot.sort_values(["rank", "symbol"]).reset_index(drop=True)

    meta: Dict[str, Any] = {
        "asof": asof_ts.strftime("%Y-%m-%d"),
        "candidate_in": int(len(candidate_symbols)),
        "eligible_out": int(len(eligible)),
        "ranked_out": int(len(snapshot)),
        "tier1": {
            "exchange": str(cfg.eligibility.exchange_code),
        },
        "tier2": dict(tier2.meta or {}),
    }

    if persist:
        write_snapshot(cfg=cfg, asof=asof_ts, frame=snapshot, meta=meta, overwrite=True)

    return UniverseSnapshot(frame=snapshot, meta=meta)
