# Two-Tier Universe Construction (EODHD + Research Parquets)

## 0. Executive summary
This module defines a two-tier, time-aware, leak-resistant universe construction architecture: Tier 1 applies *binary eligibility constraints* using only provider data (EODHD) to produce `EligibleUniverse(t)`, and Tier 2 applies *research-grade ranking* using internal merged parquets to score and rank the eligible set per-date using Stage-A family weights. Only the final date×symbol universe snapshots are persisted and passed downstream (e.g., to Mamba); raw merged features never leave the research layer.

---

## 1. Data foundations (what exists before tiers)

### 1.1 Merged Parquet Store (internal only)
Merged parquets (e.g. `cache/features/{SYM}_h{H}_merged.parquet`) are treated as a **research-only feature store**. They contain full feature panels, and are not exposed outside the Tier-2 implementation.

In code, this boundary is represented by `MergedParquetStore` (see `src/universe_two_tier/types.py`), which is used internally by Tier 2.

### 1.2 Stage-A family weights
Tier 2 consumes Stage-A weights (typically `family_weights_best.json`). The architecture supports either:
- a static weights dict, or
- a time-indexed schedule of weights selected with `asof <= t`.

---

## 2. Tier 1 — EODHD Screener (Hard Eligibility Gate)

### Purpose
Answer only one question:

> Which stocks are allowed to be considered at date *t*?

This tier does not rank. It filters.

### Inputs
- `date t`
- Global / regional scope (exchange / country)
- Screener rules (static config)

### Rules (binary, maximum safe extent)
Tier 1 is allowed to use only provider data (EODHD). Rules are modeled as boolean flags, e.g.:

**Structural**
- Exchange, country, sector allow/block
- ETF/ADR/fund exclusion
- Suspended/delisted exclusion

**Liquidity & size**
- `market_cap >= X`
- `avg_volume_20 >= Y`
- `close_price >= Z`

**Financial sanity (optional)**
- Positive cash flow / not negative book value / bankruptcy flags (provider permitting)

### Output
`EligibleUniverse(t) = { s₁, s₂, …, sₙ }`

Persisted (Tier 1 audit-only; optional) as:
- `(date, symbol, eligible, eligibility_flags...)`

### Invariants
- Uses only EODHD/provider data
- Uses only data ≤ *t*
- No ranking
- No weights
- Does not read merged parquets

---

## 3. Tier 2 — Research Ranking Gate (Merged Parquet-Driven)

### Purpose
This is where accuracy is created.

### Inputs
For each `s ∈ EligibleUniverse(t)`:
- merged parquet slice: `panel[s].loc[<=t]` (research-only)
- Stage-A family weights (static or scheduled as-of)
- peer set: `EligibleUniverse(t)`

### 3.2 Cross-Sectional Feature Projection (critical)
Tier 2 consumes **snapshots, not histories**.

Rule:
- For each family, collapse its time-series into a scalar snapshot at date *t* (or conservative `t-1` if `leakage_safe_shift_sessions=1`).
- The mapping is deterministic and must use only data ≤ *t*.

Default architecture implementation:
- `snap_family(t, s) := mean( panel[s].loc[t_eff, family_cols] )` where `t_eff = t - shift_sessions`.

This is intended as a safe baseline; you can later replace it with explicit per-family signal definitions.

### 3.3 Cross-sectional normalization (safe)
For each family snapshot `snap_f(t, s)`:
- Compute `z_f(t, s)` by z-scoring **across symbols** in `EligibleUniverse(t)`.

Important:
- normalization is per-date
- never across time
- never across folds

### 3.4 Ranking
Compute a final score:

`score(t, s) = Σ_f w_f(t) * z_f(t, s)`

Then rank symbols by `score(t, s)` descending.

### Output
A final universe snapshot parquet containing only:
- date, symbol
- Tier-1 flags (provider eligibility)
- Tier-2 z-scores (family-level summaries)
- final score and rank

Raw merged feature columns are not persisted.

---

## 4. Persistence & boundaries

Only final universe snapshots are persisted:
- `data/cache/universe/two_tier/universe_snapshot_YYYYMMDD_h{H}.parquet`
- `...meta.json` with config + provenance

Downstream systems (e.g., Mamba) consume only these snapshots.

---

## 5. Code layout
- `src/universe_two_tier/tier1_eligibility.py`: Tier-1 hard gate
- `src/universe_two_tier/tier2_research_ranking.py`: Tier-2 research gate (snapshots + cross-sectional ranking)
- `src/universe_two_tier/merged_store.py`: local merged parquet loader (research-only)
- `src/universe_two_tier/snapshot_store.py`: snapshot persistence (parquet + meta JSON)
- `src/universe_two_tier/pipeline.py`: orchestrates Tier 1 + Tier 2

---

## 6. Non-goals (for now)
- No integration into Phase2 / Mamba yet
- No hard dependency on a specific EODHD screener endpoint
- No commitment to a specific family→signal mapping; the module provides a safe baseline and explicit extension points
