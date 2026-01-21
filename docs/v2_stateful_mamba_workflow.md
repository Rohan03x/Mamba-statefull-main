# Phase 2 v2 Stateful Mamba — End-to-End Workflow (Training-Only)

This document is the detailed workflow map for the Phase 2 v2 stateful Mamba engine, from feature generation and Dagster prep_families through caching, Track-C construction, training, and portfolio backtest outputs. This is **training-only** (no live execution model).

Authoritative code references:
- Dagster prep_families asset: [dagster_prep_families/assets.py](dagster_prep_families/assets.py)
- Feature generation orchestrator: [tools/prep_families.py](tools/prep_families.py)
- Feature aggregation (family fetch/merge): [src/features/aggregator_panel.py](src/features/aggregator_panel.py)
- Stage B pipeline (panel loading, validation, Track A/B/C assembly): [src/stage_b/pipeline.py](src/stage_b/pipeline.py)
- Phase 2 v2 core: [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py)
- Phase 2 CLI runner: [tools/run_stage_b_stateful_phase2.py](tools/run_stage_b_stateful_phase2.py)
- Prediction tape export: [src/stage_b/stage_b_export.py](src/stage_b/stage_b_export.py)
- Backtest engine: [src/stage_b/backtest.py](src/stage_b/backtest.py)
- Feature families registry: [src/features/family_spec.py](src/features/family_spec.py)
- Phase2 controls and tuning space: [docs/stateful_mamba_phase2_controls_and_backtest.md](docs/stateful_mamba_phase2_controls_and_backtest.md)
- Feature family schema snapshot: [docs/feature_families_and_features.md](docs/feature_families_and_features.md)
- Data readiness summary (Track‑C and overlays): [DATA_READINESS_SUMMARY.md](DATA_READINESS_SUMMARY.md)
- Optuna search space (68+ params): [Phase2_Optuna_Search_Space.md](Phase2_Optuna_Search_Space.md)
- Sequence models (Mamba training): [src/stage_b/sequence_models.py](src/stage_b/sequence_models.py)
- Meta-optimizer (self-learning): [src/stage_b/meta_optimizer.py](src/stage_b/meta_optimizer.py)
- Maturity gating (leakage prevention): [src/stage_b_stateful/maturity.py](src/stage_b_stateful/maturity.py)
- Group map builder (sector caps): [src/stage_b_stateful/group_map.py](src/stage_b_stateful/group_map.py)
- Stage-A selector (family weights): [tools/stage_a_selector.py](tools/stage_a_selector.py)
- Universe selector (core/satellite): [src/stage_b/universe_selector.py](src/stage_b/universe_selector.py)
- Delisting meta controls: [src/stage_b/delisting_meta.py](src/stage_b/delisting_meta.py)
- Universe registry builder: [src/stage_b/universe_registry.py](src/stage_b/universe_registry.py)
- Benchmarking framework: [src/analytics/benchmarking.py](src/analytics/benchmarking.py)
- EODHD benchmark data: [src/analytics/eodhd_benchmark_data.py](src/analytics/eodhd_benchmark_data.py)
- Benchmark backtest tool: [tools/benchmark_backtest.py](tools/benchmark_backtest.py)
- Realtime benchmark tracker: [tools/realtime_benchmark_tracker.py](tools/realtime_benchmark_tracker.py)
- Benchmark dashboard plotter: [tools/plot_benchmark_dashboard.py](tools/plot_benchmark_dashboard.py)
- Horizon benchmark comparison: [tools/compare_horizon_benchmarks.py](tools/compare_horizon_benchmarks.py)
- Stage C v2 CLI runner: [tools/run_stage_c_v2.py](tools/run_stage_c_v2.py)
- EODHD data provider: [src/data_sources/eodhd_provider.py](src/data_sources/eodhd_provider.py)
- Phase2 Optuna monitor: [tools/monitor_phase2_stateful_optuna.py](tools/monitor_phase2_stateful_optuna.py)
- Audit Phase2 family coverage: [tools/audit_phase2_family_coverage.py](tools/audit_phase2_family_coverage.py)
- Audit TrackC columns: [tools/audit_trackc_columns.py](tools/audit_trackc_columns.py)
- Audit TrackC cache: [tools/audit_trackc_cache.py](tools/audit_trackc_cache.py)
- Audit merged parquet quality: [tools/audit_merged_parquet_quality.py](tools/audit_merged_parquet_quality.py)

---

## 1) End-to-end workflow (training-only)

### 1.1 High-level flow (proper workflow order)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 v2 STATEFUL MAMBA WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │ 1. UNIVERSE SETUP    │  ← Delisting registry, universe registry
  │    (Section 3)       │     Determines which symbols are eligible
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 2. DATA INGESTION    │  ← prep_families(), aggregator_panel
  │    (Section 4)       │     Fetches raw data per family per symbol
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 3. DAGSTER CACHE     │  ← prep_families_manifest asset
  │    (Section 5)       │     Writes merged parquets + provenance
  │                      │     Output: cache/features/<SYM>_h<H>_merged.parquet
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 4. STAGE-A SELECTOR  │  ← tools/stage_a_selector.py
  │    (Section 6)       │     Consumes merged parquets
  │                      │     Fits logistic model on forward returns
  │                      │     Output: artifacts/stage_a/.../family_weights_best.json
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 5. PHASE 2 v2 TRAIN  │  ← phase2_stateful.py
  │    (Section 8)       │     Loads Stage-A weights (fixed governance)
  │                      │     Builds Track-A/B/C with 3-pillar dims
  │                      │     Runs Mamba walk-forward training
  │                      │     Applies portfolio overlays (covariance, beta, etc.)
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 6. PREDICTION TAPE   │  ← BacktestEngine
  │    (Section 9)       │     Converts predictions → portfolio equity
  │                      │     Output: artifacts/prediction_tapes/*.parquet
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ 7. STAGE C POLICY    │  ← stage_c_policy_v2.py (optional, research-only)
  │    (Section 10)      │     Threshold optimization, walk-forward policy
  └──────────────────────┘
```

### 1.2 Key dependencies (what feeds what)
| Producer | Artifact | Consumer |
|----------|----------|----------|
| Universe registry | universe_registry.parquet | prep_families (symbol gating) |
| prep_families | merged.parquet | Stage-A selector |
| Stage-A selector | family_weights_best.json | Phase2 v2 (fixed weights) |
| Phase2 v2 | Track-C + predictions | BacktestEngine |
| BacktestEngine | prediction_tapes.parquet | Stage C policy |

---

## 2) Feature families (full list + grouping)

The canonical registry and ordering is defined in [src/features/family_spec.py](src/features/family_spec.py). Phase2 v2 uses Stage‑A families plus Stage‑B base families and HF blocks.

### 2.1 Stage‑A families (base families, tags and order)
From LEGACY_BASE_FAMILY_ORDER in [src/features/family_spec.py](src/features/family_spec.py):

- quantile_forecast
- arima_forecast
- cross_asset
- exchange_calendar
- econ_events_calendar
- corp_actions_splits
- marketcap_history
- garch_iv
- cboe_term
- correlation
- candle_mechanics
- microstructure
- options
- short_interest
- insider_form4
- index_constituents
- subsidiary
- earnings
- dividends
- alternative_signals
- ml_framework
- multiasset
- regime
- options_anchoring
- tft_features
- fin_g2
- fin_g3
- fin_g4
- fin_g5
- fin_g6
- peer_screener_context
- fin_g7
- finbert
- dcf
- calibration
- online_learning

### 2.2 HF modules
From LEGACY_HF_MODULE_ORDER in [src/features/family_spec.py](src/features/family_spec.py):

- earnings_transcript_hf
- doc_embedding_novelty_hf
- macro_tst_hf

### 2.3 HF blocks (HF aggregations)
From HF_BLOCK_ORDER in [src/features/family_spec.py](src/features/family_spec.py):

- tech_micro_hf
- forecast_hf
- vol_deriv_hf
- macro_regime_hf
- fundamental_val_hf
- news_nlp_hf

### 2.4 Meta families
- hf_agg (meta aggregation family; defined in [src/features/family_spec.py](src/features/family_spec.py))

### 2.5 Family schemas & columns
For per-family columns and schemas, see the generated snapshot in [docs/feature_families_and_features.md](docs/feature_families_and_features.md).

---

## 3) Universe setup (symbol eligibility gating)

**This must run BEFORE data ingestion.** Universe setup determines which symbols are eligible for training.

### 3.1 Delisting registry
Module: [src/stage_b/delisting_meta.py](src/stage_b/delisting_meta.py)

- `load_delisting_registry()`: loads locally cached delisting data from EODHD
- `get_delisted_date_map()`: returns symbol→delisted_date mapping
- `_detect_last_reliable_trade_date_from_prices()`: detects "zombie" symbols via stale price/volume

### 3.2 Universe registry builder
Module: [src/stage_b/universe_registry.py](src/stage_b/universe_registry.py)

- `build_universe_registry()`: creates authoritative universe snapshot parquet
- Schema: symbol, stop_date, is_eligible_today, reason, last_trade_date, delisted_date, days_stale
- **stop_date semantics**: eligible on date D iff D ≤ stop_date

### 3.3 Survivorship-bias controls
These controls:
- Prevent "zombie" symbols from entering training
- Cap panel history at stop_date/delist_date
- Used by Phase2 for survivorship‑bias‑aware universe gating

### 3.4 Two‑tier universe construction (optional)
Authoritative design doc: [docs/two_tier_universe_construction.md](docs/two_tier_universe_construction.md)

Code entry points:
- Orchestrator: [src/universe_two_tier/pipeline.py](src/universe_two_tier/pipeline.py)
- Tier‑1 eligibility gate (provider‑only): [src/universe_two_tier/tier1_eligibility.py](src/universe_two_tier/tier1_eligibility.py)
- Tier‑2 research ranking gate (merged parquets only): [src/universe_two_tier/tier2_research_ranking.py](src/universe_two_tier/tier2_research_ranking.py)
- Snapshot persistence: [src/universe_two_tier/snapshot_store.py](src/universe_two_tier/snapshot_store.py)
- Type definitions: [src/universe_two_tier/types.py](src/universe_two_tier/types.py)
- Weight utilities: [src/universe_two_tier/weights.py](src/universe_two_tier/weights.py)
- Merged parquet store: [src/universe_two_tier/merged_store.py](src/universe_two_tier/merged_store.py)

Outputs (parquet + meta JSON):
- cache/universe/two_tier/universe_snapshot_YYYYMMDD_h{H}.parquet
- cache/universe/two_tier/universe_snapshot_YYYYMMDD_h{H}.meta.json

---

## 4) Data ingestion and feature generation (deep dive)

### 4.1 Core generation entry point
- `prepare_families()` in [tools/prep_families.py](tools/prep_families.py) orchestrates:
   - per‑family cache creation
   - per‑symbol consolidated family caches
   - downstream unified cache artifacts
   - storage snapshots and cache telemetry (for troubleshooting)

### 4.2 Feature aggregation layer
- Family fetchers and aggregation are in [src/features/aggregator_panel.py](src/features/aggregator_panel.py).
- It merges families into a daily NYSE‑close aligned panel and attaches provenance/telemetry in DataFrame attrs.
- Supported families are explicitly enumerated in the module header and include (non‑exhaustive):
   - microstructure, cross_asset, macro_tst_hf, cboe_term, garch_iv, dividends
   - tech_micro_hf, forecast_hf, vol_deriv_hf, macro_regime_hf, fundamental_val_hf, news_nlp_hf
- The aggregation layer is the binding point for external data sources (EODHD, options, news/LLM pipelines, macro).

### 4.3 Cache locations (centralized via cache_paths.py)
All cache paths are now centralized in [src/cache_paths.py](src/cache_paths.py).

**Symbol-level caches:**
- cache/symbols/<SYMBOL>/h<H>/families/ — per-horizon family caches
- cache/symbols/<SYMBOL>/hf/ — horizon-invariant HF blocks
- cache/symbols/<SYMBOL>/transcripts/ — earnings call transcripts (DefeatBeta)
- cache/symbols/<SYMBOL>/short_interest/ — short interest history

**Data source caches:**
- cache/data_sources/eodhd/ — EODHD fundamentals, delisting, group map
- cache/data_sources/gdelt/ — GDELT news events
- cache/data_sources/gdelt_global/merged/ — GDELT global merged cache

**Embedding caches:**
- cache/embeddings/finbert/<SYMBOL>/ — FinBERT sentiment embeddings
- cache/embeddings/doc_novelty/<SYMBOL>/ — document embedding novelty

**Universe caches:**
- cache/universe/registry_YYYYMMDD.parquet
- cache/universe/two_tier/snapshot_YYYYMMDD_h<H>.parquet

**Legacy paths (backward compatible):**
- data/local_cache/<symbol>_h<horizon>/... — old family caches
- data/cache/eodhd/... — old EODHD cache
- data/cache/gdelt_global/merged/... — old GDELT global cache

### 4.4 Feature family registry (source of truth)
- All family definitions, stage membership, dependencies, ordering and meta tags are in [src/features/family_spec.py](src/features/family_spec.py).
- This registry drives:
   - which families are assembled for Stage B
   - how HF modules/blocks are ordered
   - which families are “meta” (e.g., `hf_agg`)

---

## 5) Dagster prep_families (authoritative cache producer)


### 5.1 Asset and behavior
- Asset: `prep_families_manifest` in [dagster_prep_families/assets.py](dagster_prep_families/assets.py).
- It runs `prepare_families()` in‑process and writes unified outputs.

### 5.2 Primary outputs (authoritative cache)
1) **Merged unified parquet** (single file per symbol/horizon):
   - cache/features/<SYMBOL>_h<H>_merged.parquet
2) **Provenance/metadata** (single JSON file):
   - cache/features/<SYMBOL>_h<H>_merged.meta.json
   - (Note: CSV provenance was redundant and has been removed)
3) **Completeness manifest**:
   - artifacts/prep_families/<symbol>_h<horizon>_completeness.json

### 5.3 Contract validation policy
### 5.4 Dagster‑side config (governance + safety)
### 5.5 Feast integration (merged parquet via offline tables)
When enabled, prep_families can produce merged parquets via Feast:
- Export per‑family caches → offline tables: [tools/feast/export_local_cache_to_feast_offline.py](tools/feast/export_local_cache_to_feast_offline.py)
- Generate FeatureView definitions: [tools/feast/generate_feature_definitions.py](tools/feast/generate_feature_definitions.py)
- Apply Feast repo: [feature_repo/feature_repo/feature_definitions.py](feature_repo/feature_repo/feature_definitions.py)
- Build merged parquet: [tools/feast/build_symbol_parquet.py](tools/feast/build_symbol_parquet.py)

Feast repo config:
- Feast registry + offline store: [feature_repo/feature_repo/feature_store.yaml](feature_repo/feature_repo/feature_store.yaml)
- Manual fallback definitions: [feature_repo/feature_repo/feature_definitions_manual.py](feature_repo/feature_repo/feature_definitions_manual.py)

This is wired through `build_symbol_merged_parquet_via_feast()` inside [tools/prep_families.py](tools/prep_families.py).
Key config inputs from [dagster_prep_families/assets.py](dagster_prep_families/assets.py):
- write_merged: bool (write unified merged parquet)
- enforce_no_proxy_sources / enforce_no_live_fallback / enforce_eodhd_only
- preprocess_role_normalize, preprocess_timing_rules, preprocess_timing_shift_days
- quality thresholds for variance/null/zero rates
- coverage policy: fail on missing families unless explicitly allowed

---

## 6) Stage-A selector (family weights governance)

**This runs AFTER Dagster merged parquets are produced.** It creates the fixed family weights that Phase2 consumes.

Module: [tools/stage_a_selector.py](tools/stage_a_selector.py)

### 6.1 What Stage-A does
- Consumes: Dagster merged parquets (`cache/features/<SYM>_h<H>_merged.parquet`)
- Fits: Logistic regression on forward returns to learn per-family importance
- Produces: `artifacts/stage_a/.../family_weights_best.json`

### 6.2 Output artifact
- **Path**: `artifacts/stage_a/<horizon>/<date>/family_weights_best.json`
- **Content**: per‑family weights (floats in [0,1]) learned via logistic regression

### 6.3 How Phase2 loads Stage-A weights
1. `_resolve_phase2_stage_a_payload(cfg)` locates the Stage-A artifact directory
2. `_resolve_phase2_family_weights(cfg)` reads `family_weights_best.json`
3. Weights are applied during Track‑A/B/C construction to gate family inclusion

**Key principle**: Stage‑A weights are **fixed** governance inputs—Optuna does not tune them.

---

## 7) Cache and artifact layout (training pipeline)

### 7.1 Key cache directories
- cache/features/ — unified outputs (merged/split/trackc)
- artifacts/prep_families/ — completeness manifests, logs
- artifacts/stage_a/ — Stage-A family weights (fixed governance)
- artifacts/three_pillar_cache/ — Phase2 3‑pillar dimension cache
- artifacts/optuna_studies/ — Optuna SQLite DBs
- artifacts/optuna/ — winner JSON exports
- artifacts/prediction_tapes/ — prediction tape parquets
- artifacts/backtests/ — policy/backtest outputs
- artifacts/meta_optimizer/ — group maps and other governance inputs

### 7.2 Cache file variants
### 7.3 Contract enforcement + acceptance
- Stage B validates cached panels against required/forbidden columns (see [src/features/canonical_feature_cols.py](src/features/canonical_feature_cols.py)).
- Extra columns are **allowed** and logged as drift, never auto‑deleted.
- This enables schema evolution without forcing live regeneration.

---

## 8) Phase 2 v2 stateful training (detailed steps)

Phase2 v2 expects a cached panel and will fail if none is used (unless override is enabled in code):
- Merged: cache/features/<SYMBOL>_h<H>_merged.parquet
- Provenance: cache/features/<SYMBOL>_h<H>_merged.meta.json (role maps, family maps)
- Split: cache/features/<SYMBOL>_h<H>_features.parquet + cache/features/<SYMBOL>_h<H>_index.parquet
- Split variant: cache/features/<SYMBOL>_h<H>__features.parquet + cache/features/<SYMBOL>_h<H>__index.parquet

**Data consumption flow:**
1. **Mamba/Phase2 training** → `StageBPipeline._build_panel()` reads merged parquet from `cache/features/`
2. **Portfolio engine** → `RoleAwareContext` reads merged parquet + `.meta.json` for role/family routing

### 8.1 Entry point
- CLI runner: [tools/run_stage_b_stateful_phase2.py](tools/run_stage_b_stateful_phase2.py)
- Core engine: [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py)

### 8.2 Panel load (cached only)
- `StageBPipeline` is created by `_build_stage_b_pipeline()` in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py#L971).
- It loads cached panels from `cache/features/` (via `FEATURE_PANEL_DIR` in pipeline.py).
- Provenance (`.meta.json`) provides role maps for portfolio overlays.
- If no cached artifact is used, Phase2 v2 raises an error (unless PHASE2_ALLOW_LIVE_PANEL_FALLBACK=1 is set).

### 8.3 Labels
- Labels are built with `_construct_labels()` from `StageBPipeline` for the horizon.
- The label column is chosen via `_phase2_label_id_from_cfg()` and `_phase2_label_column()` in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py).

### 8.4 Track‑A/B/C construction (pooled, multi‑symbol)
Workflow in `_build_trackc_multi_symbol()` in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py#L1476):
1) Build per‑symbol panel and labels (from cached merged parquet).
2) Pool train rows across all symbols.
3) Run 3‑pillar dim selection (PCA variance target) and cache results under artifacts/three_pillar_cache.
4) Build Track‑A (family encoders), Track‑B (block summaries), Track‑C (final feature matrix).
5) Align Track‑C schema across symbols (union of columns).

### 8.4.1 3‑pillar dimensionality analyzer
Module: [src/stage_b/optuna_optimizer.py](src/stage_b/optuna_optimizer.py) (`ThreePillarAnalyzer` class)

Computes optimal latent dimensions for each family using 3 pillars:
1. **PCA variance target** (primary): find min k s.t. explained_variance ≥ 95%
2. **AE reconstruction** (fallback): use elbow detection on reconstruction loss
3. **Clamping**: enforce dim bounds (default 4–32) to avoid under/over‑specification

Cache location: `artifacts/three_pillar_cache/*.pkl`
Cache key: symbols + horizon + train range + Stage‑A weights snapshot

Config params:
- `use_three_pillar_dims`: bool (default True)
- `three_pillar_dim_min`: 4
- `three_pillar_dim_max`: 32
- `three_pillar_pca_variance`: 0.95

### 8.5 Standardization and master arrays
Workflow in `_prepare_phase2_for_cfg()` and `_prepare_phase2_once()` in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py#L5071):
1) Compute pooled scaler stats on Track‑C train data.
2) Standardize full Track‑C (train + OOS) for each symbol.
3) Build master arrays (X, y, ts) used for walk‑forward training.
4) Optionally place masters on GPU with `MultiSymbolGPUMasterStore`.

### 8.6 Mamba training and stateful walk‑forward inference
Workflow in `evaluate_phase2_stateful_once()` in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py#L2155):
1) Train Mamba for the fold update on pooled sequences.
2) Perform stateful OOS inference with rolling buffer per symbol.
3) Produce predictions `mu_hat`, `sigma_hat`, `p_up`, `rho` in a preds dataframe.

Training implementation details (model + optimizer) live in [src/stage_b/sequence_models.py](src/stage_b/sequence_models.py):
- Gaussian NLL head is enforced in Phase2 v2 for uncertainty output.
- Uses bf16 autocast on supported CUDA devices; GradScaler is not used for bf16.
- Early stopping with patience and best‑val checkpoint restore.
- Synchronizes CUDA before returning model to avoid cross‑trial GPU spillover.

### 8.6.1 Mamba architecture details
Module: [src/stage_b/sequence_models.py](src/stage_b/sequence_models.py)

Core Mamba hyperparameters (tuned by Optuna):
- `mamba_d_model`: model dimension (128, 256)
- `mamba_n_layers`: number of Mamba layers (3–6)
- `mamba_ssm_dim`: state space model dimension (32, 64, 96)
- `mamba_expand_factor`: expansion factor for inner dim (2.0, 4.0)
- `mamba_seq_len`: sequence length for input windows (127, 159, 191, 223)
- `mamba_activation`: activation function (default: silu)
- `mamba_norm_type`: normalization layer (default: rmsnorm)
- `mamba_norm_strategy`: pre/post layer normalization

Prediction head (fixed for uncertainty):
- `mamba_head_type`: fixed to "gaussian" for mean+variance output
- `mamba_loss_fn`: fixed to "gaussian_nll" for probabilistic training
- `mamba_head_hidden_dim`: hidden dimension for head (64–256)
- `mamba_head_num_layers`: number of head layers (1–3)

### 8.6.2 GPU master store
Module: [src/stage_b/sequence_models.py](src/stage_b/sequence_models.py) (`MultiSymbolGPUMasterStore` class)

Efficiently manages per‑symbol master arrays on GPU:
- Pre‑allocates X, y, ts tensors for each symbol
- Enables fast maturity‑gated replay without CPU→GPU transfer per step
- Used by Phase2 v2 for online walk‑forward updates

### 8.7 Portfolio backtest (training-only)
This is the training‑only portfolio engine. It does **not** place live trades.
Backtest logic and signal rules live in [src/stage_b/backtest.py](src/stage_b/backtest.py).

### 8.8 Portfolio overlays and governance controls (v2 engine)
The v2 engine applies HF‑grade overlays and constraints in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py):
- Group caps: sector/group limits via [src/stage_b_stateful/group_map.py](src/stage_b_stateful/group_map.py)
- Beta neutralization: uses portfolio returns or market proxy
- Liquidity constraints: ADV‑based caps using microstructure turnover
- Drawdown throttles + kill switches
- Volatility throttles + kill switches
- Turnover caps + weight smoothing
- Optional trade delay (execution realism for backtest only)
- **Covariance‑aware sizing**: EWMA covariance matrix with shrinkage for position sizing

### 8.8.1 Group map builder
Module: [src/stage_b_stateful/group_map.py](src/stage_b_stateful/group_map.py)
- `build_symbol_group_map_from_eodhd_cache()`: builds symbol→group (GicSector/Sector) map
- Reads from local EODHD cache first (`data/cache/eodhd/*.json`)
- Auto‑fetches missing symbols from EODHD API if `auto_fetch_missing=True`
- Output: `artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv` (schema: symbol,group)

Tool: [tools/build_phase2_group_map_from_eodhd.py](tools/build_phase2_group_map_from_eodhd.py)

### 8.8.2 Covariance‑aware sizing
The v2 engine uses online covariance estimation for position sizing:
- EWMA covariance update: `cov ← λ·cov + (1−λ)·(r·r^T)` (see `_update_ewma_cov()`)
- Shrinkage toward diagonal: `shrink_cov()` with configurable alpha
- Beta estimation: `_estimate_beta_vector()` using covariance ratio to market proxy
- Config params: `phase2_cov_ewma_lambda` (0.90–0.99), `phase2_shrinkage_alpha` (0.0–0.30)

### 8.9 Universe registry + delisting controls
The v2 engine can gate symbols and cap historical panels using:
- Universe registry parquet (authoritative listing status) in [dagster_prep_families/assets.py](dagster_prep_families/assets.py)
- Delisting registry cache from EODHD (optional) in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py)

These controls:
- prevent “zombie” symbols from entering panel build
- cap panel history at stop_date/delist_date
- can run strict (fail‑fast) or fail‑open, depending on config

### 8.10 Diagnostics and telemetry
Phase2 v2 emits:
- per‑trial GPU memory snapshots
- fold‑level scores and prune metadata
- optional JSON diagnostics under artifacts/stage_b_stateful/diagnostics

---

## 8.11 Governance and safety environment flags
Important runtime flags and controls (all used in code):
- PHASE2_ALLOW_LIVE_PANEL_FALLBACK (Phase2 v2 cached‑only policy)
- PHASE2_ALLOW_PREP_FAMILIES (allows Phase2 to trigger prep_families)
- STAGE_B_PREFER_SPLIT_PANEL (forces split panel selection)
- STAGE_B_REQUIRE_ALL_FAMILIES (strict family coverage enforcement)

These are enforced in [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py) and [src/stage_b/pipeline.py](src/stage_b/pipeline.py).

---

## 8.12 Maturity gating (leakage prevention)
Critical for walk‑forward correctness: labels for day t are only known at t+H.
- Module: [src/stage_b_stateful/maturity.py](src/stage_b_stateful/maturity.py)
- `is_matured(sample_day, now_day, H)`: returns True iff sample_day is label-matured as of now_day
- `maturity_cutoff(now_day, H)`: latest sample_day eligible at now_day under horizon H (session‑aware)
- Phase2 uses maturity gating in:
  - model updates (only matured samples are eligible for training)
  - fold‑level scoring (only matured prefixes count toward reported Sharpe)

---

## 8.13 Optuna search space (68+ tunable parameters)
Full parameter documentation: [Phase2_Optuna_Search_Space.md](Phase2_Optuna_Search_Space.md)

### 6.13.1 Parameter categories
| Category | Parameters | Notes |
|----------|-----------|-------|
| Mamba Architecture | 13 | d_model, n_layers, ssm_dim, expand_factor, seq_len, etc. |
| Training | 11 | dropout, optimizer, lr, weight_decay, grad_clip, scheduler, epochs, batch_size |
| Prediction Head | 5 | head_type (fixed: gaussian), loss_fn (fixed: gaussian_nll), hidden_dim, num_layers, head_dropout |
| Phase2 Engine | 5 | update_sessions, replay_days, update_epochs, cov_ewma_lambda, shrinkage_alpha |
| Portfolio Base | 10 | target_vol, max_gross, max_name, max_net, k_spread, k_impact, z_clip, thresholds |
| Overlays (conditional) | 18 | turnover, weight smoothing, group caps, beta neutral, liquidity constraints |

### 6.13.2 Fixed governance parameters (not tuned)
- Track weights: governed by Stage‑A selector
- Family weights: from `artifacts/stage_a/.../family_weights_best.json`
- 3‑pillar dimensions: computed via variance‑based PCA analysis, cached in `artifacts/three_pillar_cache`

### 6.13.3 Key fixed constraints
- `mamba_head_type`: always "gaussian" for uncertainty estimation
- `mamba_loss_fn`: always "gaussian_nll" for probabilistic predictions
- `smoothing_type`: fixed to "none" to enable Track‑C cache reuse

---

## 8.14 Meta‑optimizer (self‑learning search space adaptation)
Module: [src/stage_b/meta_optimizer.py](src/stage_b/meta_optimizer.py)

The meta‑optimizer is a self‑learning layer on top of Optuna:
1. **TrialMemory**: stores trial history and elite trials (top 10%)
2. **AttributionEngine**: computes parameter importance via ΔSharpe/ΔStability correlation
3. **InteractionLearner**: learns non‑linear parameter relationships via LightGBM
4. **SearchSpaceUpdater**: modifies Optuna priors based on learnings

Config in `OptunaConfig`:
- `use_meta_optimizer`: bool (default True)
- `meta_optimizer_history_size`: 200
- `meta_optimizer_elite_percentile`: 0.10
- `meta_optimizer_min_trials`: 20 (before adaptation kicks in)

---

## 8.15 Ray Tune integration (trial‑level parallelism)
When enabled (`use_ray_tune=True`), Phase2 uses Ray Tune for parallel trial execution:
- [src/stage_b/optuna_optimizer.py](src/stage_b/optuna_optimizer.py): `StageBOptunaSearch` wraps Optuna with Ray Tune
- ASHA scheduler for aggressive early stopping of poor trials
- Placement groups for GPU isolation per trial
- `ray_tune_concurrent_trials`: controls how many trials run in parallel

The Ray Tune search space is built in `_build_ray_tune_search_space()` and mirrors the Optuna suggestions.

---

## 8.16 Universe selector (core + satellite dynamics)
Module: [src/stage_b/universe_selector.py](src/stage_b/universe_selector.py)

Phase2 supports dynamic universe management via core/satellite architecture:
- **Core symbols**: anchor set (e.g., SPY, AAPL, MSFT, etc.)—always included
- **Satellite candidates**: event‑driven additions (earnings volatility, macro cyclicals, etc.)

`UniverseSelectorConfig` controls:
- `satellite_min` / `satellite_max`: bounds on active satellites (default 8–15)
- `rebalance_every_sessions`: selection cadence (default 21 sessions)
- `min_stay_sessions` / `max_stay_sessions`: satellite retention rules
- `decay_fraction` / `remove_abs_threshold`: removal eligibility thresholds

The selector uses event scores (earnings, transcript, news, vol, beta) to rank and select satellites.

---

## 8.17 Feature roles & portfolio‑engine pairing

Phase2/portfolio overlays use role‑aware feature routing derived from provenance and family metadata.

Key modules:
- Feature roles and family intent: [src/features/feature_roles.py](src/features/feature_roles.py)
- Canonical family metadata registry: [src/features/family_metadata.py](src/features/family_metadata.py)
- Role‑aware portfolio context loader: [src/portfolio/role_aware_context.py](src/portfolio/role_aware_context.py)
- Centralized cache paths: [src/cache_paths.py](src/cache_paths.py)

Produced artifacts (from prep_families):
- cache/features/<SYMBOL>_h<H>_merged.meta.json

How roles are assigned:
1) Per‑column role map is read from the provenance JSON.
2) If a column lacks an explicit role, it falls back to family intent from the metadata registry.
3) Usage flags (`risk_scale_ok`, `gating_ok`, `veto_ok`) are inferred conservatively.

Portfolio overlay mapping:
- predictive → alpha direction signals
- risk → risk scaling / sigma modifiers
- regime → exposure modulation
- hygiene → veto/gating

Role‑aware context outputs (per day):
- hygiene_ok (bool)
- risk_scale (float in (0,1])
- regime_multiplier (float in [0,1])

These are consumed by the portfolio/backtest engine via role‑aware context in [src/portfolio/role_aware_context.py](src/portfolio/role_aware_context.py).

---

## 9) Prediction tape export (optional for Stage C)

Stage B can export prediction tapes using [src/stage_b/stage_b_export.py](src/stage_b/stage_b_export.py).
Outputs:
- artifacts/prediction_tapes/<SYMBOL>_h<H>_predictions.parquet

These tapes are used by Stage C policy tooling for threshold sweeps and walk‑forward threshold training (research only).

### 7.1 Backtest workflow
Workflow in `BacktestEngine.run()` in [src/stage_b/backtest.py](src/stage_b/backtest.py):
1) Convert predictions into weights via signal rule (prob/hybrid/z‑score).
2) Apply leverage caps, exposure, turnover costs, and slippage.
3) Produce equity curve and metrics (sharpe, max_drawdown, turnover, etc.).

### 9.1 Benchmarking framework (performance attribution)

The benchmarking framework provides risk-adjusted performance metrics relative to market benchmarks.

Core modules:
- [src/analytics/benchmarking.py](src/analytics/benchmarking.py): `BenchmarkFramework`, `compute_benchmark_timeseries()`, `summarize_benchmark()`
- [src/analytics/eodhd_benchmark_data.py](src/analytics/eodhd_benchmark_data.py): `EODHDBenchmarkSpec`, `fetch_eodhd_adjusted_close()`, `prices_to_returns()`

`BenchmarkFramework` defines:
- **Primary benchmark**: SPY (or equivalent total-return index)
- **Secondary benchmarks**: optional (equal-weight universe, sector-neutral)
- **Risk-free rate**: used for excess-return Sharpe/Sortino

Computed rolling metrics:
- Rolling beta (covariance / variance)
- Rolling alpha (active return)
- Rolling information ratio (alpha / tracking error)
- Rolling Sharpe/Sortino (excess return / vol)
- Rolling tracking error (std of active returns)

### 9.2 Benchmark tooling (CLI utilities)

| Tool | Purpose |
|------|---------|
| [tools/benchmark_backtest.py](tools/benchmark_backtest.py) | Run benchmark comparison on bt_equity output |
| [tools/realtime_benchmark_tracker.py](tools/realtime_benchmark_tracker.py) | Live tracking of portfolio vs benchmark |
| [tools/plot_benchmark_dashboard.py](tools/plot_benchmark_dashboard.py) | Generate PNG dashboards (NAV, drawdown, beta, alpha, IR) |
| [tools/compare_horizon_benchmarks.py](tools/compare_horizon_benchmarks.py) | Compare benchmark stats across horizons |

Outputs:
- `bt_benchmark_timeseries.parquet`: daily benchmark comparison timeseries
- `bt_benchmark_summary.json`: aggregate metrics (total return, beta, IR, tracking error)
- PNG dashboards: NAV curve, drawdown, rolling beta/alpha/IR charts

### 9.3 Global multi‑symbol prediction tapes (Optuna global mode)
When using global pooled training, prediction tapes can also be generated in the Stage‑B optimizer pipeline:
- [src/stage_b/optuna_optimizer.py](src/stage_b/optuna_optimizer.py)
- Function `generate_prediction_tapes_multi_symbol_mamba(...)` builds Track‑C per symbol, trains a shared Mamba per fold, and emits per‑symbol fold tapes.

---

## 10) Stage C policy (research backtest, no live execution)

Stage C is a research‑only policy backtest layer. It consumes prediction tapes and produces threshold policies and equity curves.

### 10.1 Stage C v2 (walk-forward + ML thresholds)
Module: [src/stage_c/stage_c_policy_v2.py](src/stage_c/stage_c_policy_v2.py)
CLI: [tools/run_stage_c_v2.py](tools/run_stage_c_v2.py)

- Static global threshold search
- Walk-forward threshold training
- ML meta-learner for dynamic thresholding
- Confidence-based trading rules

---

## 11) Data sources and providers

### 11.1 EODHD provider
Module: [src/data_sources/eodhd_provider.py](src/data_sources/eodhd_provider.py)

Primary data source for:
- Historical prices (adjusted/unadjusted)
- Fundamentals (income, balance sheet, cash flow)
- Dividends and splits
- Options chains
- Economic indicators
- Delisting data

---

## 12) Monitoring and audit tools

### 12.1 Phase2 Optuna monitor
Tool: [tools/monitor_phase2_stateful_optuna.py](tools/monitor_phase2_stateful_optuna.py)

Real-time monitoring of Phase2 Optuna runs:
- Checks PID and process health
- Logs study progress
- Reports best trial metrics

### 12.2 Audit suite

| Tool | Purpose |
|------|---------|
| [tools/audit_phase2_data_readiness.py](tools/audit_phase2_data_readiness.py) | Validates TrackC existence, group map, advanced inputs |
| [tools/audit_phase2_family_coverage.py](tools/audit_phase2_family_coverage.py) | Checks family coverage, 3-pillar compression, Stage-A weights |
| [tools/audit_mamba_trackc_inputs.py](tools/audit_mamba_trackc_inputs.py) | Verifies TrackC has all inputs needed for Mamba |
| [tools/audit_trackc_columns.py](tools/audit_trackc_columns.py) | Audits TrackC column schemas across symbols |
| [tools/audit_trackc_cache.py](tools/audit_trackc_cache.py) | Audits TrackC cache panels: rows/cols, NaN rates |
| [tools/audit_merged_parquet_quality.py](tools/audit_merged_parquet_quality.py) | Validates merged parquet quality and schema |

---

## 13) n8n workflow blueprint (training‑only)

This is the workflow map you can mirror in n8n. Each node should include its inputs, outputs, and artifacts.

1) **Dagster: prep_families_manifest**
   - Input: symbol, horizon, family selector, wf window
   - Output: merged parquet + provenance + completeness manifest

2) **Cache Validation Gate**
   - Input: merged parquet
   - Output: validation report (pass/fail + drift warnings)
   - Logic: contract enforcement in [src/stage_b/pipeline.py](src/stage_b/pipeline.py)

3) **Phase2 v2 Prepare**
   - Input: cached panel, config
   - Output: Track‑C, scaler stats, three‑pillar cache
   - Location: [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py)

4) **Phase2 v2 Train + Walk‑Forward**
   - Input: Track‑C, labels, hyperparameters
   - Output: model weights, predictions, portfolio metrics

5) **Backtest & Metrics**
   - Input: predictions
   - Output: equity curve + metrics (training‑only)
   - Location: [src/stage_b/backtest.py](src/stage_b/backtest.py)

6) **Prediction Tape Export (optional)**
   - Output: artifacts/prediction_tapes/*.parquet
   - Location: [src/stage_b/stage_b_export.py](src/stage_b/stage_b_export.py)

7) **Stage C Policy (optional research)**

8) **Audit/Readiness Checks (optional but recommended)**
   - Audit Track‑C readiness and overlays: [tools/audit_phase2_data_readiness.py](tools/audit_phase2_data_readiness.py)
   - Data readiness summary for reference: [DATA_READINESS_SUMMARY.md](DATA_READINESS_SUMMARY.md)
   - Track‑C audit and rebuild utilities are referenced in [docs/stateful_mamba_phase2_controls_and_backtest.md](docs/stateful_mamba_phase2_controls_and_backtest.md)
   - Track‑C schema/metric audit: [tools/audit_mamba_trackc_inputs.py](tools/audit_mamba_trackc_inputs.py)
   - Track‑C rebuild from caches (incl. hf_agg): [tools/rebuild_trackc_from_cache.py](tools/rebuild_trackc_from_cache.py)
   - Input: prediction tape
   - Output: threshold policy & research backtest results

---

## 14) Artifact matrix (what is written where)

| Stage | Artifact | Location | Producer |
|------|----------|----------|----------|
| Prep | Per‑family caches | data/local_cache/<symbol>_h<h>/... | prep_families |
| Prep | Merged panel | cache/features/<SYMBOL>_h<H>_merged.parquet | prep_families (Dagster) |
| Prep | Provenance JSON | cache/features/<SYMBOL>_h<H>_merged.provenance.json | prep_families |
| Prep | Provenance columns map | cache/features/<SYMBOL>_h<H>_merged.provenance.columns.csv | prep_families |
| Prep | Completeness manifest | artifacts/prep_families/<symbol>_h<h>_completeness.json | prep_families |
| Prep | Track‑C panel | cache/features/<SYMBOL>_h<H>_trackc.parquet | prep_families / rebuild_trackc_from_cache |
| Prep | Track‑C meta | cache/features/<SYMBOL>_h<H>_trackc.meta.json | prep_families / rebuild_trackc_from_cache |
| Phase2 | 3‑pillar cache | artifacts/three_pillar_cache/*.pkl | phase2_stateful |
| Phase2 | Optuna DB | artifacts/optuna_studies/*.db | phase2_stateful |
| Phase2 | Winner JSON | artifacts/optuna/*.json | phase2_stateful |
| Stage B | Prediction tape | artifacts/prediction_tapes/*.parquet | stage_b_export |
| Stage C | Policy results | artifacts/backtests/… | stage_c_policy |
| Phase2 | Group map CSV | artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv | group_map / run_stage_b_stateful_phase2 |
| Phase2 | Universe selector state | artifacts/meta_optimizer/universe_selector_state_h{H}.json | universe_selector |
| Phase2 | Delisting registry | data/cache/eodhd/delisted_companies_US.parquet | delisting_meta |
| Phase2 | Universe registry | data/cache/universe/universe_registry_YYYYMMDD.parquet | universe_registry |

---

## 15) Stage C v2 policy (research‑only, detailed)

Stage C v2 is a research‑only policy layer. It consumes prediction tapes and performs:
- static threshold search
- walk‑forward threshold training
- optional meta‑learner training for dynamic thresholds

See [src/stage_c/stage_c_policy_v2.py](src/stage_c/stage_c_policy_v2.py) for details.

---

## 16) Training‑only posture (explicitly no live execution)

- Phase2 v2 uses cached panels and backtests predictions into a simulated portfolio.
- No live order routing or execution system is present in the repo; all outputs are training‑only artifacts.

---

## 17) Next steps (if you want this as the repo's master workflow doc)

1) Decide the single tracking folder name (example: v2_stateful_mamba/).
2) Create a curated index README in that folder that links to this doc and the code entrypoints.
3) Export n8n workflow JSON using the node map above.
