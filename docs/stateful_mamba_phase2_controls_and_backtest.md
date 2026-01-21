# Stateful Mamba Phase2 — Full Controls/Heads List + Tuning Space + Backtest Engine

This document is the **authoritative, code-derived** reference for:

1. What `stage_b_stateful` Phase2 has access to (all controls/knobs).
2. Which parameters Phase2 actually tunes (and their ranges) in each mode.
3. What “heads” exist in the Mamba model and what outputs they produce.
4. How the backtest engine works (signals → positions → costs → metrics).

Scope note:
- Phase2 has two execution paths:
  - **Walk-forward stateful engine (v2)**: the default Phase2 portfolio engine.
  - **Legacy “train once + stateful predict + per-symbol backtest”**: still present as a fallback path.

December 2025 state (important):
- Phase2 can be run **v2-only** via `phase2_require_v2: true`.
- Phase2 v2 **forces** the Mamba distribution head/loss to **Gaussian uncertainty + Gaussian NLL** during walk-forward updates (HF-grade comparability).
- Track-C “single parquet per symbol” was hardened with audits + rebuild tooling, including `hf_agg_*` meta columns and canonical `*_score_raw`.

---

## Entry points / where to look

- CLI runner: [tools/run_stage_b_stateful_phase2.py](../tools/run_stage_b_stateful_phase2.py)
- Phase2 core: [src/stage_b_stateful/phase2_stateful.py](../src/stage_b_stateful/phase2_stateful.py)
- Mamba model + training/inference: [src/stage_b/sequence_models.py](../src/stage_b/sequence_models.py)
- Vectorized backtest engine: [src/stage_b/backtest.py](../src/stage_b/backtest.py)
- Global tuning config (bounds/choices): [src/stage_b/optuna_optimizer.py](../src/stage_b/optuna_optimizer.py)

---

## Phase2 date blocks / evaluation universe

**Default time blocks** (Phase2 hard-coded defaults):
- Train block: `2005-07-02 .. 2020-12-31`
- Tune OOS block: `2021-01-01 .. 2023-12-31`
- Holdout block: `2024-01-01 .. 2025-06-20`

**Default symbol universe**:
- If you don’t pass `--symbols`, Phase2 evaluates the fixed `GLOBAL_OPTUNA_SYMBOLS` ("GLOBAL13") in [src/stage_b/pipeline.py](../src/stage_b/pipeline.py).

---

## Universe registry (zombie-symbol prevention)

Phase2 supports an **authoritative universe registry parquet** that prevents “zombie” or delisted symbols from entering:
- panel build,
- training updates,
- daily inference,
- and final portfolio weights (drop + re-apply constraints / renormalize behavior).

**Registry file** (default path):
- [data/cache/universe/universe_registry.parquet](../data/cache/universe/universe_registry.parquet)

**How to build/refresh**:
- Dagster asset: `universe_registry` (group `universe`) in [dagster_prep_families/assets.py](../dagster_prep_families/assets.py)
- CLI helper: [tools/build_universe_registry.py](../tools/build_universe_registry.py)

**Phase2 config keys** (in `runtime_overrides` or trial cfg dict):
- `phase2_universe_registry_enabled`: bool (default `true`; fail-open if registry missing unless strict)
- `phase2_universe_registry_path`: optional path override
- `phase2_universe_registry_strict`: bool (if `true`, missing registry file is a hard error)

Behavior summary:
- Panel build: symbols are filtered before any feature/panel work, and each panel is capped to `stop_date`.
- Updates: training samples for walk-forward updates are filtered to eligible symbols.
- Inference: predictions for ineligible symbols are zeroed before sizing.
- Weights: ineligible symbols are zeroed and the engine logs what was dropped.

Runtime universe mode (portfolio trading set):
- `phase2_runtime_universe_mode`: default `full`
  - `full`: trade the full eligible set daily (no core/satellite rotation)
  - `core_satellite`: monthly-ish rotation using event score + optional peer snapshot gating

---

## Optuna driver parameters (Phase2 run-time knobs)

These are the parameters to `run_phase2_stateful_optuna(...)` in [src/stage_b_stateful/phase2_stateful.py](../src/stage_b_stateful/phase2_stateful.py).

Core run controls:
- `n_trials` (required)
- `search_spec.mode` (default `refinement`; allowed `{refinement, full}`)
- `study_name` (default `stage_b_stateful_phase2`)
- `study_db_path` (default: `artifacts/optuna_studies/{study_name}.db`)
- `export_winner_path` (default: `artifacts/optuna/GLOBAL13_h{H}_phase2_stateful_winner.json`)

Data/time controls:
- `train_start/train_end/oos_start/oos_end` (defaults to the fixed Phase2 blocks)
- `holdout_start/holdout_end` (default `2024-01-01 .. 2025-06-20`; optional)

Universe and portfolio aggregation:
- `symbols` (default: `GLOBAL_OPTUNA_SYMBOLS`)
- `portfolio_weights` (optional; overrides equal-weight)
- `aggregation_rule` (default `equal_weight`)

Pruning + efficiency controls:
- `no_prune` (default `False`; disables pruning)
- `prune_update_sessions` (default `21`; fold size and update cadence anchor)
- `prune_warmup_folds` (default `12`; don’t Hyperband-prune before this fold)
- `safety_prune_after_folds` (default `12`; don’t safety-prune before this fold)
- `safety_prune_sharpe_floor` (default `0.0`)
- `safety_prune_maxdd_ceiling` (default `0.12`)

Sampler controls:
- `tpe_top_fraction` (default `0.15`; clipped to `[0.05, 0.5]`)

Other:
- `deterministic_all` (default `False`; forces more determinism at some throughput cost)
- `runtime_overrides` (optional; escape hatch for ad-hoc overrides)

---

## Phase2 modes: what gets tuned

Phase2 runs Optuna with `--search-mode {refinement|full}`.

### Mode A — `refinement` (small, low-overfit search)
**Definition:** load a Stage-B “best trial” bundle JSON, freeze *everything*, and only tune 5 knobs.

**Tuned parameters (exactly 5):**
- `mamba_seq_len` (categorical; see “Seq len categorical pinning” below)
- `mamba_learning_rate` (log-uniform, `3e-4 .. 2.5e-3`)
- `mamba_dropout` (uniform, `0.0 .. 0.25`)
- `mamba_head_dropout` (uniform, `0.0 .. 0.40`)
- `mamba_grad_clip` (categorical, `{0.25, 0.5, 1.0, 2.0, 4.0}`)

Everything else is frozen to whatever is in the best-trial bundle JSON.

### Mode B — `full` (broad Stage-B space, evaluated statefully)
**Definition:** sample a broad Stage-B config (family weights/dims + thresholds + Mamba hparams) *and* sample Phase2 walk-forward risk engine knobs.

In `full`, Phase2 samples the following **exhaustive Optuna parameter key set**.

Important behavior notes:
- Phase2 full mode currently constructs an `OptunaConfig(..., sequence_model_type="mamba", use_three_pillar_dims=True)`.
- This means per-family dim keys are still **sampled** (stable keyset), but for most families they are effectively **pinned** via the 3-pillar precomputed dims/method.
- Phase2 v2 forces `mamba_head_type="gaussian"` and `mamba_loss_fn="gaussian_nll"` during evaluation; these are still sampled as categoricals to keep Optuna’s distributions stable across studies.

#### 1) Pipeline / Track-C controls
- `max_total_dims`: integer, pinned to `500` (from `OptunaConfig.max_total_dims`)
- `track_a_weight`: pinned to `1.0` (Phase2 parity rule)
- `track_b_weight`: float in `[0.5, 1.5]` (from `OptunaConfig.track_b_weight_min/max`)
- `smoothing_type`: categorical in `{none, ema, sma, gaussian}`
- `smoothing_window`: integer in `[3, 21]` with `step=3`
- `train_fraction`: float in `[0.6, 0.9]`

#### 2) Stage-A families (expanded) — weights + dim keys
Stage-A families (`STAGE_A_FAMILIES` in [src/stage_b/optuna_optimizer.py](../src/stage_b/optuna_optimizer.py)):

`alternative_signals`, `cboe_term`, `correlation`, `cross_asset`, `dcf`, `dividends`, `doc_embedding_novelty_hf`, `earnings`, `earnings_transcript_hf`, `fin_g2`, `fin_g3`, `fin_g4`, `fin_g5`, `fin_g6`, `fin_g7`, `finbert`, `garch_iv`, `macro_tst_hf`, `microstructure`, `ml_framework`, `multiasset`, `options`, `options_anchoring`, `regime`, `short_interest`, `subsidiary`, `tft_features`

For every Stage-A family `<fam>`, Phase2 full mode samples (or pins) all of the following keys:

- `weight_<fam>`: float in `[0.0, 1.0]`, then clipped to `>= 0.01` (`OptunaConfig.family_weight_clip_min`)
- `<fam>_dim_type`: categorical in `{pca, ae}` (but **not sampled** when 3-pillar dims are active; then it is fixed)
- `<fam>_pca_dim`: sampled **only if** `<fam>_dim_type == pca` (step 2) in `[pca_components_min, min(pca_components_max, family_size)]`
- `<fam>_ae_dim`: sampled **only if** `<fam>_dim_type == ae` (step 4) in `[ae_latent_dim_min, min(ae_latent_dim_max, family_size)]`
- `<fam>_ae_layers/_ae_activation/_ae_dropout/_ae_lr`: sampled **only if** `<fam>_dim_type == ae` (otherwise populated with safe defaults, but not sampled)
- `<fam>_dim`: derived convenience alias (`<fam>_pca_dim` if dim_type==pca else `<fam>_ae_dim`)

#### 3) Track-B weights (expanded) — `weight_b_*`
Phase2 full samples Track-B per-family/block weights via the Stage-B helper.

HF block families (`HF_BLOCK_FAMILIES`):
- `weight_b_tech_micro_hf`
- `weight_b_forecast_hf`
- `weight_b_vol_deriv_hf`
- `weight_b_macro_regime_hf`
- `weight_b_fundamental_val_hf`
- `weight_b_news_nlp_hf`

Track-B summary blocks (`TRACK_B_SUMMARY_BLOCKS`):
- `weight_b_quantile`
- `weight_b_calibration`
- `weight_b_online`
- `weight_b_arima`

For each of these `weight_b_*` keys:
- float in `[0.5, 1.0]` (from `OptunaConfig.stage_b_family_min_weight` to `OptunaConfig.family_weight_max`), then clipped to `>= 0.01`

#### 4) Threshold / gating parameters (Phase2 v2 regime thresholds)
Sampled directly inside Phase2 full objective (v2-only):
- `regime_threshold_base` in `[0.02, 0.20]`
- `regime_threshold_bull_mult` in `[0.6, 1.2]`
- `regime_threshold_bear_mult` in `[0.8, 2.0]`
- `regime_threshold_crisis_mult` in `[1.0, 4.0]`
- `conf_threshold` in `[0.3, 0.8]`
- `vol_scaler` in `[0.0, 1.0]`

Phase2 also injects legacy keys (not sampled) for compatibility with older diagnostics/backtest paths:
`threshold`, `bull_mult`, `bear_mult`, `crisis_mult` (mirrors the v2 regime-threshold values).

#### 5) Mamba hyperparameters
Sampled directly inside Phase2 full objective:

- `mamba_d_model`: categorical in `{96, 128, 160, 192, 224, 256, 288}`
- `mamba_n_layers`: categorical in `{3, 4, 5, 6, 8}`
- `mamba_ssm_dim`: categorical in `{64, 96, 128, 160}`
- `mamba_expand_factor`: categorical in `{2.0, 2.5, 3.0, 4.0}`
- `mamba_seq_len`: categorical (see “Seq len categorical pinning” below)
- `mamba_activation`: categorical in `{silu, gelu}`
- `mamba_norm_type`: categorical in `{rmsnorm, layernorm}`
- `mamba_norm_strategy`: categorical in `{pre, post}`
- `mamba_dropout`: float in `[0.05, 0.30]`
- `mamba_resid_dropout`: categorical in `{0.0, 0.05, 0.10}`
- `mamba_ssm_dropout`: categorical in `{0.0, 0.02, 0.05}`
- `mamba_gate_dropout`: categorical in `{0.0, 0.02, 0.05}`
- `mamba_optimizer`: categorical in `{adamw, lion}`
- `mamba_learning_rate`: log-uniform float in `[1e-4, 3e-3]`
- `mamba_weight_decay`: **conditional**
  - if `mamba_optimizer == lion`: log-uniform float in `[1e-8, 1e-4]` (tight range)
  - else: log-uniform float in `[mamba_weight_decay_min, mamba_weight_decay_max]` (defaults `1e-6..1e-2`)
- `mamba_grad_clip`: categorical in `{0.5, 1.0, 2.0}`
- `mamba_lr_scheduler`: categorical in `{cosine, one_cycle, linear_warmup_cosine}`
- `mamba_warmup_steps`: sampled **only if** `mamba_lr_scheduler == linear_warmup_cosine` (otherwise fixed to `0`)
- `mamba_max_epochs`: categorical in `{8, 10, 12, 15}`
- `mamba_batch_size`: categorical in `{16, 32, 48, 64}`

Uncertainty head / loss keys (sampled, but forced in v2 evaluation):
- `mamba_loss_fn`: categorical; **new studies** pin this to `{gaussian_nll}`. Existing studies may have broader choice sets, but v2 evaluation overrides to `gaussian_nll`.
- `mamba_head_type`: categorical; **new studies** pin this to `{gaussian}`. Existing studies may have broader choice sets, but v2 evaluation overrides to `gaussian`.

Head MLP meta-keys (always sampled for stable keyset, even when head is forced):
- `mamba_head_hidden_dim`: categorical in `{64, 96, 128, 160, 192, 224, 256}`
- `mamba_head_num_layers`: categorical in `{1, 2, 3}`
- `mamba_head_dropout`: float in `[0.0, 0.3]`

#### 6) Phase2 walk-forward portfolio / risk engine controls
These are sampled only in Phase2 full mode (inside [src/stage_b_stateful/phase2_stateful.py](../src/stage_b_stateful/phase2_stateful.py)):

- `phase2_engine`: categorical pinned to `{v2}`
- `phase2_update_sessions`: categorical pinned to `{prune_update_sessions}` (default 21)
- `phase2_replay_days`: categorical in `{63, 126, 189, 252, 315}`
- `phase2_update_epochs`: categorical in `{1, 2, 3, 4, 5}`
- `phase2_cov_ewma_lambda`: float in `[0.90, 0.99]`
- `phase2_shrinkage_alpha`: float in `[0.0, 0.30]`
- `phase2_target_vol`: float in `[0.08, 0.25]`
- `phase2_max_gross`: float in `[0.5, 2.0]`
- `phase2_max_name`: float in `[0.02, 0.25]`
- `phase2_max_net`: float in `[0.0, 0.30]`
- `phase2_k_spread`: log-uniform float in `[5e-5, 3e-4]`
- `phase2_k_impact`: float in `[0.0, 1e-3]`
- `phase2_z_clip`: categorical in `{4.0, 6.0, 8.0, 10.0}`

Optional overlays are sampled as conditional blocks (to support `TPESampler(group=True)`):
- `phase2_enable_turnover_overlay` -> samples `phase2_turnover_cap` and optionally `phase2_kill_on_turnover_gt`, `phase2_flat_cooldown_sessions`
- `phase2_enable_weight_smoothing` -> samples `phase2_weight_smoothing_alpha`
- `phase2_enable_group_caps` (only if group map exists via runtime overrides) -> samples `phase2_group_max_gross`, `phase2_group_max_net`
- `phase2_enable_beta_neutral` -> sets `phase2_beta_neutral=True` and samples `phase2_beta_max_abs_exposure`, `phase2_beta_lookback_days`
- `phase2_enable_liquidity_constraints` (only if `phase2_capital_usd>0` via runtime overrides) -> samples `phase2_adv_window`, `phase2_max_adv_frac_name`, `phase2_max_turnover_adv_frac`, `phase2_borrow_fee_bps_annual`

### Seq len categorical pinning (important operational detail)
Optuna cannot change categorical choice sets once a study has trials. Phase2 therefore:
- Re-uses the **existing** `mamba_seq_len` choice list if the study already has trials.
- Otherwise, creates a horizon-based choice list: `horizon + 32*k` up to 315, plus 315.

This is why the refinement-mode `mamba_seq_len` space is determined by the **Optuna study** configuration, not only by `Phase2RefinementSpec` defaults.

---

## Full list of Phase2 config keys (the “controls list”)

Below is the complete list of knobs Phase2 **consumes** (i.e., the code reads them and they can change behavior). This list is enforced by a fail-fast audit in Phase2 full mode.

### A) Track-C / feature construction controls
- `max_total_dims` — cap on total Track-A reduced dimensions before Track-C assembly
- `track_a_weight` — global multiplier for Track-A (Phase2 pins to 1.0 in full search)
- `track_b_weight` — global multiplier for Track-B
- `weight_<family>` — Stage-A family weights (soft drop; clipped to a floor)
- `weight_b_<name>` — Track-B family / summary-block weights
- `smoothing_type` — `{none, ema, sma, gaussian}` (gaussian falls back to SMA if unavailable)
- `smoothing_window` — integer

Per-family encoder/dim controls (always present for stability):
- `<family>_dim_type` — `{pca, ae}` (or pinned via 3-pillar)
- `<family>_pca_dim`
- `<family>_ae_dim`
- `<family>_ae_layers`
- `<family>_ae_activation`
- `<family>_ae_dropout`
- `<family>_ae_lr`
- `<family>_dim` (legacy alias)

### B) Training controls
- `train_fraction` — split fraction for train/val within the replay window
- `early_stopping_patience` — used during updates (defaults small in Phase2)

### C) Mamba model controls (architecture + training)
- Architecture: `mamba_d_model`, `mamba_n_layers`, `mamba_ssm_dim`, `mamba_expand_factor`, `mamba_seq_len`
- Activation/normalization: `mamba_activation`, `mamba_norm_type`, `mamba_norm_strategy`
- Dropout: `mamba_dropout`, `mamba_resid_dropout`, `mamba_ssm_dropout`, `mamba_gate_dropout`
- Optimizer/training: `mamba_optimizer`, `mamba_learning_rate`, `mamba_weight_decay`, `mamba_grad_clip`, `mamba_lr_scheduler`, `mamba_warmup_steps`, `mamba_max_epochs`, `mamba_batch_size`
- Head/loss: `mamba_head_type`, `mamba_head_hidden_dim`, `mamba_head_num_layers`, `mamba_head_dropout`, `mamba_loss_fn`

### D) Thresholding / gating controls
There are two related groups of threshold knobs in Phase2:

1) **Stage-B Step-7 thresholds** (used by the backtest engine and legacy Phase2 path)
- `threshold`
- `bull_long_mult`, `bull_short_mult`
- `bear_long_mult`, `bear_short_mult`
- `crisis_long_mult`, `crisis_short_mult`
- `conf_threshold`
- `vol_scaler`

2) **Phase2 walk-forward z-thresholding** (v2 engine)
- `regime_threshold_base` (fallbacks to `threshold` if present; else default 0.10)
- `regime_threshold_bull_mult` (default 0.8)
- `regime_threshold_bear_mult` (default 1.5)
- `regime_threshold_crisis_mult` (default 3.0)

### E) Phase2 v2 portfolio / risk engine controls
- `phase2_engine` (expected `walkforward_v2` / `v2`)
- `phase2_require_v2` (bool; if true, legacy fallback path is disabled and will error)
- `phase2_update_sessions`
- `phase2_replay_days`
- `phase2_update_epochs`
- `phase2_cov_ewma_lambda`
- `phase2_shrinkage_alpha`
- `phase2_target_vol`
- `phase2_max_gross`
- `phase2_max_name`
- `phase2_max_net`
- `phase2_k_spread`
- `phase2_k_impact`
- `phase2_z_clip`

HF-grade runtime overlays (deterministic):
- `phase2_safety_overlays` (bool; default true)
- `phase2_trade_delay_sessions` (int; default 1; >1 adds extra execution delay)

Drawdown throttle + kill-switch:
- `phase2_dd_throttle_1` (default 0.05), `phase2_dd_gross_mult_1` (default 0.7)
- `phase2_dd_throttle_2` (default 0.10), `phase2_dd_gross_mult_2` (default 0.4)
- `phase2_dd_kill` (default 0.25)

Realized-vol throttle + kill-switch:
- `phase2_realized_vol_window` (default 20)
- `phase2_vol_throttle_mult` (default 2.0)
- `phase2_vol_kill_mult` (default 3.0)

Turnover / weight-change controls:
- `phase2_turnover_cap` (L1 turnover budget; enforces $\sum |\Delta w| \le \tau$)
- `phase2_weight_smoothing_alpha` (EMA smoothing on weights; 0 disables)
- `phase2_kill_on_turnover_gt` (optional hard kill if turnover exceeds threshold; 0 disables)
- `phase2_flat_cooldown_sessions` (if kill triggers, remain flat for N sessions)

Exposure controls (sector/industry/cluster + beta):
- `phase2_group_map_path` (optional; JSON dict or CSV with `symbol,group`)
- `phase2_group_max_gross` (cap gross exposure per group; 0 disables)
- `phase2_group_max_net` (cap net exposure per group; 0 disables)
- `phase2_beta_neutral` (bool; neutralize market beta using rolling beta estimates)
- `phase2_beta_max_abs_exposure` (cap on $|\beta^T w|$ after neutralization; 0 disables)
- `phase2_beta_lookback_days` (default 252)

Liquidity / execution realism (optional; requires capital + volume/dollar_volume):
- `phase2_capital_usd` (if unset/0, liquidity constraints are disabled)
- `phase2_adv_window` (default 20)
- `phase2_max_adv_frac_name` (max per-name notional as fraction of ADV; 0 disables)
- `phase2_max_turnover_adv_frac` (max daily turnover notional as fraction of total ADV; 0 disables)
- `phase2_borrow_fee_bps_annual` (optional; daily borrow cost on short notional)

**Track-C liquidity proxy note (Track-C vs “raw volume”):**
- Phase2 ADV computation supports `dollar_volume`, OR `volume*price`, OR Track-C’s derived USD turnover proxy `microstructure_micro_turnover`.
- This makes liquidity constraints feasible even when raw `dollar_volume` is not present in Track-C.

---

## Mamba “heads”: what they are and what outputs they produce

The model used is `MambaLikeRegressor`.

### Supported head types
- `linear` (default): scalar regression head
  - Output: shape `(B,)` = `mu`
- `mlp`: MLP regression head
  - Output: shape `(B,)` = `mu`
- `gaussian` / `uncertainty` / `gaussian_nll`: uncertainty head
  - Output: shape `(B, 2)` = `[mu, log_var]`
  - Interpretation:
    - `var = softplus(log_var) + 1e-6`
    - `sigma = sqrt(var)`

### Important Phase2 behavior
In the **Phase2 v2 walk-forward engine**, uncertainty training is now **explicitly forced**:
- `mamba_head_type = "gaussian"`
- `mamba_loss_fn = "gaussian_nll"`

This is intentional: Phase2 v2 uses $z=\mu/\sigma$ for sizing and risk control. Phase2 full-mode Optuna trials also record this forcing on the trial (`phase2_forced_mamba_*`) so trial payloads remain truthful.

### Quantile heads (clarification)
- There is **no quantile head for Mamba** in this codebase today.
- Quantile (pinball) loss exists for the LSTM path.

---

## Data readiness: “single parquet per symbol” (Track-C)

Phase2 v2 uses **Track-C consolidated parquets** as the single-file input per symbol:
- `cache/features/{SYMBOL}_h{H}_trackc.parquet`

### Audits
- Phase2 readiness audit (Track-C existence + liquidity prereqs + group-map presence):
  - [tools/audit_phase2_data_readiness.py](../tools/audit_phase2_data_readiness.py)
- Mamba Track-C input audit (families + canonical metrics + optional strict schema):
  - [tools/audit_mamba_trackc_inputs.py](../tools/audit_mamba_trackc_inputs.py)

### Rebuild
- Track-C rebuild tool (rebuild consolidated Track-C from cached families, optionally merging `hf_agg_*` meta columns):
  - [tools/rebuild_trackc_from_cache.py](../tools/rebuild_trackc_from_cache.py)

### Canonical HF meta columns
- The `hf_agg` meta family produces `hf_agg_score`, `hf_agg_conf`, and `hf_agg_score_raw`.
- `*_score_raw` is required by the Mamba Track-C audit as a canonical metric column.

---

## Strict schema parity (why it currently fails)

The “strict schema” mode compares each symbol’s parquet columns to a baseline symbol (often AAPL). Strict failures are **not necessarily missing raw data**—many are **symbol-specific feature naming** choices.

Concrete example:
- AAPL (tech) produces `correlation_corr_20_sectorxlk*`.
- XOM (energy) produces `correlation_corr_20_sectorxle*` instead.

To make strict parity pass **without stubbing**, we would need to redesign certain feature families to emit a **fixed, symbol-invariant column set** (e.g., correlations vs a fixed list of sector ETFs for all symbols), and/or ensure optional blocks always emit all columns with defined defaults.

Operational guidance:
- For “does Phase2/Mamba have everything it needs to run correctly?”, the family+metric audit is the correct readiness gate.
- Use strict schema only if you explicitly want pooled-training column equality and are willing to enforce a fixed schema.

---

## Phase2 v2 walk-forward engine (how it works)

This is the **default** Phase2 engine and the main reason Phase2 is “stateful”.

### Step 1 — Build the union OOS calendar
- A single `union_oos_index` is built across the evaluation symbol set.
- Per-symbol returns are aligned onto that index.

### Step 2 — Online update schedule (`phase2_update_sessions`)
- Every `U = phase2_update_sessions` sessions, Phase2 retrains/updates the model.

### Step 3 — Maturity gating (leakage prevention)
For each OOS day `t`:
- Only labels that are **matured** are used.
- Cutoff is `maturity_cutoff(t, H=horizon)` which is effectively `t - horizon` (in trading-session terms).

### Step 4 — Replay window (`phase2_replay_days`)
At update time:
- Phase2 selects training samples whose label timestamps are in `[cutoff - replay_days, cutoff]`.
- It then does a train/val split using `train_fraction`.

### Step 5 — Warm-start updates
- Phase2 carries `warm_state` forward: the updated model is warm-started from the previous state dict.

### Step 6 — Daily inference (mu, sigma)
- For each day, Phase2 predicts per-symbol `mu` and `sigma`.
- It computes per-symbol `z = mu / sigma`, clipped to `±phase2_z_clip`.

### Step 7 — Regime-aware thresholding (sparsify weak z)
- A regime is computed per symbol using rolling mean/vol on returns.
- A per-regime threshold is computed from:
  - `regime_threshold_base`
  - multipliers: `regime_threshold_{bull,bear,crisis}_mult`
- If `|z| < threshold`, z is set to 0 for that asset on that day.

### Step 8 — Covariance-aware sizing (risk control)
- Covariance matrix is updated online via EWMA:
  - `cov ← ewma_update(cov, r_t, lambda=phase2_cov_ewma_lambda)`
- Shrinkage is applied:
  - `cov_shrunk = shrink_to_diag(cov, alpha=phase2_shrinkage_alpha)`
- Raw weights are computed by mean-variance style sizing:
  - `w_raw = pinv(cov_shrunk) @ z_thresholded`

### Step 9 — Portfolio constraints + vol targeting
Weights are then constrained:
- Per-name: `|w_i| <= phase2_max_name`
- Gross: `sum |w_i| <= phase2_max_gross`
- Net: `|sum w_i| <= phase2_max_net`

Then scaled to a target annual vol:
- `phase2_target_vol` (annualized)

### Step 10 — PnL timing and costs
- **PnL for day t uses execution weights** (previous weights) and same-day realized returns.
- Optional additional delay: `phase2_trade_delay_sessions > 1` delays execution beyond the default 1-session convention.
- Turnover: `tval = sum |w_t - w_{t-1}|`
- Costs:
  - Linear spread: `phase2_k_spread * tval`
  - Nonlinear impact: `phase2_k_impact * tval^(1.5)`
  - Optional borrow: `phase2_borrow_fee_bps_annual` charged daily on short notional

HF-grade overlays applied before finalizing weights:
- Drawdown throttles / kill-switch (`phase2_dd_*`)
- Realized-vol throttles / kill-switch (`phase2_vol_*`)
- Turnover budgeting (`phase2_turnover_cap`) and optional smoothing (`phase2_weight_smoothing_alpha`)
- Group exposure caps (`phase2_group_*`) and optional beta neutrality (`phase2_beta_*`)
- Optional ADV-based liquidity caps when `phase2_capital_usd` is provided

### Step 11 — Objective (what Optuna maximizes)
Portfolio score is:
- `score = sharpe(net_returns) − max_drawdown_penalty * max_drawdown − turnover_penalty * mean_turnover`

Defaults:
- `max_drawdown_penalty = 0.25`
- `turnover_penalty = 0.0` (CLI exposed)

### Pruning / intermediate reporting
Phase2 can report intermediate fold scores every `prune_update_sessions` sessions.
- Critically, fold scoring uses **matured prefixes only** (still respects horizon maturity).

**Current default pruner/sampler config (compute-efficient, Phase2 v2)**

Pruner (Hyperband):
- Type: `HyperbandPruner`
- Step unit: **fold** (one fold = `prune_update_sessions` sessions)
- `min_resource = max(1, prune_warmup_folds)` (default `12`)
- `max_resource ≈ ceil(n_oos_sessions / prune_update_sessions)` (computed from the OOS date block)
- `reduction_factor = 3`
- Safety prune (deterministic, runs before Hyperband): after `safety_prune_after_folds` (default `12`), prune if `sharpe < safety_prune_sharpe_floor` AND `max_dd > safety_prune_maxdd_ceiling`.

Sampler (TPE):
- Type: `TPESampler`
- `seed = 1337` (unless `search_spec.seed` overrides)
- `n_startup_trials = 5` (reduces early random-trial waste for expensive trials)
- `gamma = lambda n: max(1, ceil(tpe_top_fraction * n))`
- `tpe_top_fraction = 0.15` by default (stricter “good set”, especially important because pruning biases the completed-trial distribution)
- `multivariate = True` and `group = True` (learns coupled parameter structure; Optuna marks these experimental but they are supported in this repo’s current environment)

---

## Legacy per-symbol backtest path (still used as a fallback)

This path:
1) Trains once on pooled sequences,
2) runs “stateful” rolling-window inference per symbol,
3) then calls the vectorized backtest engine per symbol,
4) finally aggregates into a portfolio.

In this legacy path, Phase2 constructs a minimal `preds_df` with:
- `mu_hat`: from the model (or from rolling inference)
- `sigma_hat`: proxy from rolling std of forward returns (not true model sigma)
- `p_up`: logistic transform of `mu_hat / sigma_proxy`
- `rho`: confidence proxy derived from `p_up`

Then it calls `BacktestEngine.run()`.

If you want HF-style “v2-only” guarantees, set `phase2_require_v2=True` to hard-disable this legacy fallback.

---

## Backtest engine (how it works)

The backtest engine is in [src/stage_b/backtest.py](../src/stage_b/backtest.py).

### Inputs
`BacktestEngine.run(preds_df, horizon, strategy_cfg, symbol=...)` expects `preds_df` columns:
- `mu_hat` (required): mean prediction / score
- `sigma_hat` (optional but used by z-score rules)
- `p_up` (optional; used by prob/hybrid rule and calibration)
- `rho` (optional; used for confidence weighting / gating)
- `actual_return` (optional but important for horizon-aware Sharpe)

Price input:
- A price DataFrame with a close column (`close`, `adj_close`, etc.). Returns are computed as close-to-close pct change.

### Signal construction: two modes
1) **Regime-threshold mode (default if enabled and params exist)**
- Detect regime from returns (`bull`, `bear`, `crisis`).
- Apply regime-dependent thresholds to `mu_hat` (optionally different for long vs short).
- Optional confidence gate: if `rho < conf_threshold`, signal → 0.
- Optional volatility normalization: divide predictions by `(vol20/vol252) ** vol_scaler`.
- Direction is discrete in `{−1, 0, +1}`.

2) **Legacy continuous mode**
- Compute continuous raw signal via `signal_rule`:
  - `zscore`: `tanh(k * mu / |sigma|)`
  - `prob`: `2*(p_up - 0.5)`
  - `hybrid`: average of the two

### Position sizing and overlap logic
- If `confidence_weighting=True`, raw signal is multiplied by `rho`.
- Weights are then clipped:
  - `weights = clip(raw_signal * leverage, −max_exposure, +max_exposure)`
- If `overlap=True` and `holding_period_days > 1`, exposure is an average of the last `holding_period_days` weights.

### Costs
- Turnover: `|Δ weight|`.
- Fee: `fee_bp` is charged when a trade occurs (`turnover > 0`).
- Slippage: `slippage_bp * turnover`.

### Returns
- `gross_return = exposure * daily_return`
- `net_return = gross_return - costs`

### Metrics
Key metrics include:
- `sharpe`
  - For `horizon>1` and when `actual_return` exists: uses **direction * forward_return** sampled **non-overlapping every H days** (prevents Sharpe inflation).
- `max_drawdown` (on equity curve from net returns)
- `turnover` (mean)
- `hit_rate`
- `sortino`
- `ece`, `brier` (requires `p_up` + `actual_return`)
- `stability`, `coverage`, `drift_on/off`

---

## Practical “what do I tune?” guidance

- Use **`refinement`** mode when you trust the best-trial Track-C construction and just want to stabilize/retune training dynamics.
- Use **`full`** mode when you want to re-optimize family weights/dims and Phase2 risk sizing together — but it is more expensive and more overfit-prone.

---

## How to run Phase2 (examples)

Refinement (5-knob search over a frozen best trial):

```bash
python tools/run_stage_b_stateful_phase2.py \
  --search-mode refinement \
  --best-trial-json artifacts/optuna_studies/BEST_TRIAL_trial9_stage_b_global_mamba_h63_1510621193_reset_5ae41193_m_avg_score_focus_80b7b9ff.json \
  --horizon 63 \
  --n-trials 40
```

Full search (broad space):

```bash
python tools/run_stage_b_stateful_phase2.py \
  --search-mode full \
  --horizon 63 \
  --n-trials 100
```

