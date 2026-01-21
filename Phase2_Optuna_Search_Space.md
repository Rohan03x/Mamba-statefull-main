# Phase 2 Stateful Mamba - Optuna Search Space Parameters

## Overview
This document describes all hyperparameters tuned by Optuna in Phase 2 stateful Mamba optimization. The search space includes 68+ parameters across architecture, training, portfolio management, and optional overlays.

---

## 1. MAMBA ARCHITECTURE PARAMETERS

### Core Architecture
| Parameter | Type | Range/Choices | Default | Description |
|-----------|------|---------------|---------|-------------|
| `mamba_d_model` | Categorical | [128, 256] | 128 | Model dimension - hidden size of Mamba layers |
| `mamba_n_layers` | Categorical | [3, 4, 5, 6] | 4 | Number of Mamba layers in the network |
| `mamba_ssm_dim` | Categorical | [32, 64, 96] | 96 | State space model dimension |
| `mamba_expand_factor` | Categorical | [2.0, 4.0] | 2.0 | Expansion factor for inner dimension |
| `mamba_seq_len` | Categorical | [127, 159, 191, 223] | varies | Sequence length for input windows |
| `mamba_activation` | Categorical | ["silu", ...] | "silu" | Activation function |
| `mamba_norm_type` | Categorical | ["rmsnorm", ...] | "rmsnorm" | Normalization layer type |
| `mamba_norm_strategy` | Categorical | ["pre", "post"] | "pre" | Where to apply normalization (pre/post layer) |

### Prediction Head
| Parameter | Type | Range/Choices | Default | Description |
|-----------|------|---------------|---------|-------------|
| `mamba_head_type` | Fixed | "gaussian" | "gaussian" | **Fixed to gaussian for uncertainty estimation** |
| `mamba_loss_fn` | Fixed | "gaussian_nll" | "gaussian_nll" | **Fixed to gaussian NLL loss** |
| `mamba_head_hidden_dim` | Categorical | [64-256 step 32] | 64 | Hidden dimension for prediction head |
| `mamba_head_num_layers` | Categorical | [1, 2, 3] | varies | Number of layers in prediction head |
| `mamba_head_dropout` | Float | [0.0, 0.3] | 0.0 | Dropout rate for prediction head |

---

## 2. TRAINING PARAMETERS

### Regularization
| Parameter | Type | Range/Choices | Default | Description |
|-----------|------|---------------|---------|-------------|
| `mamba_dropout` | Float | [0.0, 0.5] | 0.0 | Main dropout rate for Mamba layers |
| `mamba_resid_dropout` | Categorical | [0.0, ...] | 0.0 | Residual connection dropout |
| `mamba_ssm_dropout` | Categorical | [0.0, ...] | 0.0 | SSM-specific dropout |
| `mamba_gate_dropout` | Categorical | [0.0, ...] | 0.0 | Gate mechanism dropout |

### Optimizer Settings
| Parameter | Type | Range/Choices | Default | Description |
|-----------|------|---------------|---------|-------------|
| `mamba_optimizer` | Categorical | ["adamw", "lion"] | "adamw" | Optimizer type |
| `mamba_learning_rate` | Float (log) | [1e-5, 1e-3] | varies | Learning rate (log scale) |
| `mamba_weight_decay` | Float (log) | [1e-6, 1e-2] or [1e-8, 1e-4] | varies | Weight decay (range depends on optimizer) |
| `mamba_grad_clip` | Categorical | [0.5, 1.0, 2.0] | 1.0 | Gradient clipping threshold |

### Learning Rate Schedule
| Parameter | Type | Range/Choices | Default | Description |
|-----------|------|---------------|---------|-------------|
| `mamba_lr_scheduler` | Categorical | ["none", "linear_warmup_cosine"] | "none" | Learning rate scheduler type |
| `mamba_warmup_steps` | Conditional | [0, ...] | 0 | Warmup steps (only if linear_warmup_cosine) |

### Training Duration
| Parameter | Type | Range/Choices | Default | Description |
|-----------|------|---------------|---------|-------------|
| `mamba_max_epochs` | Categorical | [6, 8, 10, 12] | 10 | Maximum training epochs |
| `mamba_batch_size` | Categorical | [32, 48, 64] | 32 | Training batch size |

---

## 3. TRACK-A/B/C PIPELINE PARAMETERS

### Track Weights (FIXED - Governed by Stage A)
| Parameter | Type | Value | Description |
|-----------|------|-------|-------------|
| `track_a_weight` | Fixed | 1.0 | Weight for Track-A (governance input) |
| `track_b_weight` | Fixed | 1.0 | Weight for Track-B (governance input) |
| `weight_b_{family}` | Fixed | Stage-A | Per-family weights for Track-B |
| `weight_{family}` | Fixed | Stage-A | Per-family weights for Track-C |

### 3-Pillar Dimensionality (FIXED - Governance Level)
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `three_pillar_dim_min` | Fixed | 4 | Minimum dimensions per family |
| `three_pillar_dim_max` | Fixed | 32 | Maximum dimensions per family |
| `three_pillar_pca_variance` | Fixed | 0.95 | Target variance explained by PCA |
| `{family}_dim_type` | Fixed | "pca" | Dimensionality reduction method (fixed to PCA) |
| `{family}_pca_dim` | Fixed | varies | PCA dimensions per family (computed by 3-pillar) |
| `{family}_dim` | Fixed | varies | Final dimensions per family |

### Feature Smoothing (DISABLED for Cache Reuse)
| Parameter | Type | Value | Description |
|-----------|------|-------|-------------|
| `smoothing_type` | Fixed | "none" | **Fixed to "none" to enable cache reuse** |
| `smoothing_window` | N/A | N/A | **Removed from cache signature** |

---

## 4. PHASE 2 ENGINE PARAMETERS

### Walk-Forward Configuration
| Parameter | Type | Range/Choices | Default | Description |
|-----------|------|---------------|---------|-------------|
| `phase2_engine` | Categorical | ["v2"] | "v2" | Engine version |
| `phase2_update_sessions` | Categorical | [21] | 21 | Fixed update cadence (trading sessions) |
| `phase2_replay_days` | Categorical | [63, 126, 189, 252, 315] | varies | Days of history for model retraining |
| `phase2_update_epochs` | Categorical | [1, 2, 3, 4, 5] | varies | Epochs per update during walk-forward |

### Covariance Estimation
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_cov_ewma_lambda` | Float | [0.90, 0.99] | varies | EWMA decay for covariance matrix |
| `phase2_shrinkage_alpha` | Float | [0.0, 0.30] | varies | Shrinkage intensity for covariance |

---

## 5. PORTFOLIO CONSTRUCTION PARAMETERS

### Risk & Volatility Targeting
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_target_vol` | Float | [0.08, 0.25] | varies | Annual volatility target |
| `vol_scaler` | Float | [0.0, 1.0] | varies | Volatility scaling factor |

### Position Sizing Constraints
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_max_gross` | Float | [0.5, 2.0] | varies | Maximum gross exposure (as fraction) |
| `phase2_max_name` | Float | [0.02, 0.25] | varies | Maximum single-name exposure |
| `phase2_max_net` | Float | [0.0, 0.30] | varies | Maximum net exposure |

### Transaction Costs
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_k_spread` | Float (log) | [5e-5, 3e-4] | varies | Spread cost coefficient (log scale) |
| `phase2_k_impact` | Float | [0.0, 1e-3] | varies | Market impact cost coefficient |

### Signal Filtering
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_z_clip` | Categorical | [4.0, 6.0, 8.0, 10.0] | varies | Z-score clipping threshold for signal |
| `conf_threshold` | Float | [0.3, 0.8] | varies | Confidence threshold for trade entry |

### Regime Thresholds
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `regime_threshold_base` | Float | [0.02, 0.20] | varies | Base regime threshold |
| `regime_threshold_bull_mult` | Float | [0.6, 1.2] | varies | Bull regime multiplier |
| `regime_threshold_bear_mult` | Float | [0.8, 2.0] | varies | Bear regime multiplier |
| `regime_threshold_crisis_mult` | Float | [1.0, 4.0] | varies | Crisis regime multiplier |

---

## 6. OPTIONAL OVERLAYS (Conditional Parameters)

### Turnover Overlay
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_enable_turnover_overlay` | Categorical | [False, True] | varies | Enable turnover constraints |
| `phase2_turnover_cap` | Conditional Float | [0.05, 1.0] | N/A | Maximum daily turnover (if enabled) |
| `phase2_enable_turnover_kill_switch` | Conditional Categorical | [False, True] | N/A | Enable kill switch on high turnover |
| `phase2_kill_on_turnover_gt` | Conditional Float | [0.5, 3.0] | N/A | Kill switch threshold (if enabled) |
| `phase2_flat_cooldown_sessions` | Conditional Categorical | [0, 21, 42, 63] | N/A | Cooldown period after kill (if enabled) |

### Weight Smoothing Overlay
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_enable_weight_smoothing` | Categorical | [False, True] | varies | Enable position weight smoothing |
| `phase2_weight_smoothing_alpha` | Conditional Float | [0.05, 0.35] | N/A | Smoothing alpha (if enabled) |

### Group Caps (Requires group_map_path)
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_enable_group_caps` | Conditional Categorical | [False, True] | N/A | Enable sector/group constraints |
| `phase2_group_max_gross` | Conditional Float | [0.05, 0.50] | N/A | Max gross per group (if enabled) |
| `phase2_group_max_net` | Conditional Float | [0.0, 0.30] | N/A | Max net per group (if enabled) |

### Beta Neutralization
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_enable_beta_neutral` | Categorical | [False, True] | varies | Enable market beta neutralization |
| `phase2_beta_max_abs_exposure` | Conditional Float | [0.0, 0.30] | N/A | Max absolute beta exposure (if enabled) |
| `phase2_beta_lookback_days` | Conditional Categorical | [63, 126, 252] | N/A | Lookback window for beta (if enabled) |

### Liquidity Constraints (Requires capital_usd)
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `phase2_enable_liquidity_constraints` | Conditional Categorical | [False, True] | N/A | Enable ADV-based constraints |
| `phase2_adv_window` | Conditional Categorical | [10, 20, 40] | N/A | ADV calculation window (if enabled) |
| `phase2_max_adv_frac_name` | Conditional Float (log) | [0.001, 0.05] | N/A | Max % of ADV per position (if enabled) |
| `phase2_max_turnover_adv_frac` | Conditional Float (log) | [0.001, 0.20] | N/A | Max % of ADV turnover (if enabled) |
| `phase2_borrow_fee_bps_annual` | Conditional Float | [0.0, 300.0] | N/A | Annual borrow fee in bps (if enabled) |

---

## 7. PRUNING & SAFETY MECHANISMS

### Cost Envelope Pruning
| Mechanism | Threshold | Description |
|-----------|-----------|-------------|
| `phase2_cost_proxy` | > 8.0 | Prunes trials with excessive computational cost |

**Cost Proxy Formula:**
```
cost_proxy = (d_model/128) × (n_layers/4) × (seq_len/128) × (batch_size/32) × (max_epochs/10)
```

### Safety Pruning (During Walk-Forward)
| Parameter | Purpose | Description |
|-----------|---------|-------------|
| `prune_warmup_folds` | Skip early folds | Don't prune during warmup period |
| `safety_prune_after_folds` | Minimum folds | Evaluate at least N folds before pruning |
| `safety_prune_sharpe_floor` | Performance floor | Prune if Sharpe < threshold |
| `safety_prune_maxdd_ceiling` | Risk ceiling | Prune if max drawdown > threshold |

---

## 8. PARAMETER CATEGORIZATION

### Total Parameter Count
- **Total Parameters Sampled**: 68+
- **Fixed Governance Parameters**: Track weights, family weights, 3-pillar dims
- **Conditional Parameters**: Depend on overlay enables and runtime config

### Search Space Complexity
| Category | Parameters | Notes |
|----------|-----------|-------|
| Mamba Architecture | 13 | Core network structure |
| Training | 11 | Optimization and regularization |
| Pipeline | 8 | Track construction (mostly fixed) |
| Phase2 Engine | 5 | Walk-forward mechanics |
| Portfolio Base | 10 | Risk, sizing, costs, thresholds |
| Turnover Overlay | 5 (conditional) | Requires enable flag |
| Weight Smoothing | 2 (conditional) | Requires enable flag |
| Group Caps | 3 (conditional) | Requires group map file |
| Beta Neutral | 3 (conditional) | Requires enable flag |
| Liquidity | 5 (conditional) | Requires capital_usd |

---

## 9. KEY OPTIMIZATIONS & CONSTRAINTS

### Cache Optimization (v8 Fix)
- **smoothing_type**: Fixed to `"none"` to enable Track-C cache reuse
- **smoothing_window**: Removed from cache signature
- **Impact**: ~28 hours saved across 100 trials (17 min × 99 trials with CACHE HIT)

### Fixed Parameters for Uncertainty
- **mamba_head_type**: Always `"gaussian"` for uncertainty estimation
- **mamba_loss_fn**: Always `"gaussian_nll"` for probabilistic predictions

### Governance Inputs (Not Tuned)
- **Track-A/B weights**: Provided by Stage-A selector artifact
- **Family weights**: Governed by Stage-A, not sampled by Optuna
- **3-pillar dimensions**: Computed via variance-based analysis, not tuned

### Conditional Enables
Several overlay features are only tuned if:
- Runtime config provides necessary inputs (group_map, capital_usd)
- Initial categorical enable is sampled as `True`

---

## 10. TYPICAL DEFAULT CONFIGURATION

Based on OptunaConfig defaults and code analysis:

```python
{
    # Architecture
    "mamba_d_model": 128,
    "mamba_n_layers": 4,
    "mamba_ssm_dim": 96,
    "mamba_expand_factor": 2.0,
    "mamba_seq_len": 127,
    "mamba_activation": "silu",
    "mamba_norm_type": "rmsnorm",
    "mamba_norm_strategy": "pre",
    
    # Training
    "mamba_dropout": 0.0,
    "mamba_resid_dropout": 0.0,
    "mamba_ssm_dropout": 0.0,
    "mamba_gate_dropout": 0.0,
    "mamba_optimizer": "adamw",
    "mamba_learning_rate": 1e-4,
    "mamba_weight_decay": 1e-4,
    "mamba_grad_clip": 1.0,
    "mamba_lr_scheduler": "none",
    "mamba_max_epochs": 10,
    "mamba_batch_size": 32,
    
    # Head
    "mamba_head_type": "gaussian",
    "mamba_loss_fn": "gaussian_nll",
    "mamba_head_hidden_dim": 64,
    "mamba_head_num_layers": 2,
    "mamba_head_dropout": 0.0,
    
    # Phase2 Engine
    "phase2_engine": "v2",
    "phase2_update_sessions": 21,
    "phase2_replay_days": 126,
    "phase2_update_epochs": 2,
    "phase2_cov_ewma_lambda": 0.95,
    "phase2_shrinkage_alpha": 0.10,
    
    # Portfolio
    "phase2_target_vol": 0.15,
    "phase2_max_gross": 1.0,
    "phase2_max_name": 0.10,
    "phase2_max_net": 0.10,
    "phase2_k_spread": 1e-4,
    "phase2_k_impact": 5e-4,
    "phase2_z_clip": 6.0,
    
    # Thresholds
    "regime_threshold_base": 0.08,
    "regime_threshold_bull_mult": 0.8,
    "regime_threshold_bear_mult": 1.2,
    "regime_threshold_crisis_mult": 2.0,
    "conf_threshold": 0.5,
    "vol_scaler": 0.5,
    
    # Overlays (all disabled by default)
    "phase2_enable_turnover_overlay": False,
    "phase2_enable_weight_smoothing": False,
    "phase2_enable_group_caps": False,
    "phase2_enable_beta_neutral": False,
    "phase2_enable_liquidity_constraints": False,
}
```

---

## 11. REFERENCES

### Source File
`/home/rohan/dcf_stage_b_clean/src/stage_b_stateful/phase2_stateful.py`

### Key Functions
- `run_phase2_stateful_optuna()`: Main optimization loop (line 4019+)
- `_audit_phase2_trial_params()`: Parameter validation
- `_phase2_cfg_signature_for_prepared()`: Cache key generation (line 216+)

### Related Artifacts
- Study storage: SQLite database in `artifacts/optuna_studies/`
- Trial exports: JSON files in `artifacts/optuna/`
- Stage-A weights: Governance input from `artifacts/stage_a/`

---

## Document Version
- **Created**: 2025-01-02
- **Phase 2 Version**: v8 (cache-optimized)
- **Total Parameters**: 68+
- **Study Mode**: Full search (100 trials)
