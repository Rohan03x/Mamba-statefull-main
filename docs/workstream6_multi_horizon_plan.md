# Workstream 6: Multi-Horizon Multi-Task Training Implementation Plan

## Status: ✅ COMPLETE
**Started**: 2025-01-27
**Completed**: 2025-01-27

---

## Design Decisions Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Backbone Strategy | **Option C**: FiLM injection into temporal tower | Conditional computation; same features interpreted differently per horizon |
| Horizon Encoding | **Hybrid**: Discrete embedding + Sinusoidal log(h) | Memorize known horizons + smooth ordering structure |
| Output Structure | **Batch mode**: Output all H horizons, query wrapper | Cross-horizon consistency, regularization opportunities |
| Label Handling | **Masked loss + Horizon-balanced batching** | Correctness + balanced learning signal |
| Loss Weighting | **Learned uncertainty × Static business priors** | Adapts to task difficulty while keeping business alignment |
| Scope | **Both Level 1 and Level 2** | Consistency, distillation, fallback capability |
| Phase2 Integration | **Hybrid**: Single backbone + per-horizon calibrators | Data efficiency + horizon-specific drift handling |

---

## Implementation Checklist

### PHASE 1: Data Structures
| # | Component | File | Status | Notes |
|---|-----------|------|--------|-------|
| 1.1 | `MultiHorizonSequenceData` dataclass | sequence_models.py | ✅ DONE | Single-symbol multi-horizon data |
| 1.2 | `MultiHorizonCrossSectionData` dataclass | sequence_models.py | ✅ DONE | Multi-symbol multi-horizon data |
| 1.3 | `build_multi_horizon_data()` function | sequence_models.py | ✅ DONE | Build Level 1 data |
| 1.4 | `build_multi_horizon_cross_section_data()` function | sequence_models.py | ✅ DONE | Build Level 2 data |

### PHASE 2: Horizon Encoding
| # | Component | File | Status | Notes |
|---|-----------|------|--------|-------|
| 2.1 | `HorizonEncoder` module | sequence_models.py | ✅ DONE | Hybrid discrete + sinusoidal |
| 2.2 | `FiLMLayer` module | sequence_models.py | ✅ DONE | Feature-wise Linear Modulation |

### PHASE 3: Model Architecture
| # | Component | File | Status | Notes |
|---|-----------|------|--------|-------|
| 3.1 | `MultiHorizonMambaBlock` | sequence_models.py | ✅ DONE | FiLM-conditioned Mamba block |
| 3.2 | `MultiHorizonHead` | sequence_models.py | ✅ DONE | Horizon-conditioned Gaussian head |
| 3.3 | `MultiHorizonMambaRegressor` (Level 1) | sequence_models.py | ✅ DONE | Single-symbol multi-horizon |
| 3.4 | `MultiHorizonCrossSectionMamba` (Level 2) | sequence_models.py | ✅ DONE | Multi-symbol multi-horizon |

### PHASE 4: Training Infrastructure
| # | Component | File | Status | Notes |
|---|-----------|------|--------|-------|
| 4.1 | `MultiHorizonLoss` | sequence_models.py | ✅ DONE | Uncertainty weighting + business priors |
| 4.2 | `HorizonBalancedSampler` | sequence_models.py | ✅ DONE | Oversample long-horizon samples |
| 4.3 | `train_multi_horizon_mamba()` | sequence_models.py | ✅ DONE | Level 1 training function |
| 4.4 | `train_multi_horizon_cross_section_mamba()` | sequence_models.py | ⬜ DEFER | Level 2 training (uses same pattern) |

### PHASE 5: Calibration
| # | Component | File | Status | Notes |
|---|-----------|------|--------|-------|
| 5.1 | `HorizonCalibrator` | sequence_models.py | ✅ DONE | Per-horizon affine adapter |
| 5.2 | `MultiHorizonCalibratorBank` | sequence_models.py | ✅ DONE | Bank of calibrators + save/load |

### PHASE 6: Phase2 Integration
| # | Component | File | Status | Notes |
|---|-----------|------|--------|-------|
| 6.1 | `_build_multi_horizon_data_from_prepared()` | phase2_stateful.py | ⬜ DEFER | Phase2 integration |
| 6.2 | `_train_multi_horizon_backbone()` | phase2_stateful.py | ⬜ DEFER | Training wrapper |
| 6.3 | `_update_horizon_calibrators()` | phase2_stateful.py | ⬜ DEFER | Online calibrator update |
| 6.4 | `_predict_multi_horizon()` | phase2_stateful.py | ⬜ DEFER | Inference with calibration |

### PHASE 7: Documentation & Exports
| # | Component | File | Status | Notes |
|---|-----------|------|--------|-------|
| 7.1 | Add Section 16: Multi-Horizon Parameters | Phase2_Optuna_Search_Space.md | ✅ DONE | All MH parameters |
| 7.2 | Update unified staging (Section 10) | Phase2_Optuna_Search_Space.md | ✅ DONE | MH staged tuning |
| 7.3 | Update __all__ exports | sequence_models.py | ✅ DONE | 14 new exports |
| 7.4 | Verification tests | terminal | ✅ DONE | Import + forward pass + training tests |

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MultiHorizonMamba (Level 1 / Level 2)                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Inputs:                                                                      │
│    x: [B, T, F]           (Level 1) or [B, T, N, F] (Level 2)                │
│    horizons: [5, 21, 63, 126, 252]                                           │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 1. HORIZON ENCODER                                                   │     │
│  │    discrete_emb: [H, d_horizon - continuous_dim]                    │     │
│  │    sinusoidal:   [H, continuous_dim]                                │     │
│  │    h_emb = concat(discrete, sinusoidal) → [H, d_horizon]            │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                           │                                                   │
│                           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 2. INPUT PROJECTION                                                  │     │
│  │    h = Linear(x) → [B, T, d_model]                                  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                           │                                                   │
│                           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 3. TEMPORAL TOWER (with FiLM conditioning per block)                │     │
│  │                                                                      │     │
│  │    For each block l in [1..n_layers]:                               │     │
│  │      h = MambaBlock(h)                    # [B, T, d_model]         │     │
│  │      γ_l, β_l = FiLM_l(h_emb)            # [H, d_model] each       │     │
│  │      h = γ_l * h + β_l                   # Broadcast → [B, H, T, d] │     │
│  │                                                                      │     │
│  │    After first block: h shape becomes [B, H, T, d_model]            │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                           │                                                   │
│                           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 4. TEMPORAL POOLING                                                  │     │
│  │    h_pooled = h[:, :, -1, :]  → [B, H, d_model]                     │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                           │                                                   │
│                           ▼ (Level 2 only)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 5. CROSS-SECTION ATTENTION (Level 2 only)                           │     │
│  │    For each cross-section layer:                                    │     │
│  │      H = CrossSectionAttention(H, adjacency, mask)                  │     │
│  │    Output: [B, H, N, d_model]                                       │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                           │                                                   │
│                           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 6. MULTI-HORIZON HEAD                                               │     │
│  │    Input: h + h_emb (concatenated)                                  │     │
│  │    (mu_h, sigma_h) = Head(concat(h, h_emb))                        │     │
│  │                                                                      │     │
│  │    Level 1 Output: mu [B, H], sigma [B, H]                         │     │
│  │    Level 2 Output: mu [B, H, N], sigma [B, H, N]                   │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                           │                                                   │
│                           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 7. CALIBRATION ADAPTERS (inference only, per-horizon)              │     │
│  │    mu_cal = a_h * mu_h + b_h                                        │     │
│  │    sigma_cal = softplus(c_h) * sigma_h + softplus(d_h)             │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Loss Function Design

```
MultiHorizonLoss = Σ_h [ (1 / (2 * τ_h²)) * NLL_h + log(τ_h) ] * w_business_h + λ_mono * MonotonicPenalty

Where:
- τ_h: Learned per-horizon uncertainty (homoscedastic task weighting)
- NLL_h: Gaussian NLL for horizon h, masked for valid samples only
- w_business_h: Static business priority weights (e.g., 63d gets 0.4)
- MonotonicPenalty: ReLU(σ_short - σ_long) to encourage sigma increases with horizon
```

---

## Horizon-Balanced Batching

```
Problem: 
  - 5-day horizon: ~T-5 valid samples (many)
  - 252-day horizon: ~T-252 valid samples (few)
  
Solution (sqrt balancing):
  - weight_h = sqrt(valid_count_h) / sum(sqrt(valid_count_h))
  - Sample probability ∝ weight_h / valid_count_h per sample
  
This ensures long horizons appear frequently enough for gradient signal.
```

---

## Storage Structure (Phase2)

```
artifacts/models/
├── multi_horizon_backbone.pt          # Single backbone for all horizons
├── multi_horizon_criterion.pt         # Learned uncertainty weights τ_h
└── calibrators/
    ├── calibrator_h5.pt               # Calibrator for 5-day
    ├── calibrator_h21.pt              # Calibrator for 21-day
    ├── calibrator_h63.pt              # Calibrator for 63-day (primary)
    ├── calibrator_h126.pt             # Calibrator for 126-day
    └── calibrator_h252.pt             # Calibrator for 252-day
```

---

## Parameter Reference (Optuna)

| Parameter | Type | Range | Default | Stage | Description |
|-----------|------|-------|---------|-------|-------------|
| `mh_enabled` | Bool | - | False | - | Enable multi-horizon |
| `mh_horizons` | Fixed | - | [5,21,63,126,252] | - | Horizon set |
| `mh_d_horizon` | Cat | {16,32,48,64} | 32 | 2 | Horizon embedding dim |
| `mh_continuous_dim` | Cat | {8,16,24} | 16 | 2 | Sinusoidal dim |
| `mh_d_model` | Cat | {96,128,192,256} | 128 | 3 | Model dimension |
| `mh_n_layers` | Cat | {2,4,6,8} | 4 | 3 | Temporal blocks |
| `mh_film_hidden_mult` | Cat | {1,2,4} | 2 | 2 | FiLM hidden multiplier |
| `mh_learning_rate` | Log | [1e-5,1e-3] | 1e-4 | 1 | Learning rate |
| `mh_weight_decay` | Log | [1e-6,1e-2] | 1e-4 | 1 | Weight decay |
| `mh_dropout` | Float | [0.0,0.3] | 0.1 | 1 | Dropout |
| `mh_batch_size` | Cat | {16,32,64} | 32 | 1 | Batch size |
| `mh_max_epochs` | Cat | {10,20,30} | 10 | 1 | Max epochs |
| `mh_grad_clip` | Float | [0.3,2.0] | 1.0 | 1 | Gradient clip |
| `mh_balance_mode` | Cat | {equal,sqrt,log} | sqrt | 1 | Horizon balancing |
| `mh_sigma_mono_penalty` | Log | [1e-4,1e-1] | 0.01 | 2 | Monotonic sigma penalty |
| `mh_weight_h5` | Float | [0.05,0.25] | 0.10 | 1 | Business weight 5d |
| `mh_weight_h21` | Float | [0.05,0.25] | 0.15 | 1 | Business weight 21d |
| `mh_weight_h63` | Float | [0.20,0.60] | 0.40 | 1 | Business weight 63d |
| `mh_weight_h126` | Float | [0.10,0.30] | 0.20 | 1 | Business weight 126d |
| `mh_weight_h252` | Float | [0.05,0.25] | 0.15 | 1 | Business weight 252d |
| `mh_sigma_floor` | Log | [1e-4,5e-2] | 0.01 | 3 | Min sigma |
| `mh_sigma_ceiling` | Log | [0.5,3.0] | 2.0 | 3 | Max sigma |
| `mh_sigma_monotonic` | Bool | - | True | 2 | Enforce monotonic sigma |
| `mh_calibrator_lr` | Log | [1e-4,1e-2] | 1e-3 | 3 | Calibrator learning rate |
| `mh_calibrator_update_freq` | Cat | {1,5,10} | 5 | 3 | Days between updates |

---

## Staged Tuning (within Unified Stages)

### Unified Stage 1: Fit & Stability
- MH: Tune `mh_learning_rate`, `mh_weight_decay`, `mh_dropout`, `mh_batch_size`, `mh_balance_mode`, `mh_weight_h*`

### Unified Stage 2: Structure/Capacity
- MH: Tune `mh_d_horizon`, `mh_continuous_dim`, `mh_film_hidden_mult`, `mh_sigma_mono_penalty`, `mh_sigma_monotonic`

### Unified Stage 3: Temporal + Output
- MH: Tune `mh_d_model`, `mh_n_layers`, `mh_sigma_floor`, `mh_sigma_ceiling`, `mh_calibrator_lr`

---

## Implementation Notes

### FiLM Initialization
- Initialize FiLM to identity: γ=1, β=0
- This ensures model starts as if no horizon conditioning (stable)

### Horizon Encoding Scale
- Use log(horizon) for sinusoidal encoding (5→1.6, 252→5.5)
- Normalize by log(504) to keep in [0, 1] range

### Memory Considerations
- Level 2 with H=5 horizons: VRAM × 5 roughly
- May need to reduce batch size or use gradient accumulation
- Consider computing horizons sequentially if VRAM-limited

### Validation Metrics
- Compute IC per horizon
- Track learned uncertainty weights τ_h
- Monitor sigma monotonicity compliance

---

## Next Steps

1. ⬜ Create PHASE 1: Data Structures
2. ⬜ Create PHASE 2: Horizon Encoding
3. ⬜ Create PHASE 3: Model Architecture
4. ⬜ Create PHASE 4: Training Infrastructure
5. ⬜ Create PHASE 5: Calibration
6. ⬜ Create PHASE 6: Phase2 Integration
7. ⬜ Create PHASE 7: Documentation
8. ⬜ Verification: Import tests
9. ⬜ Verification: Forward pass tests
10. ⬜ Verification: Training loop tests
