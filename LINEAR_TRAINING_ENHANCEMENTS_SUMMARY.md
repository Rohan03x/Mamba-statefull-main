# Linear Combiner Training Enhancements

**Branch**: Rudra  
**Commit**: 1008da5  
**Date**: 2025  
**Status**: ✅ Implemented, Tested, Pushed

## Overview

Two critical improvements to LinearCombinerState training pipeline that enhance alpha quality and reduce unintended beta exposure.

## Enhancement 1: Alpha Residual Targets

### Problem
- Training targets were `y = r / sigma_exec` (raw standardized returns)
- This captures market mode along with idiosyncratic signals
- Model learns market beta instead of pure alpha
- Results in unintended beta exposure in portfolio

### Solution
Cross-sectional demeaning before target computation:

```python
# OLD: Train on raw returns
y_matured = realized_ret / (sigma_matured + eps)

# NEW: Train on alpha residual (idiosyncratic return)
valid_ret_mask = np.isfinite(realized_ret)
if np.sum(valid_ret_mask) > 1:
    ret_mean = float(np.mean(realized_ret[valid_ret_mask]))
    r_alpha = realized_ret - ret_mean  # Remove market mode
else:
    r_alpha = realized_ret

y_matured = r_alpha / (sigma_matured + eps)
```

### Impact
- **Reduces beta**: Model trained on idiosyncratic component only
- **Improves harmony**: Higher alpha, lower beta accidents
- **Better generalization**: Less sensitive to market regime changes

### Code Location
- **File**: [src/portfolio/linear_alpha_combiner.py](src/portfolio/linear_alpha_combiner.py#L565-L578)
- **Method**: `update_if_matured()`
- **Lines**: 565-578

---

## Enhancement 2: Per-Day Sample Weighting

### Problem
- Training data concatenated equally: `X_all = np.vstack(X_chunks)`
- Days with more eligible symbols get proportionally more weight in Ridge fit
- Universe size bias: 30-symbol days dominate 10-symbol days 3x
- Training skewed toward days with larger universes

### Solution
Equal per-day weighting using sample size normalization:

```python
# OLD: Simple concatenation (implicit weighting by day size)
X_all = np.vstack(X_chunks)
y_all = np.concatenate(y_chunks)

# NEW: Weighted concatenation (each day contributes equally)
X_weighted = []
y_weighted = []

for X_day, y_day, n_day in zip(X_chunks, y_chunks, day_sizes):
    # Weight = sqrt(1/n_day) so that when squared in Ridge loss,
    # each day's total contribution is 1/n_days regardless of its sample count
    day_weight = 1.0 / float(np.sqrt(max(n_day, 1)))
    X_weighted.append(X_day * day_weight)
    y_weighted.append(y_day * day_weight)

X_all = np.vstack(X_weighted)
y_all = np.concatenate(y_weighted)
```

### Mathematical Justification
Ridge regression minimizes:
```
L = Σ (y_i - X_i·β)² + λ||β||²
```

With weighting by `w_day = sqrt(1/n_day)`:
```
L = Σ_days [ Σ_samples_in_day (w·y_i - w·X_i·β)² ] + λ||β||²
  = Σ_days [ (1/n_day) · Σ_samples (y_i - X_i·β)² ] + λ||β||²
```

Each day contributes equally to total loss regardless of `n_day`.

### Impact
- **Fair sampling**: Days contribute equally regardless of universe size
- **Robust to composition changes**: Universe expansion/contraction doesn't bias training
- **Better temporal balance**: Recent small-universe days weighted equally with large-universe days

### Code Location
- **File**: [src/portfolio/linear_alpha_combiner.py](src/portfolio/linear_alpha_combiner.py#L636-L656)
- **Method**: `_do_refit()`
- **Lines**: 636-656

---

## Testing

Comprehensive test suite covering both enhancements:

### Test File
[tests/test_linear_training_enhancements.py](tests/test_linear_training_enhancements.py)

### Test Coverage
1. ✅ **test_alpha_residual_cross_sectional_demean**
   - Verifies targets are demeaned: `mean(y_day) ≈ 0`
   - Confirms market mode removal

2. ✅ **test_day_weighting_equal_contribution**
   - Tests varying universe sizes (10 vs 30 symbols)
   - Verifies equal day contribution via weighting

3. ✅ **test_combined_fix_reduces_market_beta**
   - Integration test with simulated market factor
   - Confirms idiosyncratic training via demeaning

4. ✅ **test_edge_case_single_symbol_day**
   - Handles days with only 1 valid symbol
   - Weight = 1/sqrt(1) = 1.0 (no scaling)

5. ✅ **test_zero_variance_day_handling**
   - All stocks have identical return
   - After demeaning, all targets ≈ 0 (graceful handling)

### Test Results
```bash
$ pytest tests/test_linear_training_enhancements.py -v
============================= 5 passed in 0.37s =============================
```

---

## Integration Points

### Phase-2 Stateful Loop
The enhancements are automatically active in the main walk-forward loop:

**File**: [src/stage_b_stateful/phase2_stateful.py](src/stage_b_stateful/phase2_stateful.py)

```python
# Training happens via update_if_matured() call
if linear_state is not None:
    linear_state.observe_day(i, X_day, sigma_exec)
    
    refit_info = linear_state.update_if_matured(
        i=i,
        fwd_ret_mat=fwd_ret_mat,
        freeze=(calib_score < 0.55)  # Freeze if calibration poor
    )
    
    # Alpha residual + day weighting applied automatically
```

No config changes needed - enhancements are embedded in training pipeline.

---

## Expected Production Impact

### Before Enhancements
- Linear layer captures market beta along with alpha
- Universe size bias: days with more symbols dominate training
- Higher beta exposure from blend weight `w_L`

### After Enhancements
- **Lower beta**: Training on `r_alpha = r - mean(r)` removes market mode
- **Fairer training**: Each day contributes equally regardless of universe size
- **Improved harmony**: Higher alpha, lower beta accidents
- **Robust to universe changes**: Expansion/contraction doesn't skew model

### Metrics to Watch
1. **Portfolio beta**: Should decrease (closer to market-neutral)
2. **Information ratio**: Should improve (higher alpha / beta)
3. **Correlation to SPY**: Should decrease
4. **Blend weight stability**: `w_L` should remain stable across regimes

---

## Commit History

### Commit 1: Audit Fixes
- **SHA**: 56d7f60
- **Message**: "Phase-2 Audit Fixes: 7 gap implementations"
- **Changes**: Robust optimizer, policy controls, learning governor, .gitignore

### Commit 2: Adaptive Blending
- **SHA**: b66b32e
- **Message**: "Add quality-adaptive blend weight for linear combiner"
- **Changes**: Dynamic `w_L` based on R², drift, correlation, sign disagreement

### Commit 3: Training Enhancements (Current)
- **SHA**: 1008da5
- **Message**: "Enhance linear combiner training: alpha residual + day weighting"
- **Changes**: Cross-sectional demeaning, per-day sample weighting
- **Files**:
  - [src/portfolio/linear_alpha_combiner.py](src/portfolio/linear_alpha_combiner.py)
  - [tests/test_linear_training_enhancements.py](tests/test_linear_training_enhancements.py)

---

## References

### Related Documents
- [Phase2_Optuna_Search_Space.md](Phase2_Optuna_Search_Space.md) - Hyperparameter search space
- [GAP_C_D_FIXES_SUMMARY.md](GAP_C_D_FIXES_SUMMARY.md) - Previous audit gap fixes

### Key Files Modified
1. [src/portfolio/linear_alpha_combiner.py](src/portfolio/linear_alpha_combiner.py)
   - `update_if_matured()`: Alpha residual target computation
   - `_do_refit()`: Per-day sample weighting

2. [tests/test_linear_training_enhancements.py](tests/test_linear_training_enhancements.py)
   - 5 comprehensive tests
   - Edge case coverage

### Git Branch
```bash
git checkout Rudra
git log --oneline -3
# 1008da5 Enhance linear combiner training: alpha residual + day weighting
# b66b32e Add quality-adaptive blend weight for linear combiner
# 56d7f60 Phase-2 Audit Fixes: 7 gap implementations
```

---

## Next Steps (Optional Future Work)

### Potential Enhancements
1. **Market beta residualization**: Alternative to cross-sectional demeaning
   - Compute `r_alpha = r_i - beta_i * r_mkt` using SPY returns
   - More sophisticated than demeaning, requires market data

2. **Volatility-weighted day weighting**: 
   - Weight days by `1 / realized_vol_day` instead of equal
   - Down-weight high-volatility days that may have outliers

3. **Rolling window decay**:
   - Apply exponential decay to older days within window
   - Currently all days in window equally weighted by day

4. **Feature-specific regularization**:
   - Different `lambda` for z_mamba vs context features
   - Currently uses global `ridge_lambda` for all features

### Config Flags (If Needed)
If reverting to old behavior is desired:

```python
class LinearCombinerState:
    use_alpha_residual: bool = True   # If False, train on raw returns
    use_day_weighting: bool = True    # If False, simple concatenation
```

Currently hardcoded to `True` (enhancements always active).

---

## Summary

✅ **Alpha Residual**: Cross-sectional demeaning removes market beta from training targets  
✅ **Day Weighting**: Per-day normalization fixes universe size bias  
✅ **Testing**: 5 comprehensive tests, all passing  
✅ **Integration**: Automatically active in Phase-2 loop  
✅ **Git**: Committed to Rudra branch, pushed to remote  

**Expected Impact**: Lower beta, higher alpha, improved harmony in linear blend path.
