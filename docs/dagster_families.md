# Dagster Feature Families Reference

> Generated: January 22, 2026  
> Total Families: **48**

---

## Overview

The Dagster pipeline materializes feature families as partitioned assets. Families are organized into four categories:

| Category | Count | Description |
|----------|-------|-------------|
| Base Families | 37 | Core feature generators |
| HF Modules | 4 | High-frequency modules requiring special compute |
| HF Blocks | 6 | High-frequency aggregation blocks |
| Meta Families | 1 | Derived aggregation families |
| **Total** | **48** | |

---

## Base Families (37)

| # | Family | Description |
|---|--------|-------------|
| 1 | `alternative_signals` | Alternative data signals (42 features + 3 governance = 45 total) - [See full feature list](#alternative_signals-family-full-feature-reference) |
| 2 | `arima_forecast` | ARIMA time-series forecasts (10 features + 3 governance = 13 total) - [See full feature list](#arima_forecast-family-full-feature-reference) |
| 3 | `calibration` | Forecast calibration metrics (22 features + 3 governance = 25 total) - [See full feature list](#calibration-family-full-feature-reference) |
| 4 | `candle_mechanics` | OHLCV-derived candle mechanics (36 features + 3 governance = 39 total) - [See full feature list](#candle_mechanics-family-full-feature-reference) |
| 5 | `cboe_term` | VIX term structure features (16 features + 3 governance = 19 total) - [See full feature list](#cboe_term-family-full-feature-reference) |
| 6 | `corp_actions_splits` | Stock split event features (8 features + 3 governance = 11 total) - [See full feature list](#corp_actions_splits-family-full-feature-reference) |
| 7 | `correlation` | Cross-asset correlation features (28 features + 3 governance = 31 total) - [See full feature list](#correlation-family-full-feature-reference) |
| 8 | `cross_asset` | Cross-asset macro risk/regime signals (22 features + 3 governance = 25 total) - [See full feature list](#cross_asset-family-full-feature-reference) |
| 9 | `dcf` | Price-relative valuation anchors (19 features + 3 governance = 22 total) - [See full feature list](#dcf-family-full-feature-reference) |
| 10 | `dividends` | Dividend event features (8 features + 4 governance = 12 total) - [See full feature list](#dividends-family-full-feature-reference) |
| 11 | `earnings` | Earnings event & quality signals (15 features + 4 governance = 19 total) - [See full feature list](#earnings-family-full-feature-reference) |
| 12 | `econ_events_calendar` | Macro economic releases + surprises (102 features + 4 governance = 106 total) - [See full feature list](#econ_events_calendar-family-full-feature-reference) |
| 13 | `exchange_calendar` | Exchange holidays and early closes (5 features + 4 governance = 9 total) - [See full feature list](#exchange_calendar-family-full-feature-reference) |
| 14 | `fin_g1` | Fundamental group 1 - Liquidity ratios (6 features + 3 governance = 9 total) - [See full feature list](#fin_g1-family-full-feature-reference) |
| 15 | `fin_g2` | Fundamental group 2 - Leverage/capital structure (12 features + 3 governance = 15 total) - [See full feature list](#fin_g2-family-full-feature-reference) |
| 16 | `fin_g3` | Fundamental group 3 - Efficiency/turnover ratios (9 features + 3 governance = 12 total) - [See full feature list](#fin_g3-family-full-feature-reference) |
| 17 | `fin_g4` | Fundamental group 4 - Cash flow & earnings quality (9 features + 3 governance = 12 total) - [See full feature list](#fin_g4-family-full-feature-reference) |
| 18 | `fin_g5` | Fundamental group 5 - Growth metrics (8 features + 3 governance = 11 total) - [See full feature list](#fin_g5-family-full-feature-reference) |
| 19 | `fin_g6` | Fundamental group 6 - Valuation multiples (9 features + 3 governance = 12 total) - [See full feature list](#fin_g6-family-full-feature-reference) |
| 20 | `fin_g7` | Fundamental group 7 - Dividend/shareholder yield metrics (8 features + 3 governance = 11 total) - [See full feature list](#fin_g7-family-full-feature-reference) |
| 21 | `finbert` | FinBERT sentiment embeddings (2 sentiment scores + 4 governance = 6 total) - [See full feature list](#finbert-family-full-feature-reference) |
| 22 | `garch_iv` | GARCH(1,1) volatility forecasts + regime features (19 features + 3 governance = 22 total) - [See full feature list](#garch_iv-family-full-feature-reference) |
| 23 | `index_constituents` | Index membership events + flows (20 features + 4 governance = 24 total) - [See full feature list](#index_constituents-family-full-feature-reference) |
| 24 | `insider_form4` | SEC Form 4 insider transactions (29 features + 3 governance = 32 total) - [See full feature list](#insider_form4-family-full-feature-reference) |
| 25 | `marketcap_history` | Historical market cap + size dynamics (8 features + 1 governance = 9 total) - [See full feature list](#marketcap_history-family-full-feature-reference) |
| 26 | `microstructure` | Raw microstructure from daily OHLCV (31 features + 3 governance = 34 total) - [See full feature list](#microstructure-family-full-feature-reference) |
| 27 | `ml_framework` | EODHD technical indicators (10 features + 3 governance = 13 total) - [See full feature list](#ml_framework-family-full-feature-reference) |
| 28 | `multiasset` | Equity benchmark exposures: SPY/QQQ/IWM/ACWI correlations + PCA style factors (12 features + 3 governance = 15 total) - [See full feature list](#multiasset-family-full-feature-reference) |
| 29 | `online_learning` | Online learning adaptation metrics (32 features + 3 governance = 35 total) - [See full feature list](#online_learning-family-full-feature-reference) |
| 30 | `options` | Raw options chain snapshot from EODHD: IV, volume, OI, moneyness (23 features + 1 governance = 24 total) - [See full feature list](#options-family-full-feature-reference) |
| 31 | `options_anchoring` | Institutional-grade options anchoring with decay weighting (11 features + 1 governance = 12 total) - [See full feature list](#options_anchoring-family-full-feature-reference) |
| 32 | `peer_screener_context` | Cross-sectional peer ranks/percentiles using cached fundamentals (23 features + 1 governance = 24 total) - [See full feature list](#peer_screener_context-family-full-feature-reference) |
| 33 | `quantile_forecast` | Quantile distribution forecasts (22 features + 3 governance = 25 total) - [See full feature list](#quantile_forecast-family-full-feature-reference) |
| 34 | `regime` | Institutional-grade regime classification with temporal context (9 features, no separate governance) - [See full feature list](#regime-family-full-feature-reference) |
| 35 | `short_interest` | Short interest tracking with squeeze probability analysis (20 features + 1 governance = 21 total) - [See full feature list](#short_interest-family-full-feature-reference) |
| 36 | `subsidiary` | Organizational complexity from quarterly R&D/OpEx (8 features + 4 governance = 12 total) - [See full feature list](#subsidiary-family-full-feature-reference) |
| 37 | `tft_features` | TFT-style temporal patterns for forecasting (17 features, no governance) - [See full feature list](#tft_features-family-full-feature-reference) |

---

## HF Modules (4)

High-frequency modules requiring specialized compute resources:

| # | Family | Description |
|---|--------|-------------|
| 1 | `doc_embedding_novelty_hf` | Document embedding novelty detection (symbol-independent, shared cache) |
| 2 | `earnings_transcript_hf` | Earnings call transcript analysis |
| 3 | `macro_tst_hf` | Macro time-series transformer |
| 4 | `news_sentiment_hf` | News sentiment analysis (excluded from Dagster assets) |

---

## HF Blocks (6)

High-frequency aggregation blocks that combine multiple signal types:

| # | Family | Horizon-Bound | Description |
|---|--------|---------------|-------------|
| 1 | `forecast_hf` | ✅ Yes | Forecast aggregation features |
| 2 | `fundamental_val_hf` | ❌ No | Fundamental valuation HF features |
| 3 | `macro_regime_hf` | ❌ No | Macro regime HF indicators |
| 4 | `news_nlp_hf` | ❌ No | News NLP HF features |
| 5 | `tech_micro_hf` | ❌ No | Technical/microstructure HF features |
| 6 | `vol_deriv_hf` | ❌ No | Volatility/derivatives HF features |

---

## Meta Families (1)

Derived families that aggregate other family outputs:

| # | Family | Description |
|---|--------|-------------|
| 1 | `hf_agg` | Aggregates all HF blocks into unified panel (horizon-bound) |

---

## Partitioning Strategy

### Horizon-Bound Families
These families are partitioned by **(symbol, horizon)** and produce horizon-specific outputs:

- `quantile_forecast`
- `calibration`
- `online_learning`
- `forecast_hf`
- `hf_agg`

### Symbol-Only Families
All other families are partitioned by **symbol** only.

### Symbol-Only HF Blocks
These HF blocks use symbol-only caching (no horizon dimension):

- `fundamental_val_hf`
- `macro_regime_hf`
- `news_nlp_hf`
- `tech_micro_hf`
- `vol_deriv_hf`

---

## Cache Paths

| Family Type | Cache Path Pattern |
|-------------|-------------------|
| Standard (horizon-linked) | `cache/symbols/<SYMBOL>/h<H>/<family>.parquet` |
| Symbol-only HF blocks | `cache/symbols/<SYMBOL>/hf/<family>.parquet` |
| Shared (doc_embedding) | `cache/shared/doc_embedding/doc_embedding_novelty_hf.parquet` |

---

## Excluded from Dagster

The following families are excluded from Dagster asset materialization:

| Family | Reason |
|--------|--------|
| `news_sentiment_hf` | Runs separately outside Dagster |
| `peer_screener_context` | Universe-wide cross-sectional snapshot, not per-symbol |

---

## Governance Columns

Each family produces three governance columns:

| Column | Description |
|--------|-------------|
| `{family}_has_data` | Boolean indicating data presence |
| `{family}_activity` | Activity score (0-1) |
| `{family}_days_since_update` | Days since last data update |

---

## Configuration

Dagster family assets accept the following configuration:

```python
class FamilyRunConfig:
    wf_start: Optional[str]           # Walk-forward start date
    wf_end: Optional[str]             # Walk-forward end date
    allow_wf_override: bool = False   # Allow WF date overrides
    wf_train_years: Optional[int]     # Training window years
    wf_step_years: Optional[int]      # Step size in years
    wf_step_days: Optional[int]       # Step size in days
    workers: int = 1                  # Parallel workers
    hf_workers: Optional[int]         # HF-specific workers
    strict: bool = False              # Strict validation mode
    horizon: int = 63                 # Default horizon
    min_variance_threshold: float     # Variance validation threshold
    max_zero_percentage: float        # Max allowed zero percentage
    max_null_percentage: float        # Max allowed null percentage
    fail_on_fallback_columns: bool    # Fail on fallback column detection
```

---

## Optional Mamba Columns (Unified CLI)

Columns that are portfolio-only by default can be optionally routed to Mamba stream via environment variables.

### Unified Format (Recommended)

Use a single `MAMBA_OPTIONAL_COLUMNS` environment variable with semicolon-separated family blocks:

```bash
# Format: MAMBA_OPTIONAL_COLUMNS="family:col1,col2;family2:col3,col4;family3:all"
export MAMBA_OPTIONAL_COLUMNS="correlation:corr_decoupling_z,corr_spread_20_60_spy;dcf:overextension;cross_asset:all"
```

**Supported Family Keys:**
| Family Key | Family Name | Example Columns |
|------------|-------------|-----------------|
| `correlation` | correlation | `corr_decoupling_z`, `corr_spread_20_60_spy` |
| `corp_splits` | corp_actions_splits | `flag`, `log_ratio`, `post_5d`, `post_20d`, `recency` |
| `cross_asset` | cross_asset | `tnx_corr_change_5d`, `irx_corr_change_5d`, `coupling_change` |
| `dcf` | dcf | `overextension`, `log_p2fv_1y`, `downside_skew_stress` |
| `dividends` | dividends | `dividend_event_intensity`, `ex_dividend_window_strength`, `days_to_ex_dividend` |
| `earnings` | earnings | `days_since_earnings`, `days_to_next_earnings` |
| `alt_signals` | alternative_signals | `overnight_return`, `gap_vs_vix_interaction` |
| `arima` | arima_forecast | `drift`, `volatility_forecast` |
| `quantile` | quantile_forecast | `spread_50_10`, `spread_90_50` |
| `candle` | candle_mechanics | `body_pct_range`, `shadow_ratio` |

**Special Value `all`:** Routes ALL optional columns to Mamba for that family.

```bash
# Route all optional DCF columns + specific correlation columns
export MAMBA_OPTIONAL_COLUMNS="dcf:all;correlation:corr_decoupling_z"
```

### Legacy Format (Still Supported)

Individual family-specific environment variables remain backward compatible:

```bash
# Individual family env vars
export CORRELATION_MAMBA_OPTIONAL="corr_decoupling_z,corr_spread_20_60_spy"
export CORP_SPLITS_MAMBA_OPTIONAL="flag,log_ratio,post_5d,post_20d,recency"
export CROSS_ASSET_MAMBA_OPTIONAL="tnx_corr_change_5d,irx_corr_change_5d"
export DCF_MAMBA_OPTIONAL="overextension,log_p2fv_1y"
export DIVIDENDS_MAMBA_OPTIONAL="dividend_event_intensity,ex_dividend_window_strength"
export EARNINGS_MAMBA_OPTIONAL="days_since_earnings,days_to_next_earnings"
export ALT_SIGNALS_MAMBA_OPTIONAL="overnight_return,gap_vs_vix_interaction"
export ARIMA_FORECAST_MAMBA_OPTIONAL="drift,volatility_forecast"
export QUANTILE_FORECAST_MAMBA_STACKING="spread_50_10,spread_90_50"
export CANDLE_MECHANICS_MAMBA_OPTIONAL="body_pct_range,shadow_ratio"
```

**Note:** Both formats can be used together — a column is enabled if it appears in either the unified or legacy env var.

---

## Related Files

- [dagster_prep_families/family_assets.py](../dagster_prep_families/family_assets.py) - Dagster asset definitions
- [tools/prep_families.py](../tools/prep_families.py) - Family preparation logic
- [src/features/family_spec.py](../src/features/family_spec.py) - Family specifications
- [src/features/aggregator_panel.py](../src/features/aggregator_panel.py) - Feature aggregation handlers

---

## alternative_signals Family — Full Feature Reference

> **Total Columns: 45** (42 features + 3 governance)  
> **Family Intent:** PREDICTIVE  
> **Update Cadence:** irregular  
> **Decay:** fast  
> **Allow Direct Alpha:** conditional

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `alternative_signals_has_data` | HYGIENE | Boolean flag: data exists for this row |
| 2 | `alternative_signals_activity` | HYGIENE | Activity score (0-1) |
| 3 | `alternative_signals_days_since_update` | HYGIENE | Days since last data update |

### Beta/Correlation Features (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `alternative_signals_beta_20d` | RISK | 20-day rolling beta to SPY |
| 5 | `alternative_signals_beta_change_rate` | RISK | Rate of change in beta |
| 6 | `alternative_signals_beta_vix_interaction` | RISK | Beta × VIX interaction term |
| 7 | `alternative_signals_spy_correlation_20d` | RISK | 20-day rolling correlation with SPY |
| 8 | `alternative_signals_qqq_correlation_20d` | RISK | 20-day rolling correlation with QQQ |
| 9 | `alternative_signals_sector_etf_correlation_20d` | RISK | 20-day rolling correlation with sector ETF |

### Volatility Features (9)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `alternative_signals_rv_5d` | RISK | 5-day realized volatility |
| 11 | `alternative_signals_rv_10d` | RISK | 10-day realized volatility |
| 12 | `alternative_signals_rv_20d` | RISK | 20-day realized volatility |
| 13 | `alternative_signals_rv_ratio_5_20` | RISK | Ratio of 5d to 20d realized volatility |
| 14 | `alternative_signals_rv_z_20` | RISK | Z-score of 20-day realized volatility |
| 15 | `alternative_signals_close_to_close_volatility` | RISK | Close-to-close volatility |
| 16 | `alternative_signals_open_to_close_volatility` | RISK | Open-to-close (intraday) volatility |
| 17 | `alternative_signals_high_low_volatility_ratio` | RISK | High-low range as volatility proxy |
| 18 | `alternative_signals_intraday_volatility_ratio` | RISK | Ratio of intraday to total volatility |

### Intraday/Overnight Features (9)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 19 | `alternative_signals_intraday_range_pct` | PREDICTIVE | Intraday range as % of price |
| 20 | `alternative_signals_intraday_range_z` | PREDICTIVE | Z-score of intraday range |
| 21 | `alternative_signals_opening_reversal` | PREDICTIVE | Opening reversal signal (gap fade) |
| 22 | `alternative_signals_closing_ramp` | PREDICTIVE | End-of-day momentum ramp |
| 23 | `alternative_signals_overnight_return` | PREDICTIVE | Overnight (close-to-open) return |
| 24 | `alternative_signals_overnight_return_z` | PREDICTIVE | Z-score of overnight return |
| 25 | `alternative_signals_gap_up_pct` | PREDICTIVE | Gap up percentage |
| 26 | `alternative_signals_gap_down_pct` | PREDICTIVE | Gap down percentage |
| 27 | `alternative_signals_gap_vs_vix_interaction` | PREDICTIVE | Gap × VIX interaction |

### Volume Features (7)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 28 | `alternative_signals_relative_volume_20d` | PREDICTIVE | Volume relative to 20-day average |
| 29 | `alternative_signals_volume_z_20d` | PREDICTIVE | Z-score of volume (20d lookback) |
| 30 | `alternative_signals_volume_trend_10d` | PREDICTIVE | 10-day volume trend (slope) |
| 31 | `alternative_signals_opening_volume_surge` | PREDICTIVE | Opening volume vs daily avg ratio |
| 32 | `alternative_signals_buy_volume_proxy` | PREDICTIVE | Estimated buy-side volume |
| 33 | `alternative_signals_volume_price_divergence` | PREDICTIVE | Price-volume divergence signal |
| 34 | `alternative_signals_liquidity_stress_pct` | RISK | Liquidity stress indicator |

### Earnings Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 35 | `alternative_signals_days_since_last_earnings` | REGIME | Days since last earnings announcement |
| 36 | `alternative_signals_days_to_next_earnings` | REGIME | Days until next earnings announcement |
| 37 | `alternative_signals_earnings_runup_10d` | PREDICTIVE | 10-day pre-earnings drift |
| 38 | `alternative_signals_post_earnings_drift_5d` | PREDICTIVE | 5-day post-earnings drift (PEAD) |

### News/Sentiment Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 39 | `alternative_signals_news_volume_count` | PREDICTIVE | Raw news article count |
| 40 | `alternative_signals_news_volume_change` | PREDICTIVE | Change in news volume |
| 41 | `alternative_signals_news_volume_z` | PREDICTIVE | Z-score of news volume |
| 42 | `alternative_signals_google_trends_score` | PREDICTIVE | Google Trends interest score |

### Signal Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 43 | `alternative_signals_mean_reversion_signal` | PREDICTIVE | Mean reversion indicator |
| 44 | `alternative_signals_trend_acceleration` | PREDICTIVE | Trend acceleration/momentum |
| 45 | `alternative_signals_turn_of_month_flag` | REGIME | Turn-of-month seasonality flag |

---

### Feature Role Classification Rules

Roles are assigned using ordered token-matching rules from [src/features/feature_roles.py](../src/features/feature_roles.py):

| Priority | Rule | Tokens | Role |
|----------|------|--------|------|
| A1 | HYGIENE | `has_data`, `coverage`, `is_valid`, `days_since_update`, `missing`, etc. | HYGIENE |
| A2 | REGIME | `regime`, `state`, `prob`, `calendar`, `month_end`, `quarter_end`, etc. | REGIME |
| A3 | RISK | `vol`, `volatility`, `correlation`, `beta`, `liquidity`, `spread`, `skew`, etc. | RISK |
| A4 | PREDICTIVE | `return`, `alpha`, `momentum`, `mean_reversion`, `sentiment`, `zscore`, `forecast`, etc. | PREDICTIVE |
| A5 | Fallback | (no token match) → use family primary intent | PREDICTIVE |

---

### Role Definitions

| Role | Purpose | Normalization | Alpha Usage |
|------|---------|---------------|-------------|
| **HYGIENE** | Data quality gating | None (pass-through) | Never |
| **REGIME** | Context/state classification | None (booleans/probs) | Gating only |
| **RISK** | Uncertainty/volatility signals | Winsorize + scale | Risk scaling |
| **PREDICTIVE** | Alpha signals | Z-score + clip | Conditional |

---

## arima_forecast Family — Full Feature Reference

> **Total Columns: 13** (10 features + 3 governance)  
> **Family Intent:** PREDICTIVE (not in DEFAULT_FAMILY_META, inferred)  
> **Update Cadence:** daily  
> **Model:** ARIMA/SARIMAX on LOG RETURNS (stationary)  
> **Exogenous Variables:** log_return_5d, log_return_10d, realized_vol_5d, realized_vol_20d

### Overview

The ARIMA forecast family provides time-series model features for mean-reversion structure in returns. Alpha comes from short-term autocorrelation patterns, residual shocks, and momentum indicators.

**Model Inputs (ALL STATIONARY):**
- **Endogenous:** `log_return_1d = log(close_t / close_{t-1})`
- **Exogenous:** Multi-period returns + realized volatility

**Model Selection:** AutoARIMA with hedge-fund constraints:
- AR: 1-3 terms, MA: 1-3 terms, d=0 (no differencing)
- Fallback: SARIMAX(2,0,1)

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `arima_forecast_has_data` | HYGIENE | Boolean flag: ARIMA model successfully fitted |
| 2 | `arima_forecast_activity` | HYGIENE | Activity score (0-1) |
| 3 | `arima_forecast_days_since_update` | HYGIENE | Days since last model update |

### Forecast Features (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `arima_forecast_1d` | PREDICTIVE | 1-day ahead return forecast (in-sample fitted values) |
| 5 | `arima_forecast_5d` | PREDICTIVE | 5-day ahead cumulative forecast (scaled by persistence) |

### Residual Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `arima_forecast_arima_residual_t` | PREDICTIVE | Current residual = actual - forecast |
| 7 | `arima_forecast_arima_abs_residual` | RISK | Absolute residual magnitude (volatility shift detector) |
| 8 | `arima_forecast_arima_residual_zscore` | PREDICTIVE | Standardized residual (shock indicator, 20d rolling) |
| 9 | `arima_forecast_arima_uncertainty_proxy` | RISK | Variance of past residuals (20d rolling) |

### Model Diagnostic Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `arima_forecast_arima_momentum_indicator` | PREDICTIVE | Sign of forecast: +1 (bullish), -1 (bearish), 0 (neutral) |
| 11 | `arima_forecast_arima_persistence` | REGIME | Sum of AR coefficients: >0.5=MOMENTUM, <0=MEAN-REVERSION |
| 12 | `arima_forecast_arima_innovation` | RISK | Std dev of innovations (regime change indicator) |
| 13 | `arima_forecast_arima_log_likelihood` | HYGIENE | Model log-likelihood (quality signal) |

### Additional Output (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| — | `arima_forecast_confidence` | HYGIENE | Model confidence score (if available) |

---

### Interpretation Guide

| Feature | Interpretation | Trading Signal |
|---------|----------------|----------------|
| `arima_forecast_1d` > 0 | Model expects positive return | Bullish |
| `arima_residual_zscore` > 2 | Positive surprise vs model | Potential continuation |
| `arima_residual_zscore` < -2 | Negative surprise vs model | Potential reversal |
| `arima_persistence` > 0.5 | Strong momentum regime | Trend-following |
| `arima_persistence` < 0 | Mean-reversion regime | Contrarian |
| `arima_abs_residual` spike | Volatility regime shift | Caution/reduce size |

---

### Model Constraints (Hedge-Fund Safe)

```python
auto_arima(
    start_p=1, max_p=3,      # AR(1) to AR(3)
    start_q=1, max_q=3,      # MA(1) to MA(3)
    d=0, max_d=0,            # Force d=0 (stationary input)
    seasonal=False,          # No seasonality
    information_criterion="aic",
    stepwise=True,
    maxiter=100,
    with_intercept=True,
)
```

---

## calibration Family — Full Feature Reference

> **Total Columns: 27** (24 features + 3 governance)  
> **Family Intent:** HYGIENE (model quality tracking)  
> **Update Cadence:** monthly (every 21 trading days, forward-filled)  
> **Horizon-Bound:** Yes  
> **Data Source:** ProbabilityCalibrationSystem evaluating quantile forecast accuracy

### Overview

The calibration family measures how well quantile forecasts match realized returns over rolling windows. This provides model quality signals that can:
- Gate trading when forecasts are unreliable
- Trigger model recalibration
- Weight ensemble members by accuracy

**Evaluation Window:** 180 days (~6 months) of historical forecast vs. actual comparisons

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `calibration_has_data` | HYGIENE | Boolean flag: calibration metrics available |
| 2 | `calibration_activity` | HYGIENE | Activity score (0-1) |
| 3 | `calibration_days_since_update` | HYGIENE | Days since last calibration update |

### Per-Quantile Coverage Features (8)

Measures what fraction of realized returns fell below each quantile threshold:

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `calibration_q05_coverage` | HYGIENE | Fraction of actuals < Q05 forecast (ideal: 0.05) |
| 5 | `calibration_q10_coverage` | HYGIENE | Fraction of actuals < Q10 forecast (ideal: 0.10) |
| 6 | `calibration_q25_coverage` | HYGIENE | Fraction of actuals < Q25 forecast (ideal: 0.25) |
| 7 | `calibration_q50_coverage` | HYGIENE | Fraction of actuals < Q50 forecast (ideal: 0.50) |
| 8 | `calibration_q75_coverage` | HYGIENE | Fraction of actuals < Q75 forecast (ideal: 0.75) |
| 9 | `calibration_q90_coverage` | HYGIENE | Fraction of actuals < Q90 forecast (ideal: 0.90) |
| 10 | `calibration_q95_coverage` | HYGIENE | Fraction of actuals < Q95 forecast (ideal: 0.95) |
| 11 | `calibration_q99_coverage` | HYGIENE | Fraction of actuals < Q99 forecast (ideal: 0.99) |

### Per-Quantile Error Features (8)

Absolute deviation from ideal coverage:

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `calibration_q05_error` | HYGIENE | |coverage - 0.05| (lower is better) |
| 13 | `calibration_q10_error` | HYGIENE | |coverage - 0.10| (lower is better) |
| 14 | `calibration_q25_error` | HYGIENE | |coverage - 0.25| (lower is better) |
| 15 | `calibration_q50_error` | HYGIENE | |coverage - 0.50| (lower is better) |
| 16 | `calibration_q75_error` | HYGIENE | |coverage - 0.75| (lower is better) |
| 17 | `calibration_q90_error` | HYGIENE | |coverage - 0.90| (lower is better) |
| 18 | `calibration_q95_error` | HYGIENE | |coverage - 0.95| (lower is better) |
| 19 | `calibration_q99_error` | HYGIENE | |coverage - 0.99| (lower is better) |

### Interval Calibration Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 20 | `calibration_interval_coverage` | HYGIENE | Actual coverage of prediction intervals |
| 21 | `calibration_interval_expected` | HYGIENE | Expected coverage (e.g., 0.90 for 90% interval) |
| 22 | `calibration_interval_error` | HYGIENE | |actual - expected| interval coverage error |

### Summary Metrics (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 23 | `calibration_mean_calibration_error` | HYGIENE | Average across all quantile errors |
| 24 | `calibration_overall_score` | HYGIENE | Composite calibration quality (0-1, higher is better) |
| 25 | `calibration_requires_recalibration` | HYGIENE | Binary flag: model needs recalibration |
| 26 | `calibration_sample_size` | HYGIENE | Number of samples used in evaluation |
| 27 | `calibration_confidence` | HYGIENE | Confidence based on sample size (clip(n/500, 0, 1)) |

---

### Interpretation Guide

| Metric | Good | Warning | Action |
|--------|------|---------|--------|
| `mean_calibration_error` | < 0.05 | 0.05-0.15 | > 0.15 → recalibrate |
| `overall_score` | > 0.8 | 0.5-0.8 | < 0.5 → reduce position size |
| `q50_coverage` | 0.45-0.55 | 0.4-0.6 | Outside → median forecast biased |
| `sample_size` | > 100 | 50-100 | < 50 → insufficient confidence |

---

## online_learning Family — Full Feature Reference

> **Total Columns: 35** (32 features + 3 governance)  
> **Family Intent:** PREDICTIVE (adaptive learning metrics)  
> **Update Cadence:** every 5 samples (incremental)  
> **Horizon-Bound:** Yes  
> **Data Source:** OnlineLearningSystem with ADWIN drift detection

### Overview

The online_learning family tracks how well models adapt to changing market regimes. Key alpha sources:
- **Drift detection** signals regime changes before they're obvious
- **Recalibration strength** indicates conviction in new regime
- **Trust scores** help weight ensemble members dynamically

**Drift Detection Methods:** ADWIN, Page-Hinkley

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `online_learning_has_data` | HYGIENE | Boolean flag: online learning active |
| 2 | `online_learning_activity` | HYGIENE | Activity score (0-1) |
| 3 | `online_learning_days_since_update` | HYGIENE | Days since last model update |

### Performance Metrics (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `online_learning_mae` | HYGIENE | Mean Absolute Error |
| 5 | `online_learning_rmse` | HYGIENE | Root Mean Squared Error |
| 6 | `online_learning_mape` | HYGIENE | Mean Absolute Percentage Error |
| 7 | `online_learning_direction_accuracy` | PREDICTIVE | Fraction of correct direction predictions |
| 8 | `online_learning_samples` | HYGIENE | Total samples processed |

### Drift Detection Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `online_learning_drift_flag` | REGIME | Binary: drift detected this period |
| 10 | `online_learning_drift_events` | REGIME | Cumulative drift events detected |
| 11 | `online_learning_partial_retrain_flag` | REGIME | Binary: partial retrain triggered |
| 12 | `online_learning_partial_retrains` | HYGIENE | Cumulative partial retrains |

### Model Update Tracking (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 13 | `online_learning_model_updated_flag` | REGIME | Binary: model updated this period |
| 14 | `online_learning_incremental_updates` | HYGIENE | Cumulative incremental updates |
| 15 | `online_learning_model_confidence` | PREDICTIVE | Current model confidence (0-1) |

### Trust & Reliability Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 16 | `online_learning_trust_score` | PREDICTIVE | Composite reliability score (0-1) |
| 17 | `online_learning_recalibration_needed` | HYGIENE | Binary: recalibration recommended |
| 18 | `online_learning_recalibration_strength` | PREDICTIVE | Strength of recalibration signal |

### Quantile Drift Features (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 19 | `online_learning_quantile_drift_avg` | RISK | Average drift across quantiles |
| 20 | `online_learning_quantile_drift_q50` | RISK | Drift in median quantile |
| 21 | `online_learning_quantile_drift_std` | RISK | Std dev of quantile drifts |
| 22 | `online_learning_quantile_spread_current` | RISK | Current quantile spread |
| 23 | `online_learning_quantile_spread_shock` | PREDICTIVE | Sudden change in spread |

### Calibration Asymmetry Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 24 | `online_learning_upside_calibration_error` | RISK | Calibration error for upside quantiles |
| 25 | `online_learning_downside_calibration_error` | RISK | Calibration error for downside quantiles |
| 26 | `online_learning_asymmetric_calibration_ratio` | PREDICTIVE | Upside/downside error ratio |
| 27 | `online_learning_prediction_skewness` | PREDICTIVE | Skewness in prediction errors |

### Uncertainty Compression Features (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 28 | `online_learning_uncertainty_compression_ratio` | RISK | Current vs. historical uncertainty |
| 29 | `online_learning_uncertainty_compression_alert` | REGIME | Binary: uncertainty abnormally low |

### Regime Indicators (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 30 | `online_learning_regime_trend_positive` | REGIME | Probability of positive trend regime |
| 31 | `online_learning_regime_trend_negative` | REGIME | Probability of negative trend regime |
| 32 | `online_learning_regime_trend_neutral` | REGIME | Probability of neutral/range-bound regime |
| 33 | `online_learning_regime_vol_high` | REGIME | Probability of high volatility regime |
| 34 | `online_learning_regime_vol_low` | REGIME | Probability of low volatility regime |
| 35 | `online_learning_regime_vol_normal` | REGIME | Probability of normal volatility regime |

---

### Interpretation Guide

| Signal | Condition | Action |
|--------|-----------|--------|
| Drift detected | `drift_flag = 1` | Reduce position size, await regime clarity |
| High trust | `trust_score > 0.7` | Full conviction on signals |
| Recalibration needed | `recalibration_needed = 1` | Weight down this model in ensemble |
| Uncertainty compression | `compression_alert = 1` | Warning: model overconfident |
| Direction accuracy drop | `< 0.5` | Model worse than random → invert or disable |

---

## quantile_forecast Family — Full Feature Reference

> **Total Columns: 25** (22 features + 3 governance)  
> **Family Intent:** PREDICTIVE  
> **Update Cadence:** daily  
> **Horizon-Bound:** Yes  
> **Model:** GBMQuantileForecaster (Gradient Boosting for quantile regression)  
> **Allow Direct Alpha:** true

### Overview

The quantile_forecast family provides distribution-aware return predictions. Instead of a single point forecast, it predicts the entire return distribution via quantiles. Alpha sources:
- **Tail quantiles** capture crash/rally probabilities
- **Spread dynamics** predict volatility regime changes
- **Skewness** indicates directional bias

**Input Features (ALL STATIONARY):**
- Log returns: 1d, 5d, 10d, 20d
- Realized volatility: 5d, 10d, 20d
- Price vs. SMA: 10, 20, 50
- Volume z-score and change
- Regime flags: low_vol, high_vol, breakout

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `quantile_forecast_has_data` | HYGIENE | Boolean flag: forecasts available |
| 2 | `quantile_forecast_activity` | HYGIENE | Activity score (0-1) |
| 3 | `quantile_forecast_days_since_update` | HYGIENE | Days since last forecast update |

### Raw Quantile Forecasts (8)

Predicted return distribution percentiles:

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `quantile_forecast_q05` | PREDICTIVE | 5th percentile (extreme downside) |
| 5 | `quantile_forecast_q10` | PREDICTIVE | 10th percentile (downside risk) |
| 6 | `quantile_forecast_q25` | PREDICTIVE | 25th percentile (bearish scenario) |
| 7 | `quantile_forecast_q50` | PREDICTIVE | 50th percentile (median forecast) |
| 8 | `quantile_forecast_q75` | PREDICTIVE | 75th percentile (bullish scenario) |
| 9 | `quantile_forecast_q90` | PREDICTIVE | 90th percentile (upside potential) |
| 10 | `quantile_forecast_q95` | PREDICTIVE | 95th percentile (extreme upside) |
| 11 | `quantile_forecast_q99` | PREDICTIVE | 99th percentile (tail upside) |

### High-Alpha Derived Features (9)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `quantile_forecast_q_low_5` | PREDICTIVE | Alias for Q05 (crash regime indicator) |
| 13 | `quantile_forecast_q_low_25` | PREDICTIVE | Alias for Q25 (downside skew capture) |
| 14 | `quantile_forecast_q_median_50` | PREDICTIVE | Alias for Q50 (central tendency) |
| 15 | `quantile_forecast_q_high_75` | PREDICTIVE | Alias for Q75 (upside momentum bias) |
| 16 | `quantile_forecast_q_high_95` | PREDICTIVE | Alias for Q95 (tail upside probability) |
| 17 | `quantile_forecast_q_spread_95_5` | RISK | Q95 - Q05 (predicts regime volatility) |
| 18 | `quantile_forecast_q_skewness_proxy` | PREDICTIVE | (Q75-Q50) - (Q50-Q25) (forecast skewness) |
| 19 | `quantile_forecast_q_tilt_direction` | PREDICTIVE | Sign of (Q75 - |Q25|) (bullish/bearish tilt) |
| 20 | `quantile_forecast_q_vol_forecast` | RISK | (Q95-Q05)/1.645 (implied volatility from quantiles) |

### Distribution Shape Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 21 | `quantile_forecast_width` | RISK | Q90 - Q10 (distribution width) |
| 22 | `quantile_forecast_skew` | PREDICTIVE | (Q90-Q50) - (Q50-Q10) (distribution skew) |
| 23 | `quantile_forecast_uncertainty` | RISK | Width / rolling_vol (normalized uncertainty) |

### HF Transformer Signals (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 24 | `quantile_forecast_hf_score` | PREDICTIVE | Transformed directional signal (-1 to +1) |
| 25 | `quantile_forecast_hf_conf` | HYGIENE | Confidence in HF signal (0 to 1) |

---

### Interpretation Guide

| Feature | Signal | Trading Implication |
|---------|--------|---------------------|
| `q50 > 0` | Positive median forecast | Bullish bias |
| `q_spread_95_5` expanding | Volatility regime shift | Reduce position size or hedge |
| `q_skewness_proxy > 0` | Right-skewed distribution | Upside momentum, favor longs |
| `q_skewness_proxy < 0` | Left-skewed distribution | Downside risk, favor shorts/hedges |
| `q_tilt_direction = +1` | Bullish tilt | Higher probability of upside |
| `hf_score > 0.5` | Strong bullish signal | High-confidence long |
| `uncertainty` spike | Model uncertainty high | Reduce conviction |

---

### Model Configuration

```python
ForecastConfig(
    forecast_horizon=63,  # Or custom horizon
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    use_macro=False,
    trend_dampening=0.95,
)
```

---

## candle_mechanics Family — Full Feature Reference

> **Total Columns: 39** (36 features + 3 governance)  
> **Family Intent:** PREDICTIVE (price action)  
> **Update Cadence:** daily  
> **Data Source:** OHLCV price bars  
> **Design:** Scale-free (ATR/range normalized), leak-safe (shifted baselines)

### Overview

The candle_mechanics family extracts continuous features from daily OHLCV bars. Unlike named chart patterns (doji, hammer, etc.), these features are:
- **Continuous** — gradient-friendly for deep learning
- **Scale-free** — normalized by ATR or range
- **Leak-safe** — all rolling baselines are shifted to avoid lookahead

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `candle_mechanics_has_data` | HYGIENE | Boolean flag: valid OHLCV data present |
| 2 | `candle_mechanics_activity` | HYGIENE | Activity score (0-1) |
| 3 | `candle_mechanics_days_since_update` | HYGIENE | Days since last update |

### Additional Metadata (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `candle_mechanics_confidence` | HYGIENE | Data quality confidence |

### Candle Anatomy Features (8)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `candle_mechanics_close_pos` | PREDICTIVE | Close position within range (-1 to +1) |
| 6 | `candle_mechanics_body_pct_range` | PREDICTIVE | Body as % of range (clipped ±5) |
| 7 | `candle_mechanics_upper_wick_pct_range` | PREDICTIVE | Upper wick as % of range |
| 8 | `candle_mechanics_lower_wick_pct_range` | PREDICTIVE | Lower wick as % of range |
| 9 | `candle_mechanics_wick_imbalance` | PREDICTIVE | (Upper - Lower wick) / range |
| 10 | `candle_mechanics_body_atr14` | PREDICTIVE | Body normalized by 14-day ATR |
| 11 | `candle_mechanics_range_atr14` | RISK | Range normalized by 14-day ATR |
| 12 | `candle_mechanics_atr_ratio_14_60` | RISK | ATR(14) / ATR(60) regime indicator |

### Gap & Extension Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 13 | `candle_mechanics_gap_atr14` | PREDICTIVE | Gap (open - prev_close) / ATR(14) |
| 14 | `candle_mechanics_close_vs_prev_close_atr14` | PREDICTIVE | Close extension from prev close / ATR |
| 15 | `candle_mechanics_high_vs_prev_close_atr14` | PREDICTIVE | High extension from prev close / ATR |
| 16 | `candle_mechanics_low_vs_prev_close_atr14` | PREDICTIVE | Low extension from prev close / ATR |

### Return & Volatility Features (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 17 | `candle_mechanics_logret_1d` | PREDICTIVE | 1-day log return |
| 18 | `candle_mechanics_ret_1d` | PREDICTIVE | 1-day percentage return |
| 19 | `candle_mechanics_ret_5d` | PREDICTIVE | 5-day percentage return |
| 20 | `candle_mechanics_rv_5` | RISK | 5-day realized volatility |
| 21 | `candle_mechanics_rv_20` | RISK | 20-day realized volatility |
| 22 | `candle_mechanics_rng_z_20` | PREDICTIVE | Range z-score vs 20-day baseline |

### Trend & Momentum Features (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 23 | `candle_mechanics_trend_3` | PREDICTIVE | 3-day cumulative log return |
| 24 | `candle_mechanics_trend_5` | PREDICTIVE | 5-day cumulative log return |
| 25 | `candle_mechanics_sign_sum_3` | PREDICTIVE | Sum of return signs (3-day) |
| 26 | `candle_mechanics_sign_sum_5` | PREDICTIVE | Sum of return signs (5-day) |
| 27 | `candle_mechanics_body_z_20` | PREDICTIVE | Body z-score vs 20-day baseline |
| 28 | `candle_mechanics_dir_change` | REGIME | Binary: direction changed from yesterday |

### Breakout Distance Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 29 | `candle_mechanics_dist_to_high_20_atr14` | PREDICTIVE | Distance to 20-day high / ATR |
| 30 | `candle_mechanics_dist_to_low_20_atr14` | PREDICTIVE | Distance to 20-day low / ATR |
| 31 | `candle_mechanics_dist_to_mean_20_atr14` | PREDICTIVE | Distance to 20-day mean / ATR |

### Bar Pattern Features (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 32 | `candle_mechanics_inside_bar` | REGIME | Binary: today's range inside yesterday's |
| 33 | `candle_mechanics_outside_bar` | REGIME | Binary: today's range engulfs yesterday's |

### Volume Features (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 34 | `candle_mechanics_vol_ratio_20` | PREDICTIVE | Volume / 20-day average volume |
| 35 | `candle_mechanics_vol_log_chg` | PREDICTIVE | Log change in volume |

### Seasonality Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 36 | `candle_mechanics_dow_sin` | REGIME | Day-of-week (sine encoding) |
| 37 | `candle_mechanics_dow_cos` | REGIME | Day-of-week (cosine encoding) |
| 38 | `candle_mechanics_moy_sin` | REGIME | Month-of-year (sine encoding) |
| 39 | `candle_mechanics_moy_cos` | REGIME | Month-of-year (cosine encoding) |

---

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Scale-free** | All features normalized by ATR(14) or range |
| **Leak-safe** | Rolling baselines are `.shift(1)` to avoid lookahead |
| **Continuous** | No binary pattern flags (doji, hammer, etc.) |
| **Bounded** | All features clipped to prevent gradient explosion |

---

## cboe_term Family — Full Feature Reference

> **Total Columns: 19** (16 features + 3 governance)  
> **Family Intent:** RISK (volatility term structure)  
> **Update Cadence:** daily  
> **Data Source:** CBOE VIX indices (VIX9D, VIX, VIX3M, VIX6M)  
> **Allow Direct Alpha:** false (risk scaling only)

### Overview

The cboe_term family captures VIX term structure dynamics — the relationship between short-term and long-term implied volatility. Alpha sources:
- **Contango/backwardation** signals regime and crash risk
- **Term slope changes** predict volatility regime shifts
- **Panic premium** detects short-term fear spikes

**VIX Indices Used:**
- VIX9D (^VIX9D): 9-day implied volatility
- VIX (^VIX): 30-day implied volatility
- VIX3M (^VIX3M): 3-month implied volatility
- VIX6M (^VIX6M): 6-month implied volatility

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `cboe_term_has_data` | HYGIENE | Boolean flag: VIX data available |
| 2 | `cboe_term_activity` | HYGIENE | Activity score (0-1) |
| 3 | `cboe_term_days_since_update` | HYGIENE | Days since last update |

### Term Slope Features (3)

Z-scored and winsorized at ±3σ:

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `cboe_term_vxst_vix_term_slope` | RISK | VIX9D - VIX (9-day vs 30-day spread, z-scored) |
| 5 | `cboe_term_vix_vxv_term_slope` | RISK | VIX - VIX3M (30-day vs 3-month spread, z-scored) |
| 6 | `cboe_term_vix_vxmt_term_slope` | RISK | VIX - VIX6M (30-day vs 6-month spread, z-scored) |

### Advanced Structure Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `cboe_term_normalized_term_slope` | RISK | (VIX6M - VIX) / (VIX6M + VIX), range-bounded |
| 8 | `cboe_term_vix_term_curvature` | RISK | (VIX6M - VIX) - (VIX - VIX9D), curve shape |
| 9 | `cboe_term_front_back_spread` | RISK | VIX9D - VIX3M, pure shock measurement |
| 10 | `cboe_term_panic_premium` | RISK | VIX9D / VIX, short-term panic ratio |

### Continuous Regime Indicators (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `cboe_term_vix_roll_yield` | RISK | (VIX3M - VIX) / VIX, futures roll cost/benefit |
| 12 | `cboe_term_vix_ratio_term` | RISK | VIX3M / VIX, term structure ratio |
| 13 | `cboe_term_vix_contango_strength` | RISK | (VIX3M - VIX) / VIX, regime strength |
| 14 | `cboe_term_vol_risk_premium` | RISK | IV - RV_20d, z-scored, capped at ±3σ |
| 15 | `cboe_term_vol_risk_premium_pct` | RISK | (IV - RV) / RV, z-scored, capped |

### Change/Shock Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 16 | `cboe_term_vix_term_slope_change_1d` | PREDICTIVE | 1-day change in term slope |
| 17 | `cboe_term_vix_term_slope_change_5d` | PREDICTIVE | 5-day change in term slope |
| 18 | `cboe_term_vix_curvature_change` | PREDICTIVE | 1-day change in curvature |
| 19 | `cboe_term_panic_premium_change` | PREDICTIVE | 1-day change in panic premium |

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `vix_vxv_term_slope` < 0 | **Contango** (normal) | Vol selling strategies favored |
| `vix_vxv_term_slope` > 0 | **Backwardation** (fear) | Reduce risk, hedges expensive |
| `panic_premium` > 1.2 | Short-term fear spike | Crash risk elevated |
| `front_back_spread` spike | Imminent vol event | Position for volatility |
| `vol_risk_premium` > 2σ | IV overpriced vs RV | Vol selling opportunity |
| `vix_contango_strength` < -0.1 | Strong backwardation | Crisis mode, reduce exposure |

---

### Removed Features

Binary flags were removed (hurt Mamba gradients):
- `contango_flag`
- `backwardation_flag`
- `curve_shape_flag`

Replaced with continuous alternatives above.

---

## corp_actions_splits Family — Full Feature Reference

**Source:** EODHD Corporate Actions Splits API  
**Columns:** 12 (9 features + 3 governance)  
**Philosophy:** Stock splits create regime changes in microstructure. Post-split periods often exhibit elevated volatility and altered trading patterns as price discovery occurs at new levels.

**Leakage Policy:** Split is known at effective date EOD → pulse applied next trading session.

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `corp_actions_splits_has_data` | HYGIENE | Boolean: split data available |
| 2 | `corp_actions_splits_activity` | HYGIENE | Activity score (0-1) |
| 3 | `corp_actions_splits_days_since_update` | HYGIENE | Days since last data update |

### Split Event Detection (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `corp_actions_splits_flag` | REGIME | Binary pulse: 1.0 on split effective session, 0.0 otherwise |
| 5 | `corp_actions_splits_ratio` | RISK | Split ratio (e.g., 4.0 for 4:1 split, 0.5 for 1:2 reverse) — affects microstructure stress |
| 6 | `corp_actions_splits_log_ratio` | REGIME | log(ratio) — better for ML gradients |
| 7 | `corp_actions_splits_days_since` | REGIME | Trading days since last split (9999 if never) — **WARNING: use recency instead** |
| 8 | `corp_actions_splits_recency` | REGIME | **NEW** Bounded recency intensity in [0,1]: exp(-min(days_since, 252)/20). Recent split → ~1, old/no split → ~0. Safe for role-aware averaging. |
| 9 | `corp_actions_splits_confidence` | HYGIENE | Data confidence score (0-1) |

### Post-Split Regime Windows (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `corp_actions_splits_post_5d` | REGIME | Binary: 1.0 if within 5 trading days of split |
| 11 | `corp_actions_splits_post_20d` | REGIME | Binary: 1.0 if within 20 trading days of split |

### Historical Context (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `corp_actions_splits_count_5y` | REGIME | Rolling 5-year count of split events |

---

### Stream Routing

| Stream | Columns | Notes |
|--------|---------|-------|
| **Portfolio** | All columns | HYGIENE/REGIME/RISK overlays, split_stress computation |
| **Mamba** (optional) | flag, log_ratio, post_5d, post_20d, recency | Enable via `CORP_SPLITS_MAMBA_OPTIONAL=flag,log_ratio,...` |

### Split Stress Execution Discipline

RoleAwareContext computes per-asset `split_stress` for explicit execution discipline:

```
split_stress = max(flag, post_5d, post_20d, recency) × |log_ratio|
```

Applied in phase2_stateful.py:
```
z *= (1 - k × split_stress)  # where k = PORTFOLIO_SPLIT_STRESS_K (default 0.3)
```

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `flag` = 1.0 | Split just occurred | Expect elevated volatility, altered microstructure |
| `post_5d` = 1.0 | Within first week of split | Price discovery phase, wider spreads |
| `post_20d` = 1.0 | Within first month of split | Transition period, stabilizing |
| `recency` > 0.8 | Very recent split (<10 days) | Strong microstructure effects |
| `recency` < 0.1 | Distant or no split (>60 days) | Normal microstructure |
| `ratio` > 2.0 | Large forward split | Often signals strong stock performance pre-split |
| `ratio` < 1.0 | Reverse split | Often signals distress, compliance requirements |
| `count_5y` > 2 | Frequent splitter | High-growth tech pattern |

---

## correlation Family — Full Feature Reference

**Source:** EODHD EOD Prices + Tiingo benchmarks (SPY, QQQ, VXX, sector ETFs)  
**Columns:** 31 (28 features + 3 governance)  
**Philosophy:** Correlation features are META-SIGNALS that condition risk, timing, and confidence. They measure how a stock moves relative to benchmarks and itself over time.

**Hedge-Fund Grade:** Compressed from 118 → 31 features. Continuous z-scores replace binary flags.

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `correlation_has_data` | HYGIENE | Boolean: correlation data available |
| 2 | `correlation_activity` | HYGIENE | Activity score (0-1) |
| 3 | `correlation_days_since_update` | HYGIENE | Days since last update |

### Traditional Rolling Correlations (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `corr_20_spy` | RISK | 20-day rolling correlation with S&P 500 (fast beta) |
| 5 | `corr_60_spy` | RISK | 60-day rolling correlation with S&P 500 (medium beta) |
| 6 | `corr_20_qqq` | RISK | 20-day correlation with Nasdaq 100 (tech/growth exposure) |
| 7 | `corr_20_vxx` | RISK | 20-day correlation with VIX (volatility/fear exposure) |
| 8 | `corr_20_sector` | RISK | 20-day correlation with sector ETF (XLK, XLE, etc.) |

### Decoupling Detection (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `corr_decoupling_z` | REGIME | Z-score of CORR_20_SPY vs 252-day mean. Negative = decoupling |
| 10 | `corr_spread_20_60_spy` | REGIME | CORR_20_SPY - CORR_60_SPY (short vs medium term) |

### Lagged Cross-Correlations (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `lag_corr_1_spy` | PREDICTIVE | 1-day SPY shock propagation (winsorized ±0.95) |
| 12 | `lag_corr_5_spy` | PREDICTIVE | 5-day SPY weekly spillover |
| 13 | `lag_corr_1_vxx` | PREDICTIVE | 1-day VIX shock (fear contagion) |
| 14 | `lag_corr_2_vxx` | PREDICTIVE | 2-day VIX spillover |

### Correlation Momentum/Trend (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 15 | `corr_20_spy_trend` | REGIME | SPY correlation slope over 5-day lookback |
| 16 | `corr_20_qqq_trend` | REGIME | QQQ correlation slope (not in current build) |
| 17 | `corr_20_vxx_trend` | REGIME | VXX correlation slope |

### Correlation Volatility (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 18 | `corr_20_spy_vol` | RISK | 20-day rolling std of SPY correlation |
| 19 | `corr_20_vxx_vol` | RISK | 20-day rolling std of VXX correlation |

### VIX Lag Correlations (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 20 | `corr_20_vix_lag1` | PREDICTIVE | 20-day correlation with VXX lagged 1 day |
| 21 | `corr_60_vix_lag1` | PREDICTIVE | 60-day correlation with VXX lagged 1 day |
| 22 | `corr_20_vix_lag2` | PREDICTIVE | 20-day correlation with VXX lagged 2 days |
| 23 | `corr_60_vix_lag2` | PREDICTIVE | 60-day correlation with VXX lagged 2 days |

### Cross-Feature Correlations (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 24 | `corr_return_vol_20` | REGIME | 20-day correlation: returns vs volume (accumulation/distribution) |
| 25 | `corr_return_range_10` | REGIME | 10-day correlation: returns vs daily range (trend/compression) |
| 26 | `corr_vol_volatility_20` | REGIME | 20-day correlation: volume vs volatility (panic/stealth) |

### Autocorrelation Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 27 | `acf_ret_1` | PREDICTIVE | 1-day return autocorrelation (momentum vs reversion) |
| 28 | `acf_ret_5` | PREDICTIVE | 5-day return autocorrelation (weekly pattern) |
| 29 | `acf_absret_1` | REGIME | 1-day absolute return autocorrelation (volatility clustering) |
| 30 | `acf_vol_1` | REGIME | 1-day volatility autocorrelation (volatility persistence) |

### Spread Features (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 31 | `corr_spread_20_60_qqq` | REGIME | CORR_20_QQQ - CORR_60_QQQ (if 60d QQQ available) |

---

### Stream Routing

| Stream | Columns | Notes |
|--------|---------|-------|
| **Mamba** (default) | `lag_corr_1_spy`, `lag_corr_5_spy`, `lag_corr_1_vxx`, `lag_corr_2_vxx`, `corr_20_vix_lag1`, `corr_60_vix_lag1`, `corr_20_vix_lag2`, `corr_60_vix_lag2`, `acf_ret_1`, `acf_ret_5` | Predictive spillovers and serial structure |
| **Mamba** (optional) | `corr_decoupling_z`, `corr_spread_20_60_spy` | Enable via `CORRELATION_MAMBA_OPTIONAL=corr_decoupling_z,...` |
| **Portfolio** | All other columns | RISK (exposure) and REGIME (instability) overlays |

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `corr_20_spy` > 0.8 | High beta | Stock moves with market, systemic risk |
| `corr_20_spy` < 0.3 | Decoupled | Stock-specific drivers dominate |
| `corr_decoupling_z` < -1.5 | Strong decoupling | Idiosyncratic event, reduced hedge effectiveness |
| `corr_decoupling_z` > +1.5 | Strong fusion | Macro-driven, increase hedges |
| `corr_20_vxx` > 0.3 | Fear-correlated | Crashes with volatility spikes |
| `corr_20_vxx` < -0.2 | Fear-inverse | Defensive/haven characteristics |
| `lag_corr_1_spy` high | Lagged beta | Stock follows market with delay |
| `acf_ret_1` > 0.1 | Momentum present | Trend-following viable |
| `acf_ret_1` < -0.1 | Mean reversion | Contrarian strategies viable |
| `corr_return_vol_20` > 0.3 | Volume confirms moves | Strong price action |
| `corr_return_vol_20` < -0.3 | Divergence | Distribution/accumulation phase |

---

## cross_asset Family — Full Feature Reference

**Source:** EODHD EOD Prices (SPY, QQQ, VIX.INDX, TNX.INDX, IRX.INDX, HYG, LQD, UUP, sector ETFs)  
**Columns:** 26 (23 features + 3 governance)  
**Philosophy:** Level tells you where you are, CHANGE tells you when it breaks. Macro risk-on/off and shock propagation signals.

**Design Principle:** Addresses 2021-2022 beta regime failures. All change features for early regime detection.

**CRITICAL: Absolute Magnitude Conversion**  
Regime-change fields are converted to **ABSOLUTE MAGNITUDE** for proper stress aggregation.
RoleAwareContext treats large values as "stress"; signed values confuse the aggregator.
- `beta_change_rate` → abs(zscore) of beta change rate
- `beta_volatility_change` → abs(pct_change) of beta stability
- `tnx_corr_change_5d` → abs(change) in rate correlation
- `irx_corr_change_5d` → abs(change) in rate correlation
- `cross_asset_coupling_change` → abs(change) in coupling factor

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `cross_asset_has_data` | HYGIENE | Boolean: cross-asset data available |
| 2 | `cross_asset_activity` | HYGIENE | Activity score (0-1) |
| 3 | `cross_asset_days_since_update` | HYGIENE | Days since last update |

### ETF-Based Correlations (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `cross_asset_spy_corr_20d` | RISK | 20-day rolling correlation with S&P 500 |
| 5 | `cross_asset_qqq_corr_20d` | RISK | 20-day rolling correlation with Nasdaq 100 |
| 6 | `cross_asset_sector_etf_corr_20d` | RISK | 20-day correlation with sector ETF (XLK, XLY, XLF, etc.) |

### Beta & Factor Exposure (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `cross_asset_beta_20d` | RISK | Rolling 20-day beta to SPY (systematic risk) |
| 8 | `cross_asset_beta_change_rate` | REGIME | **ABSOLUTE MAGNITUDE**: abs(zscore) of beta change rate (0 to 3) |
| 9 | `cross_asset_beta_volatility_20d` | RISK | Rolling std of beta estimates (stability measure) |
| 10 | `cross_asset_beta_volatility_change` | REGIME | **ABSOLUTE MAGNITUDE**: abs(pct_change) of beta stability (0 to 5) |
| 11 | `cross_asset_beta_sign_flip_flag` | REGIME | Binary: 1 if beta crossed zero (sign flip) |

### Volatility Relationships (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `cross_asset_vix_corr_20d` | RISK | 20-day correlation with VIX (fear sensitivity) |
| 13 | `cross_asset_vix_spread_indicator` | RISK | VIX / Realized Vol ratio (implied vs realized) |
| 14 | `cross_asset_realized_vol_vs_spy_corr` | RISK | Volatility regime co-movement with market |

### Lead/Lag Analysis (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 15 | `cross_asset_spy_leads_stock_5d` | PREDICTIVE | Correlation of SPY(t-1) with STOCK(t) — market leads |
| 16 | `cross_asset_stock_leads_spy_5d` | PREDICTIVE | Correlation of STOCK(t-1) with SPY(t) — sector leadership |

### Rate Sensitivity (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 17 | `cross_asset_tnx_corr_20d` | RISK | 20-day correlation with 10Y Treasury yields |
| 18 | `cross_asset_irx_corr_20d` | RISK | 20-day correlation with 3M T-bills |
| 19 | `cross_asset_tnx_corr_change_5d` | REGIME | **ABSOLUTE MAGNITUDE**: abs(change) in 10Y rate correlation (0 to 1) |
| 20 | `cross_asset_irx_corr_change_5d` | REGIME | **ABSOLUTE MAGNITUDE**: abs(change) in 3M rate correlation (0 to 1) |

### Credit Spread Dynamics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 21 | `cross_asset_credit_spread_level` | RISK | HYG/LQD ratio deviation from 1-year baseline |
| 22 | `cross_asset_asset_corr_hyg_60` | RISK | 60-day correlation with high-yield credit |

### FX Risk-On/Off (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 23 | `cross_asset_asset_corr_uup_60` | RISK | 60-day correlation with US Dollar Index |

### Composite Risk Factor (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 24 | `cross_asset_risk_onoff_factor` | REGIME | Fixed-weight combo: 0.5×SPY + 0.3×QQQ - 0.2×UUP - 0.2×VIX (directional, for policy state only) |
| 25 | `cross_asset_coupling_change` | REGIME | **ABSOLUTE MAGNITUDE**: abs(5-day change) in risk factor (0 to 1) |
| 26 | `cross_asset_risk_offness` | RISK | **NEW**: One-sided stress = clip((0.5 - risk_onoff_factor) / 0.5, 0, 1). Risk-off → positive stress; risk-on → zero. |

---

### Stream Routing

| Stream | Columns | Notes |
|--------|---------|-------|
| **Mamba** | `spy_leads_stock_5d`, `stock_leads_spy_5d` | Per-asset timing/leadership alpha |
| **Portfolio** | All other columns | RISK/REGIME overlays, policy state |

### Critical Fix: risk_offness for Overlays

`risk_onoff_factor` is **directional** (higher = risk-on). Feeding it raw into stress aggregation would *reduce* exposure on risk-on days, which is backwards.

**Solution**: Split into two signals:
- `risk_onoff_factor` → Keep for Policy Controller state (directional)
- `risk_offness` → One-sided stress for overlays: `clip((0.5 - risk_onoff_factor) / 0.5, 0, 1)`

This ensures:
- Risk-off (factor < 0.5) → positive stress → lower exposure
- Risk-on (factor ≥ 0.5) → zero stress → no penalty

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `beta_20d` > 1.5 | High systematic risk | Reduce in risk-off regimes |
| `beta_sign_flip_flag` = 1 | Beta regime break | Major factor rotation occurring |
| `beta_change_rate` > 2 | Rapid beta shift (abs magnitude) | Defensive posture, hedge |
| `vix_corr_20d` > 0.3 | Fear-correlated | Crashes with vol spikes |
| `vix_spread_indicator` > 1.5 | IV overpriced vs RV | Vol selling opportunity |
| `spy_leads_stock_5d` high | Lagging beta | Stock follows market with delay |
| `stock_leads_spy_5d` high | Sector leader | Stock may forecast market moves |
| `tnx_corr_change_5d` > 0.3 | Rate sensitivity shift (abs) | Duration regime change |
| `credit_spread_level` < -0.1 | Credit stress | Risk-off, defensive positioning |
| `risk_offness` > 0.5 | Risk-off regime | Reduce exposure via overlays |
| `risk_onoff_factor` > 0.5 | Strong risk-on | Aggressive positioning viable (policy state) |

---

## dcf Family — Full Feature Reference

**Source:** EODHD EOD Prices (price-based only, no fundamental data)  
**Columns:** 28 (25 features + 3 governance)  
**Philosophy:** Price-relative ANCHORING signals for conditioning confidence, modulating position sizing, and trend/mean-reversion regime detection. NOT true DCF valuation.

**Governance:** Low Stage-A weight. Makes primary alpha drivers (alternative_signals, correlation, cboe_term) SAFER and more ADAPTIVE.

**CRITICAL: One-Sided Stress Signals**  
Raw anchor ratios are asymmetric and can dominate aggregation. Log versions are symmetric around 0.
RoleAwareContext treats high values as "stress"; directional signals need conversion:
- `dcf_overextension` = clip(log_p2fv_1y, 0, 1) — only penalize "too expensive"
- `dcf_downside_skew_stress` = clip(-scenario_skew, 0, 1) — only penalize downside asymmetry
- `dcf_undervaluation` = clip(log_p2fv_1y, -1, 0) — directional alpha for Mamba (NOT stress)

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `dcf_has_data` | HYGIENE | Boolean: DCF data available |
| 2 | `dcf_activity` | HYGIENE | Activity score (0-1) |
| 3 | `dcf_days_since_update` | HYGIENE | Days since last update |
| 4 | `dcf_confidence` | HYGIENE | Data confidence score (0-1) |

### Valuation Anchors — Raw Ratios (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `dcf_price_to_fairvalue_1y` | REGIME | Price / 252-day EMA (slow fair value anchor) |
| 6 | `dcf_price_to_fairvalue_3m` | REGIME | Price / 63-day EMA (fast fair value anchor) |
| 7 | `dcf_price_regime` | REGIME | Price / 20-day Bollinger mid (short-term positioning) |

### Valuation Anchors — Log Versions (3) — NEW

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `dcf_log_p2fv_1y` | REGIME | log(price / 252-day EMA) — symmetric around 0 |
| 9 | `dcf_log_p2fv_3m` | REGIME | log(price / 63-day EMA) — symmetric around 0 |
| 10 | `dcf_log_price_regime` | REGIME | log(price / BB_mid) — symmetric around 0 |

### One-Sided Stress + Alpha (2) — NEW

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `dcf_overextension` | RISK | **One-sided stress**: clip(log_p2fv_1y, 0, 1). High = overextended = reduce exposure |
| 12 | `dcf_undervaluation` | PREDICTIVE | **Directional alpha**: clip(log_p2fv_1y, -1, 0). More negative = more undervalued (opportunity) |

### Multi-Horizon Momentum (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `dcf_price_to_fairvalue_1y` | REGIME | Price / 252-day EMA (slow fair value anchor) |
| 6 | `dcf_price_to_fairvalue_3m` | REGIME | Price / 63-day EMA (fast fair value anchor) |
| 7 | `dcf_price_regime` | REGIME | Price / 20-day Bollinger mid (short-term positioning) |

### Multi-Horizon Momentum (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `dcf_mom_1m` | PREDICTIVE | 21-day price momentum ratio |
| 9 | `dcf_mom_3m` | PREDICTIVE | 63-day price momentum ratio |
| 10 | `dcf_mom_12m` | PREDICTIVE | 252-day price momentum ratio |
| 11 | `dcf_mom_vol_adjusted` | PREDICTIVE | 21-day momentum / realized vol (winsorized ±10) |
| 12 | `dcf_mom_sharped` | PREDICTIVE | 21-day momentum / downside vol (winsorized ±10) |

### Mean Reversion Z-Scores (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 13 | `dcf_zscore_1m` | PREDICTIVE | (Price - 21d MA) / 21d std |
| 14 | `dcf_zscore_3m` | PREDICTIVE | (Price - 63d MA) / 63d std |

### Trend Quality (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 15 | `dcf_trend_slope_1m` | REGIME | 21-day price slope / price (normalized trend) |
| 16 | `dcf_trend_stability` | REGIME | |slope| / std (trend-to-noise ratio) |

### Volatility-Adjusted Value (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 17 | `dcf_vol_adjusted_value` | RISK | Valuation ratio / 63-day annualized vol |
| 18 | `dcf_value_momentum_ratio` | PREDICTIVE | 63-day momentum / 63-day vol |

### Scenario Awareness (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 19 | `dcf_scenario_spread` | RISK | Bull-bear scenario width / fair value (uncertainty) |
| 20 | `dcf_scenario_skew` | PREDICTIVE | (Upside - Downside) / (Upside + Downside) — directional alpha for Mamba |
| 21 | `dcf_downside_skew_stress` | RISK | **NEW**: clip(-scenario_skew, 0, 1) — one-sided stress for overlays |
| 22 | `dcf_scenario_position` | REGIME | Current position in bull-bear range (0=bear, 1=bull) |

### Horizon Structure (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 23 | `dcf_terminal_value_pct` | REGIME | 252d MA / 63d MA (growth vs value duration proxy) |

---

### Stream Routing

| Stream | Columns | Notes |
|--------|---------|-------|
| **Mamba** | `mom_1m`, `mom_3m`, `mom_12m`, `mom_vol_adjusted`, `mom_sharped`, `zscore_1m`, `zscore_3m`, `value_momentum_ratio`, `undervaluation`, `scenario_skew` | Directional alpha signals |
| **Portfolio** | All other columns | RISK/REGIME overlays, one-sided stress |

### Critical Fix: One-Sided Stress Signals

**Problem**: Raw ratios like `price_to_fairvalue_1y` are asymmetric. Values > 1 (overextended) and < 1 (undervalued) have different meanings, but stress aggregation treats both as "badness".

**Solution**: Split into log space + one-sided signals:
- `log_p2fv_1y` = log(price_to_fairvalue_1y) — symmetric around 0
- `overextension` = clip(log_p2fv_1y, 0, 1) — only penalize overextension
- `undervaluation` = clip(log_p2fv_1y, -1, 0) — directional alpha (NOT stress)
- `downside_skew_stress` = clip(-scenario_skew, 0, 1) — only penalize downside asymmetry

This ensures:
- Overextended stocks → higher stress → lower exposure via overlays
- Undervalued stocks → zero stress (potential opportunity via Mamba alpha)
- Downside-skewed distributions → higher stress → lower exposure

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `overextension` > 0.5 | Extended above fair value (log > 0.5) | Reduce exposure via overlays |
| `undervaluation` < -0.3 | Undervalued (log < -0.3) | Potential opportunity (Mamba alpha) |
| `mom_vol_adjusted` > 5 | Very strong risk-adjusted momentum | Trend continuation likely |
| `mom_vol_adjusted` < -5 | Strong negative momentum | Avoid catching falling knife |
| `zscore_1m` > 2 | Overbought (2σ above mean) | Mean reversion candidate |
| `zscore_1m` < -2 | Oversold (2σ below mean) | Bounce candidate |
| `trend_stability` > 3 | Clean trend, low noise | Trend-following effective |
| `trend_stability` < 1 | Noisy, no clear trend | Range-trading or avoid |
| `scenario_skew` > 0.3 | More upside than downside | Favorable risk/reward (Mamba) |
| `downside_skew_stress` > 0.5 | Downside-heavy distribution | Reduce exposure via overlays |
| `scenario_position` < 0.2 | Near bear scenario | Potential bottom (with confirmation) |
| `scenario_position` > 0.8 | Near bull scenario | Potential top (with confirmation) |
| `terminal_value_pct` > 1.1 | Mature/value stock | Short duration, rate sensitive |

---

## dividends Family — Full Feature Reference

**Source:** EODHD Dividends API  
**Columns:** 13 (9 features + 4 governance)  
**Philosophy:** Event timing + intensity, NOT dividend policy/quality. Policy features belong in FIN_G7.

**Hedge-Fund Grade:** Binary ex_dividend_flag removed → replaced with continuous window_strength.

### Stream Routing (Jan 2026)

| Stream | Columns | Purpose |
|--------|---------|---------|
| **Portfolio** (default) | All governance, yield metrics, event_stress, regime | Risk/regime overlays + policy state |
| **Mamba** (conditional) | `dividend_event_intensity` | Only if labels are dividend-adjusted AND shifted |
| **Mamba** (optional) | `ex_dividend_window_strength`, `days_to_ex_dividend` | Enable via `DIVIDENDS_MAMBA_OPTIONAL` or `MAMBA_OPTIONAL_COLUMNS` |

**CRITICAL: Label Pipeline Check**
- If returns use **raw close**: dividend-event signals create fake alpha (model "learns" ex-date drop)
- If returns use **adjusted close**: safe to include event_intensity in Mamba
- Action: Confirm return source before enabling dividend features in Mamba

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `dividends_has_data` | HYGIENE | Boolean: dividend data available |
| 2 | `dividends_activity` | HYGIENE | Activity score (0-1) |
| 3 | `dividends_days_since_update` | HYGIENE | Days since last data update |
| 4 | `dividends_confidence` | HYGIENE | Data confidence score (0-1) |

### Corporate Action (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `dividends_dividend_amount` | REGIME | Dividend amount (winsorized for special dividends at 99th pct) |

### Income Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `dividends_dividend_yield_est` | RISK | Trailing 252-day rolling dividend yield estimate |
| 7 | `dividends_dividend_yield_zscore` | RISK | Z-score of yield vs 252-day rolling mean (clipped ±3) |

### Event Timing (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `dividends_days_to_ex_dividend` | REGIME | Signed days to next ex-dividend date (clipped ±30) |
| 9 | `dividends_ex_dividend_window_strength` | REGIME | exp(-|days_to_ex| / τ), τ=2.5 days (smooth event salience) |
| 10 | `dividends_dividend_event_intensity` | PREDICTIVE | (amount / price) × window_strength (time-localized impact) |
| 11 | `dividends_dividend_frequency` | REGIME | Annual dividend count (4=quarterly, 12=monthly) |

### Event Stress (1) — NEW

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `dividends_dividend_event_stress` | RISK | **One-sided stress**: window_strength × clip(|amount/price|, 0, 0.10) × 10 → [0,1]. High near large dividend events. |

---

### Design Rationale: One-Sided Event Stress

```
dividend_event_stress = window_strength × clip(|dividend_amount / price|, 0, 0.10) × 10
```

- **Window_strength**: Peaks at ex-date (τ=2.5 days), decays smoothly
- **Yield normalization**: Caps at 10% yield to handle special dividends
- **Scaling**: × 10 normalizes to [0, 1] range for stress aggregation
- **Portfolio overlay**: Use to reduce position size near dividend events
- **Policy state**: Aggregate across universe for market-wide dividend stress

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `days_to_ex_dividend` < 5 | Approaching ex-date | Dividend capture opportunity |
| `days_to_ex_dividend` < 0 | Post ex-date | Gap-down expected, dividend priced out |
| `ex_dividend_window_strength` > 0.5 | Near ex-date (±2d) | Event-driven volatility likely |
| `dividend_event_intensity` spike | Large dividend imminent | Significant price impact expected |
| `dividend_event_stress` > 0.5 | Near large dividend | Reduce position size (portfolio overlay) |
| `dividend_yield_zscore` > 2 | Yield unusually high | Possible distress or special dividend |
| `dividend_yield_zscore` < -2 | Yield unusually low | Growth mode or dividend cut risk |
| `dividend_frequency` = 0 | No dividend history | Non-dividend payer (growth stock) |
| `dividend_frequency` ≥ 4 | Regular quarterly payer | Income-oriented, defensive |

---

## earnings Family — Full Feature Reference

**Source:** EODHD Fundamentals (via EarningsAnalyzer)  
**Columns:** 21 (17 features + 4 governance)  
**Philosophy:** Surprises + consistency + revisions, NOT raw levels. Raw EPS/revenue removed (scale issues, not cross-sectionally comparable).

**Hedge-Fund Grade:** Growth winsorized ±200%. Binary flags → continuous. Event decay added. **Surprises SHIFTED by 1 day to prevent leakage.**

### Stream Routing (Jan 2026)

| Stream | Columns | Purpose |
|--------|---------|---------|
| **Mamba** (default) | surprises, growth, beat patterns, revisions, event_decay | Alpha signals for μ/σ conditioning |
| **Portfolio** (default) | governance, miss_streak, dispersion, surprise_volatility, event stress | Risk/regime overlays + policy state |
| **Mamba** (optional) | `days_since_earnings`, `days_to_next_earnings` | Enable via `EARNINGS_MAMBA_OPTIONAL` or `MAMBA_OPTIONAL_COLUMNS` |

**CRITICAL: Leakage Prevention**
- Surprise features (`eps_surprise_pct`, `revenue_surprise_pct`) are **SHIFTED by 1 day**
- If day t bar includes earnings reaction, surprise is only known after release
- Safe rule: surprise at t can only be used for decision at t+1

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `earnings_has_data` | HYGIENE | Boolean: earnings data available |
| 2 | `earnings_activity` | HYGIENE | Activity score (0-1) |
| 3 | `earnings_days_since_update` | HYGIENE | Days since last data update |
| 4 | `earnings_confidence` | HYGIENE | Data confidence score (0-1) |

### Core Surprise Features (2) — SHIFTED BY 1 DAY

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `earnings_eps_surprise_pct` | PREDICTIVE | (Actual - Estimate) / |Estimate| × 100 **[SHIFTED +1 day]** |
| 6 | `earnings_revenue_surprise_pct` | PREDICTIVE | Revenue surprise percentage **[SHIFTED +1 day]** |

### Growth Trends (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `earnings_eps_growth_qoq` | PREDICTIVE | Quarter-over-quarter EPS growth (winsorized ±200%) |
| 8 | `earnings_eps_growth_yoy` | PREDICTIVE | Year-over-year EPS growth (winsorized ±200%) |
| 9 | `earnings_revenue_growth_qoq` | PREDICTIVE | Quarter-over-quarter revenue growth (winsorized ±200%) |
| 10 | `earnings_revenue_growth_yoy` | PREDICTIVE | Year-over-year revenue growth (winsorized ±200%) |

### Pattern Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `earnings_beat_streak` | PREDICTIVE | Consecutive quarters of beats (positive = beats) |
| 12 | `earnings_miss_streak` | RISK | Consecutive quarters of misses (positive = misses) |
| 13 | `earnings_surprise_volatility` | RISK | Rolling std of surprise_pct (replaces binary flag) |

### Beat Consistency (2) — HIGH ALPHA

| # | Column | Role | Description |
|---|--------|------|-------------|
| 14 | `earnings_beat_rate_3y` | PREDICTIVE | 3-year beat rate (0-1) — management conservatism proxy |
| 15 | `earnings_surprise_percent` | PREDICTIVE | Alias for eps_surprise_pct |

### Revision Dynamics (2) — CRITICAL ALPHA

| # | Column | Role | Description |
|---|--------|------|-------------|
| 16 | `earnings_revision_breadth` | PREDICTIVE | Net analyst revisions (up - down) / total |
| 17 | `earnings_estimate_dispersion` | RISK | Analyst estimate std / mean — uncertainty proxy |

### Event Timing (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 18 | `earnings_days_since_earnings` | REGIME | Trading days since last earnings release |
| 19 | `earnings_days_to_next_earnings` | REGIME | Trading days until next earnings (pre-event positioning) |
| 20 | `earnings_event_decay` | PREDICTIVE | exp(-days_since / τ), τ≈10-15 days (PEAD capture) |

### Event Stress (2) — NEW

| # | Column | Role | Description |
|---|--------|------|-------------|
| 21 | `earnings_pre_event_stress` | RISK | **One-sided stress**: exp(-days_to_next / τ_pre), τ_pre=7 days. Peaks before earnings, reduces position size. |
| 22 | `earnings_post_event_stress` | RISK | **One-sided stress**: exp(-days_since / τ_post), τ_post=5 days. Suppresses over-sizing during post-event turbulence. |

---

### Design Rationale: Event Stress Signals

```
pre_event_stress  = exp(-days_to_next_earnings / 7.0)
post_event_stress = exp(-days_since_earnings / 5.0)
```

- **Pre-event stress**: Peaks when approaching earnings (uncertainty high)
  - Use to reduce position size before event
  - Policy state: aggregate across universe for market-wide earnings stress
  
- **Post-event stress**: Peaks immediately after earnings, decays over ~5 days
  - Separate from PEAD alpha (which Mamba captures via event_decay × surprise)
  - Suppresses over-sizing during post-event volatility
  
- **Portfolio overlay application**: Both are RISK signals → stress aggregator → reduce exposure

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `eps_surprise_pct` > 10% | Large beat | Post-earnings drift likely (PEAD) |
| `eps_surprise_pct` < -10% | Large miss | Negative drift, potential gap-fill |
| `beat_streak` ≥ 4 | Consistent beater | Management sandbagging estimates |
| `miss_streak` ≥ 2 | Execution problems | Credibility gap, analyst downgrades |
| `beat_rate_3y` > 0.8 | Reliable beater | HIGH ALPHA: consistent execution |
| `revision_breadth` > 0.3 | Net upgrades | Forward momentum, positive sentiment |
| `revision_breadth` < -0.3 | Net downgrades | Headwinds, negative revision cycle |
| `estimate_dispersion` > 0.15 | High uncertainty | Wide outcome range, vol opportunity |
| `days_to_next_earnings` < 10 | Earnings imminent | Pre-earnings volatility spike expected |
| `pre_event_stress` > 0.5 | Within ~5 days of earnings | Reduce position size (portfolio overlay) |
| `post_event_stress` > 0.5 | Within ~3 days after earnings | Suppress over-sizing, let volatility settle |
| `event_decay` > 0.5 | Fresh earnings (<10d) | PEAD still active, momentum trades |
| `event_decay` < 0.1 | Stale earnings (>25d) | Event impact faded |

---

## econ_events_calendar Family — Full Feature Reference

**Source:** EODHD Economic Events Data API  
**Columns:** 127 (121 features + 6 governance) — Updated Jan 2026  
**Philosophy:** Symbol-agnostic macro release schedule. "No lookahead" policy: surprises applied on next NYSE session after release.

**Coverage:** EODHD economic-events coverage starts ~2020. Family emitted per-symbol for pipeline consistency but data is macro (not stock-specific).

**Critical Fixes (Jan 2026):**
- Replaced 9999 sentinels with bounded proximity/recency signals (safe for aggregation)
- Added composite macro features for policy state (macro_upcoming_major, macro_shock_major)
- Proper stream routing: compact context to Mamba, regime/risk overlays to portfolio

**Event Types Tracked (9):**
1. `cpi` - Consumer Price Index
2. `core_cpi` - Core Consumer Price Index (excluding food/energy)
3. `pce` - Personal Consumption Expenditures
4. `nfp` - Nonfarm Payrolls
5. `unemployment` - Unemployment Rate
6. `gdp` - Gross Domestic Product
7. `fomc` - FOMC Interest Rate Decisions
8. `retail_sales` - Retail Sales
9. `ism` - ISM Manufacturing/Services PMI

### Stream Routing (Jan 2026)

| Stream | Columns | Purpose |
|--------|---------|---------|
| **Mamba** (default) | `pulse_surprise_*`, `pulse_strength_*`, `prox_next_*` | Compact macro context (shifted, bounded) |
| **Mamba** (optional) | `surprise_z_*` | Enable via `ECON_EVENTS_MAMBA_OPTIONAL` |
| **Portfolio** (default) | All governance, windows, occurrences, recency_*, composites | Risk/regime overlays + policy state |

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `econ_events_calendar_has_data` | HYGIENE | Boolean: economic events data available |
| 2 | `econ_events_calendar_activity` | HYGIENE | Activity score (0-1) |
| 3 | `econ_events_calendar_days_since_update` | HYGIENE | Days since last data update |
| 4 | `econ_events_calendar_confidence` | HYGIENE | Data confidence score (0-1) |

### Per-Event Features (showing structure for `cpi`; repeats for all 9 event types)

#### Schedule Features — DEPRECATED (2 × 9 = 18)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `econ_events_calendar_days_to_next_cpi` | REGIME | **DEPRECATED**: Trading days to next release (9999 if none) — use `prox_next_*` instead |
| 6 | `econ_events_calendar_days_since_last_cpi` | REGIME | **DEPRECATED**: Trading days since last release (9999 if none) — use `recency_last_*` instead |

#### Bounded Proximity/Recency — NEW (2 × 9 = 18)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `econ_events_calendar_prox_next_cpi` | PREDICTIVE | **NEW**: exp(-min(days_to_next, 252) / 5). Peaks when event imminent. Safe for aggregation. |
| 8 | `econ_events_calendar_recency_last_cpi` | REGIME | **NEW**: exp(-min(days_since_last, 252) / 10). Peaks after event, decays over ~2 weeks. |

#### Event Windows (2 × 9 = 18)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `econ_events_calendar_pre_window_3d_cpi` | REGIME | Binary: 1 if within 3 sessions before release |
| 10 | `econ_events_calendar_post_window_3d_cpi` | REGIME | Binary: 1 if within 3 sessions after release |

#### Surprise Features (4 × 9 = 36)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `econ_events_calendar_surprise_cpi` | PREDICTIVE | (Actual - Forecast) raw surprise |
| 12 | `econ_events_calendar_surprise_z_cpi` | PREDICTIVE | Z-score of surprise vs 5-year rolling mean |
| 13 | `econ_events_calendar_surprise_w_cpi` | PREDICTIVE | Weighted surprise (surprise × importance weight) |
| 14 | `econ_events_calendar_surprise_w_z_cpi` | PREDICTIVE | Z-score of weighted surprise |

#### Meta Features (5 × 9 = 45)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 15 | `econ_events_calendar_surprise_has_forecast_cpi` | HYGIENE | Binary: 1 if forecast data available |
| 16 | `econ_events_calendar_event_occurrence_cpi` | REGIME | Binary: 1 on release date (unshifted) |
| 17 | `econ_events_calendar_pulse_occurrence_cpi` | REGIME | Binary: 1 on next session after release (shifted) |
| 18 | `econ_events_calendar_pulse_surprise_cpi` | PREDICTIVE | Surprise value on pulse session (shifted) |
| 19 | `econ_events_calendar_pulse_strength_cpi` | PREDICTIVE | |surprise_w_z| on pulse session (magnitude only) |

### Composite Macro Features — NEW (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 120 | `econ_events_calendar_macro_upcoming_major` | RISK | **NEW**: max(prox_next_{fomc, cpi, nfp}) — peaks when major event is imminent |
| 121 | `econ_events_calendar_macro_shock_major` | RISK | **NEW**: max(pulse_strength_{fomc, cpi, nfp}) — peaks on day after major event |
| 122 | `econ_events_calendar_macro_surprise_signed` | REGIME | **NEW**: importance-weighted sum of pulse surprises — directional macro impulse for policy state |

---

### Design Rationale: Bounded Proximity/Recency

**Problem**: Raw `days_to_next_*` and `days_since_last_*` with 9999 sentinels corrupt aggregation. If scaled naively, they behave as "max stress forever" or dominate normalization.

**Solution**: Bounded exponential transforms:

```
prox_next_e   = exp(-min(days_to_next_e, 252) / tau_next)     # tau_next = 5 days
recency_last_e = exp(-min(days_since_last_e, 252) / tau_last)  # tau_last = 10 days
```

- **prox_next**: Peaks at 1.0 when event is today, decays to ~0.05 after 15 days
- **recency_last**: Peaks at 1.0 right after event, decays to ~0.05 after 30 days
- Both bounded in [0, 1] — safe for role-aware averaging and stress aggregation

### Design Rationale: Composite Macro Features

**Purpose**: Reduce 100+ columns to 3 actionable signals for policy state and overlays.

```
macro_upcoming_major = max(prox_next_fomc, prox_next_cpi, prox_next_nfp)
macro_shock_major    = max(pulse_strength_fomc, pulse_strength_cpi, pulse_strength_nfp)
macro_surprise_signed = Σ(weight_e × pulse_surprise_e)
```

**Usage in Policy Controller**:
- `macro_upcoming_major > 0.5` → select conservative knobs (lower target_vol, higher threshold)
- `macro_shock_major > 0` → reduce turnover, widen stops
- `macro_surprise_signed` → directional bias for regime detection

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `prox_next_fomc` > 0.8 | FOMC within 2-3 days | Reduce risk, await policy direction |
| `macro_upcoming_major` > 0.5 | Major event imminent | Conservative positioning |
| `macro_shock_major` > 2σ | Large macro surprise just hit | Volatility spike, reduce turnover |
| `pre_window_3d_fomc` = 1 | Pre-FOMC window | Reduce risk, await direction |
| `post_window_3d_fomc` = 1 | Post-FOMC window | Volatility spike, position rebalancing |
| `pulse_strength_cpi` high | Hot inflation surprise | Fed hawkish pressure, growth stocks hit |
| `recency_last_nfp` > 0.5 | Within ~7 days of NFP | Post-employment data turbulence |
| `pulse_occurrence_nfp` = 1 | Day after NFP | Directional move continuation |

---

### Design Notes

**Leakage Policy:**
- Schedule features (prox_next, windows): Safe unshifted
- Realized surprises: Applied on **next** NYSE session after release
- `event_occurrence`: Unshifted (1 on release date)
- `pulse_occurrence`: Shifted (1 on next session)
- `pulse_surprise`, `pulse_strength`: Shifted (safe for same-session use)

**Importance Weights:**
- FOMC, CPI, NFP: High = 3.0
- GDP, PCE, Unemployment: Medium-High = 2.0-2.5
- Retail Sales, ISM, Core CPI: Medium = 1.5-2.0

**Feature Engineering:**
- Z-scores: 5-year rolling window (1825 days) for regime stability
- Bounded proximity: τ=5 days (events matter mainly within ~1 week)
- Bounded recency: τ=10 days (post-event turbulence decays over ~2 weeks)

---

## exchange_calendar Family — Full Feature Reference

**Source:** EODHD Exchange Details API + exchange_calendars (deterministic fallback)  
**Columns:** 12 (8 features + 4 governance) — Updated Jan 2026  
**Philosophy:** Calendar-aware structural features for execution context. Not alpha — purely microstructure/liquidity awareness.

**Exchange Coverage:** Currently US/NYSE-session aligned. Multi-exchange support ready via hooks.

**Critical Fixes (Jan 2026):**
- Replaced 9999 sentinels with bounded holiday_prox/holiday_recency signals (safe for aggregation)
- Added `calendar_liquidity_stress` composite for single-column execution overlays
- ALL columns routed to portfolio only (no Mamba) — execution context, not predictive

### Stream Routing (Jan 2026)

| Stream | Columns | Purpose |
|--------|---------|---------|
| **Mamba** | None | Exchange calendar is execution context, not alpha |
| **Portfolio** | All columns | Risk/regime overlays for execution timing |

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `exchange_calendar_has_data` | HYGIENE | Boolean: calendar data available |
| 2 | `exchange_calendar_activity` | HYGIENE | Activity score (0-1) |
| 3 | `exchange_calendar_days_since_update` | HYGIENE | Days since last data update |
| 4 | `exchange_calendar_confidence` | HYGIENE | Data confidence score (0-1) |

### Calendar Structure Features — DEPRECATED (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `exchange_calendar_days_to_holiday` | REGIME | **DEPRECATED**: Calendar days to next holiday (9999 if none) — use `holiday_prox` |
| 6 | `exchange_calendar_days_since_holiday` | REGIME | **DEPRECATED**: Calendar days since last holiday (9999 if none) — use `holiday_recency` |

### Calendar Structure Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `exchange_calendar_is_holiday_adjacent` | REGIME | Binary: 1 if ±1 calendar day from holiday |
| 8 | `exchange_calendar_is_half_day` | REGIME | Binary: 1 if early close session |
| 9 | `exchange_calendar_is_month_end` | REGIME | Binary: 1 if last trading day of month (rebalance flows) |

### Bounded Proximity/Recency — NEW (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `exchange_calendar_holiday_prox` | REGIME | **NEW**: exp(-min(days_to_holiday, 60) / 3). Peaks when holiday is imminent. |
| 11 | `exchange_calendar_holiday_recency` | REGIME | **NEW**: exp(-min(days_since_holiday, 60) / 3). Peaks right after holiday, decays quickly. |

### Composite Features — NEW (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `exchange_calendar_liquidity_stress` | RISK | **NEW**: clip(max(is_half_day, is_holiday_adjacent) + 0.5×holiday_prox + 0.3×holiday_recency, 0, 1). Single column for execution overlays. |

---

### Design Rationale: Bounded Proximity/Recency

**Problem**: Raw `days_to_holiday` and `days_since_holiday` with 9999 sentinels corrupt aggregation. Calendar effects are very local (2-3 days).

**Solution**: Bounded exponential transforms with short decay:

```
holiday_prox   = exp(-min(days_to_holiday, 60) / 3)    # tau = 3 days
holiday_recency = exp(-min(days_since_holiday, 60) / 3)  # tau = 3 days
```

- **holiday_prox**: Peaks at 1.0 when holiday is today, decays to ~0.05 after 9 days
- **holiday_recency**: Peaks at 1.0 right after holiday, decays to ~0.05 after 9 days
- Short τ=3 reflects that calendar effects are very localized

### Design Rationale: liquidity_stress Composite

**Purpose**: Single column for execution layer to adjust slippage/position sizing.

```
liquidity_stress = clip(
    max(is_half_day, is_holiday_adjacent) + 0.5 * holiday_prox + 0.3 * holiday_recency,
    0, 1
)
```

**Interpretation**:
- 0.0 = Normal liquidity conditions
- 0.3-0.5 = Approaching holiday, some caution warranted
- 0.7-0.8 = Holiday-adjacent or half-day, reduce aggression
- 1.0 = Maximum stress (half-day + holiday-adjacent + approaching holiday)

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `liquidity_stress` > 0.7 | High stress | Reduce position size, widen stops |
| `liquidity_stress` > 0.5 | Moderate stress | Cautious execution, expect spreads |
| `is_holiday_adjacent` = 1 | Pre/post holiday | Reduced liquidity, thin trading |
| `holiday_prox` > 0.5 | Within ~2 days of holiday | Early close risk, position squaring |
| `holiday_recency` > 0.5 | Within ~2 days after holiday | Catch-up flows, gap fills |
| `is_half_day` = 1 | Early close (1pm ET) | Compressed trading window, vol risk |
| `is_month_end` = 1 | Month-end rebalance | Large fund flows, trend reversal risk |

---

### Design Notes

**Data Sources:**
1. **EODHD Exchange Details API:**
   - Holidays list per exchange
   - Early close days with actual close times
   - Preferred source (vendor-maintained)

2. **exchange_calendars (fallback):**
   - Deterministic holiday calendars
   - Regular holidays + adhoc holidays
   - Schedule-based early close detection

**Union Strategy:**
- Both sources are unioned for maximum coverage
- EODHD takes precedence when available
- Deterministic fallback ensures robustness

**Leakage Policy:**
- None (calendar metadata is public/deterministic)
- All features safe to use unshifted

**Usage:**
- **liquidity_stress**: Primary signal for execution layer
- **Pre-holiday:** Reduce position sizes, widen stops
- **Post-holiday:** Gap risk management
- **Half-day sessions:** Execution timing, vol adjustments
- **Month-end:** Rebalance flow awareness

---

## fin_g1 Family — Full Feature Reference

**Source:** EODHD Fundamentals API (Balance Sheet, quarterly)  
**Columns:** 13 (9 features + 4 stress + 3 governance) — Updated Jan 2026  
**Philosophy:** STRICT liquidity ratios only. Measures short-term debt-paying ability.

**Critical Fixes (Jan 2026):**
- Added STRESS features (liquidity_stress, quick_stress, cash_stress, liquidity_trend_stress)
- STRESS features are HIGHER = WORSE (safe for RoleAwareContext risk aggregation)
- Raw ratios kept for optional Mamba use (horizon >= 21d only)
- Per-family staleness threshold: 130 days (quarterly cadence)

### Stream Routing (Jan 2026)

| Stream | Columns | Purpose |
|--------|---------|---------|
| **Portfolio** (default) | All governance, stress features, trend | Risk overlays + staleness gating |
| **Mamba** (optional) | current_ratio, quick_ratio, cash_ratio | Enable via `FIN_G1_MAMBA_OPTIONAL=all` |

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `fin_g1_has_data` | HYGIENE | Boolean: fundamental data available |
| 2 | `fin_g1_activity` | HYGIENE | Activity score (0-1) |
| 3 | `fin_g1_days_since_update` | HYGIENE | Days since last data update (threshold: 130 days) |

### Core Liquidity Ratios (3) — OPTIONAL FOR MAMBA

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `fin_g1_current_ratio` | PREDICTIVE | Current Assets / Current Liabilities (HIGHER = safer) |
| 5 | `fin_g1_quick_ratio` | PREDICTIVE | (Current Assets - Inventory) / Current Liabilities |
| 6 | `fin_g1_cash_ratio` | PREDICTIVE | Cash / Current Liabilities |

### Stress Features — NEW (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `fin_g1_liquidity_stress` | RISK | **NEW**: 1/current_ratio, clipped [0, 2]. HIGHER = worse. |
| 8 | `fin_g1_quick_stress` | RISK | **NEW**: 1/quick_ratio, clipped [0, 2]. HIGHER = worse. |
| 9 | `fin_g1_cash_stress` | RISK | **NEW**: 1/cash_ratio, clipped [0, 4]. HIGHER = worse. |
| 10 | `fin_g1_liquidity_trend_stress` | RISK | **NEW**: clip(-trend_3y, 0, 1). Deterioration = stress. |

### Trend & Normalization (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `fin_g1_liquidity_trend_3y` | REGIME | 3-year linear trend slope of current_ratio |
| 12 | `fin_g1_liquidity_zscore_5y` | RISK | Z-score of current_ratio vs 5-year mean |

---

### Design Rationale: Stress Features

**Problem**: Raw ratios (current_ratio, quick_ratio) are HIGHER = SAFER. RoleAwareContext uses `1/(1+risk_agg)` for risk_scale, so feeding raw ratios would INVERT the meaning (safest companies get downweighted).

**Solution**: Inverted stress features where HIGHER = WORSE:

```python
liquidity_stress = clip(1.0 / current_ratio, 0, 2)
quick_stress     = clip(1.0 / quick_ratio, 0, 2)
cash_stress      = clip(1.0 / cash_ratio, 0, 4)
liquidity_trend_stress = clip(-liquidity_trend_3y, 0, 1)
```

**Usage in Portfolio**:
- `liquidity_stress > 1.0` → current_ratio < 1 (danger zone)
- `liquidity_stress < 0.5` → current_ratio > 2 (safe)
- These flow into RoleAwareContext.risk_scale correctly

### Interpretation Guide

| Condition | Financial State | Trading Implication |
|-----------|-----------------|---------------------|
| `liquidity_stress` > 1.0 | Weak liquidity (CR < 1) | High distress risk, reduce exposure |
| `liquidity_stress` < 0.5 | Strong liquidity (CR > 2) | Defensible, stable |
| `liquidity_trend_stress` > 0.5 | Deteriorating liquidity | Credit concerns, reduce exposure |
| `liquidity_zscore_5y` < -2 | Unusually low liquidity | Distress signal |

---

## fin_g2 Family — Full Feature Reference

**Source:** EODHD Fundamentals API (Balance Sheet + Income Statement, quarterly)  
**Columns:** 18 (12 features + 3 stress + 3 governance) — Updated Jan 2026  
**Philosophy:** STRICT leverage/capital structure only. Measures debt burden and solvency.

**Critical Fixes (Jan 2026):**
- Added interest_coverage_stress (INVERTED: higher = worse)
- Added robust transforms (debt_to_equity_robust, net_debt_to_ebitda_robust)
- Per-family staleness threshold: 130 days (quarterly cadence)

### Stream Routing (Jan 2026)

| Stream | Columns | Purpose |
|--------|---------|---------|
| **Portfolio** (default) | All governance, leverage ratios, stress, robust | Risk overlays |
| **Mamba** (optional) | interest_coverage | Enable via `FIN_G2_MAMBA_OPTIONAL=interest_coverage` |

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `fin_g2_has_data` | HYGIENE | Boolean: fundamental data available |
| 2 | `fin_g2_activity` | HYGIENE | Activity score (0-1) |
| 3 | `fin_g2_days_since_update` | HYGIENE | Days since last data update (threshold: 130 days) |

### Debt Levels (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `fin_g2_total_debt` | REGIME | Short-term debt + Long-term debt |
| 5 | `fin_g2_long_term_debt` | REGIME | Long-term debt obligations |

### Core Leverage Ratios (3) — HIGHER = WORSE

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `fin_g2_debt_to_equity` | RISK | Total Debt / Total Equity |
| 7 | `fin_g2_debt_to_assets` | RISK | Total Debt / Total Assets |
| 8 | `fin_g2_equity_multiplier` | RISK | Total Assets / Total Equity (leverage amplifier) |

### Coverage Ratios (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `fin_g2_net_debt_to_ebitda` | RISK | (Total Debt - Cash) / EBITDA |
| 10 | `fin_g2_net_debt_to_fcf` | RISK | (Total Debt - Cash) / Free Cash Flow |
| 11 | `fin_g2_interest_burden` | RISK | Interest Expense / Operating Income |
| 12 | `fin_g2_interest_coverage` | PREDICTIVE | Operating Income / Interest Expense (HIGHER = safer) |

### Stress & Robust Features — NEW (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 13 | `fin_g2_interest_coverage_stress` | RISK | **NEW**: 1/(1+max(coverage, 0)). HIGHER = worse. |
| 14 | `fin_g2_debt_to_equity_robust` | RISK | **NEW**: sign(x)*log1p(|x|). Safe for exploding ratios. |
| 15 | `fin_g2_net_debt_to_ebitda_robust` | RISK | **NEW**: sign(x)*log1p(|x|). Safe for exploding ratios. |

### Trend & Normalization (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 16 | `fin_g2_leverage_trend_3y` | REGIME | 3-year linear trend slope of debt_to_equity |
| 17 | `fin_g2_leverage_zscore_5y` | RISK | Z-score of debt_to_equity vs 5-year mean |

---

### Design Rationale: Stress & Robust Features

**Problem 1**: `interest_coverage` is HIGHER = SAFER. If fed raw into risk aggregation, safest companies (high coverage) get penalized.

**Solution**: Inverted stress feature:
```python
interest_coverage_stress = clip(1.0 / (1.0 + max(coverage, 0)), 0, 1)
```
- Coverage = 10 → stress = 0.09 (safe)
- Coverage = 1 → stress = 0.5 (moderate)
- Coverage = 0 or negative → stress = 1.0 (distressed)

**Problem 2**: Debt ratios can explode when equity is small/negative.

**Solution**: Robust signed_log1p transform:
```python
debt_to_equity_robust = sign(x) * log1p(|x|)
```
- Preserves sign (negative equity → negative ratio)
- Compresses extreme values
- Safe for normalization and aggregation

---

### Interpretation Guide

| Condition | Financial State | Trading Implication |
|-----------|-----------------|---------------------|
| `debt_to_equity` > 2.0 | High leverage | Amplified beta, credit risk |
| `debt_to_equity` < 0.5 | Conservative capital structure | Lower risk, less financial flexibility |
| `debt_to_assets` > 0.6 | Debt-heavy | Asset liquidation risk |
| `equity_multiplier` > 3.0 | High financial leverage | ROE amplification, volatile |
| `net_debt_to_ebitda` > 4.0 | Elevated debt burden | Refinancing concerns |
| `net_debt_to_ebitda` < 2.0 | Manageable debt | Investment-grade profile |
| `interest_coverage` > 5.0 | Strong coverage | Comfortable debt service |
| `interest_coverage` < 2.0 | Weak coverage | Distress risk if earnings slip |
| `interest_burden` > 0.3 | High interest cost | Profitability drag |
| `leverage_trend_3y` > 0.3 | Rising leverage | Aggressive financing, monitor |
| `leverage_trend_3y` < -0.3 | Deleveraging | Balance sheet repair, defensive |
| `leverage_zscore_5y` > 2 | Unusually high leverage | Elevated credit risk |

---

## fin_g3 Family — Full Feature Reference

**Source:** EODHD Fundamentals API (Income Statement + Balance Sheet, quarterly)  
**Columns:** 14 (9 features + 2 stress + 3 governance) — Updated Jan 2026  
**Philosophy:** STRICT efficiency/turnover ratios only. Measures operational effectiveness.

**Critical Fixes (Jan 2026):**
- Added CCC (Cash Conversion Cycle) as explicit column
- Added ccc_stress for risk aggregation (higher CCC = worse)
- Turnover ratios are INDUSTRY-STRUCTURAL (use sector-normalization for Mamba)
- Per-family staleness threshold: 130 days (quarterly cadence)

### Stream Routing (Jan 2026)

| Stream | Columns | Purpose |
|--------|---------|---------|
| **Portfolio** (default) | All governance, days metrics, ccc_stress, turnover_volatility_3y | Risk/regime overlays |
| **Mamba** (optional) | asset_turnover, inventory_turnover, receivables_turnover, payables_turnover | Enable via `FIN_G3_MAMBA_OPTIONAL=all` (requires sector normalization) |

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `fin_g3_has_data` | HYGIENE | Boolean: fundamental data available |
| 2 | `fin_g3_activity` | HYGIENE | Activity score (0-1) |
| 3 | `fin_g3_days_since_update` | HYGIENE | Days since last data update (threshold: 130 days) |

### Core Turnover Ratios (4) — INDUSTRY-STRUCTURAL

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `fin_g3_asset_turnover` | PREDICTIVE | Revenue / Total Assets (industry-dependent) |
| 5 | `fin_g3_inventory_turnover` | PREDICTIVE | COGS / Average Inventory (industry-dependent) |
| 6 | `fin_g3_receivables_turnover` | PREDICTIVE | Revenue / Accounts Receivable (collection speed) |
| 7 | `fin_g3_payables_turnover` | PREDICTIVE | COGS / Accounts Payable (payment timing) |

### Days Metrics & CCC (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `fin_g3_dsos` | REGIME | Days Sales Outstanding = 365 / receivables_turnover |
| 9 | `fin_g3_dios` | REGIME | Days Inventory Outstanding = 365 / inventory_turnover |
| 10 | `fin_g3_dpos` | REGIME | Days Payables Outstanding = 365 / payables_turnover |
| 11 | `fin_g3_ccc` | REGIME | **NEW**: Cash Conversion Cycle = DSO + DIO - DPO |

### Stress & Volatility Features — NEW (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `fin_g3_ccc_stress` | RISK | **NEW**: sigmoid((ccc-60)/30). HIGHER = worse. |
| 13 | `fin_g3_turnover_volatility_3y` | RISK | Rolling std of asset_turnover (operational instability) |

---

### Design Rationale: CCC Stress

**Problem**: Raw CCC varies widely by industry. For risk aggregation, we need a normalized stress signal.

**Solution**: Sigmoid transform centered at 60 days:
```python
ccc_stress = 1 / (1 + exp(-(ccc - 60) / 30))
```
- CCC = 30 → stress = 0.27 (healthy)
- CCC = 60 → stress = 0.50 (moderate)
- CCC = 90 → stress = 0.73 (stressed)
- CCC = 120 → stress = 0.88 (severe)

**Warning**: Turnover ratios are INDUSTRY-STRUCTURAL. If fed raw to Mamba, the model may learn "industry ID" rather than alpha. Prefer sector-normalized versions.

### Interpretation Guide

| Condition | Operational State | Trading Implication |
|-----------|-------------------|---------------------|
| `ccc_stress` > 0.7 | High CCC (>90 days) | Working capital tied up, cash flow drag |
| `ccc_stress` < 0.3 | Low CCC (<30 days) | Efficient cycle, strong cash conversion |
| `turnover_volatility_3y` > 0.5 | Unstable operations | Execution risk, business model stress |
| `dsos` > 90 | Slow collections | Credit quality concerns |
| `dios` > 90 | Excess inventory | Demand weakness, markdown risk |

---

### Cash Conversion Cycle (CCC)

**Formula:** CCC = DSO + DIO - DPO

| CCC Value | Interpretation |
|-----------|----------------|
| CCC < 0 | Negative cycle: Receive cash before paying suppliers (Amazon, Walmart) |
| CCC 0-30 | Excellent: Very efficient working capital |
| CCC 30-60 | Good: Healthy cycle |
| CCC 60-90 | Moderate: Industry-dependent |
| CCC > 90 | Elevated: Working capital stress, cash tied up |

---

## fin_g4 Family — Full Feature Reference

**Source:** EODHD Fundamentals API (Cash Flow + Income Statement + Balance Sheet, quarterly)  
**Columns:** 12 (9 features + 3 governance)  
**Philosophy:** Cash flow & earnings quality. Measures cash generation and quality of reported earnings.

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `fin_g4_has_data` | HYGIENE | Boolean: fundamental data available |
| 2 | `fin_g4_activity` | HYGIENE | Activity score (0-1) |
| 3 | `fin_g4_days_since_update` | HYGIENE | Days since last data update |

### Cash Flow Core (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `fin_g4_operating_cash_flow` | PREDICTIVE | Total Cash from Operating Activities |
| 5 | `fin_g4_free_cash_flow` | PREDICTIVE | Operating CF - CapEx (cash available to shareholders) |

### Cash Flow Quality Ratios (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `fin_g4_fcf_to_revenue` | PREDICTIVE | Free Cash Flow / Revenue (cash conversion efficiency) |
| 7 | `fin_g4_fcf_to_net_income` | PREDICTIVE | FCF / Net Income (earnings quality: >1 = high quality) |
| 8 | `fin_g4_fcf_margin` | PREDICTIVE | FCF / Revenue (profitability after growth investment) |
| 9 | `fin_g4_cfo_to_net_income` | PREDICTIVE | Operating CF / Net Income (cash backing of earnings) |

### Capital Intensity (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `fin_g4_capex_to_revenue` | REGIME | |CapEx| / Revenue (capital intensity of business model) |

### Earnings Quality (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `fin_g4_accruals_ratio` | RISK | (Net Income - Operating CF) / Total Assets. HIGH = lower quality |

---

### Interpretation Guide

| Condition | Financial State | Trading Implication |
|-----------|-----------------|---------------------|
| `operating_cash_flow` > 0 | Cash-generative | Sustainable business model |
| `free_cash_flow` > 0 | Cash after growth | Shareholder returns possible |
| `free_cash_flow` < 0 | Cash-consumptive | Funding gap, dilution risk |
| `fcf_to_net_income` > 1.2 | High earnings quality | Earnings backed by cash, trust NI |
| `fcf_to_net_income` < 0.8 | Low earnings quality | Accrual-heavy, investigate |
| `cfo_to_net_income` > 1.0 | Strong quality | Cash exceeds accounting earnings |
| `cfo_to_net_income` < 0.8 | Weak quality | Earnings not converting to cash |
| `accruals_ratio` > 0.1 | High accruals | Aggressive accounting, earnings suspect |
| `accruals_ratio` < 0 | Negative accruals | Conservative accounting, quality |
| `fcf_margin` > 20% | Excellent cash generation | High-quality business (SaaS, asset-light) |
| `fcf_margin` < 5% | Weak cash generation | Capital-intensive, margin pressure |
| `capex_to_revenue` > 15% | Capital-intensive | Heavy reinvestment needs (manufacturing) |
| `capex_to_revenue` < 5% | Asset-light | Low reinvestment needs (software, services) |

---

## fin_g5 Family — Full Feature Reference

**Source:** EODHD Fundamentals API (Income Statement + Cash Flow, quarterly)  
**Columns:** 11 (8 features + 3 governance)  
**Philosophy:** STRICT growth metrics only. Measures revenue, earnings, and cash flow growth.

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `fin_g5_has_data` | HYGIENE | Boolean: fundamental data available |
| 2 | `fin_g5_activity` | HYGIENE | Activity score (0-1) |
| 3 | `fin_g5_days_since_update` | HYGIENE | Days since last data update |

### Year-over-Year Growth (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `fin_g5_revenue_growth_yoy` | PREDICTIVE | (Revenue_t - Revenue_t-4) / Revenue_t-4 (quarterly YoY) |
| 5 | `fin_g5_ebitda_growth_yoy` | PREDICTIVE | EBITDA YoY growth (operating leverage) |
| 6 | `fin_g5_cf_growth_yoy` | PREDICTIVE | Operating Cash Flow YoY growth |

### Multi-Year Growth (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `fin_g5_revenue_cagr_3y` | PREDICTIVE | 3-year revenue CAGR (compound annual growth rate) |
| 8 | `fin_g5_eps_cagr_3y` | PREDICTIVE | 3-year EPS CAGR |

### Margin Dynamics (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `fin_g5_margin_expansion` | PREDICTIVE | Change in net margin YoY (operating leverage indicator) |

### Growth Quality (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `fin_g5_growth_volatility_3y` | RISK | Rolling std of revenue_growth_yoy (growth stability) |

---

### Interpretation Guide

| Condition | Growth State | Trading Implication |
|-----------|--------------|---------------------|
| `revenue_growth_yoy` > 20% | Hyper-growth | High-growth stock, premium valuation |
| `revenue_growth_yoy` 10-20% | Strong growth | Quality growth name |
| `revenue_growth_yoy` 0-10% | Moderate growth | Mature growth, value transition |
| `revenue_growth_yoy` < 0 | Contracting | Secular decline or cyclical trough |
| `ebitda_growth_yoy` > `revenue_growth_yoy` | Margin expansion | Operating leverage kicking in |
| `ebitda_growth_yoy` < `revenue_growth_yoy` | Margin compression | Investment phase or pricing pressure |
| `cf_growth_yoy` > `revenue_growth_yoy` | Improving cash conversion | Quality growth |
| `cf_growth_yoy` < 0, revenue > 0 | Cash burn during growth | Funding risk |
| `revenue_cagr_3y` > 25% | Sustained hyper-growth | Rule of 40 candidate |
| `eps_cagr_3y` > `revenue_cagr_3y` | Margin expansion | Profitability improving |
| `margin_expansion` > 5% | Strong operating leverage | Earnings inflection |
| `margin_expansion` < -5% | Margin pressure | Competitive headwinds |
| `growth_volatility_3y` > 20% | Erratic growth | Cyclical or execution issues |
| `growth_volatility_3y` < 10% | Stable growth | Predictable, high-quality |

---

## fin_g6 Family — Full Feature Reference

**Source:** EODHD Fundamentals API (quarterly) + Daily Price Data  
**Columns:** 12 (9 features + 3 governance)  
**Philosophy:** Time-varying valuation multiples using daily prices + forward-filled quarterly fundamentals.

**Design:** Ratios computed DAILY (price changes daily, fundamentals persist until next quarterly report).

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `fin_g6_has_data` | HYGIENE | Boolean: fundamental data available |
| 2 | `fin_g6_activity` | HYGIENE | Activity score (0-1) |
| 3 | `fin_g6_days_since_update` | HYGIENE | Days since last data update |

### Core Valuation Multiples (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `fin_g6_pe_ratio` | RISK | Price / TTM EPS (earnings multiple) |
| 5 | `fin_g6_pb_ratio` | RISK | Price / Book Value per Share (asset multiple) |
| 6 | `fin_g6_ps_ratio` | RISK | Price / TTM Revenue per Share (sales multiple) |
| 7 | `fin_g6_ev_ebitda` | RISK | Enterprise Value / EBITDA (snapshot from API) |
| 8 | `fin_g6_dividend_yield` | RISK | Dividend Yield % (snapshot from API) |

### Historical Normalization (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `fin_g6_pe_ratio_zscore_5y` | RISK | PE ratio z-score vs 5-year rolling mean |
| 10 | `fin_g6_pb_ratio_zscore_5y` | RISK | PB ratio z-score vs 5-year rolling mean |
| 11 | `fin_g6_ev_ebitda_zscore_5y` | RISK | EV/EBITDA z-score vs 5-year rolling mean |

---

### Interpretation Guide

| Condition | Valuation State | Trading Implication |
|-----------|-----------------|---------------------|
| `pe_ratio` < 15 | Value territory | Potentially cheap (if quality) |
| `pe_ratio` 15-25 | Fair value | Market multiple range |
| `pe_ratio` > 30 | Growth premium | Expensive, needs growth delivery |
| `pe_ratio` < 0 | Loss-making | Use PS or EV/EBITDA instead |
| `pb_ratio` < 1.0 | Trading below book | Liquidation value, distress or value trap |
| `pb_ratio` 1-3 | Normal range | Asset-backed valuation |
| `pb_ratio` > 5 | Asset-light premium | Tech, SaaS, intangible-heavy |
| `ps_ratio` < 2 | Cheap on sales | Value or cyclical trough |
| `ps_ratio` 5-10 | Growth multiple | SaaS, high-growth software |
| `ps_ratio` > 15 | Extreme premium | Hyper-growth or bubble |
| `ev_ebitda` < 8 | Attractive | Potential value or LBO candidate |
| `ev_ebitda` 10-15 | Market range | Fair value |
| `ev_ebitda` > 20 | Expensive | Needs margin expansion or growth |
| `dividend_yield` > 4% | High yield | Income stock or distress |
| `dividend_yield` 2-4% | Moderate yield | Balanced return profile |
| `dividend_yield` < 1% | Growth focus | Reinvestment over distribution |
| `pe_zscore_5y` > 2 | Expensive vs history | Mean reversion risk |
| `pe_zscore_5y` < -2 | Cheap vs history | Value opportunity or structural decline |
| `pb_zscore_5y` > 2 | Premium to history | Re-rating or bubble |
| `ev_ebitda_zscore_5y` < -1 | Below normal | Margin pressure or cyclical trough |

---

## fin_g7 Family — Full Feature Reference

**Source:** EODHD Fundamentals API (quarterly Income Statement, Cash Flow, Balance Sheet)  
**Columns:** 11 (8 features + 3 governance)  
**Philosophy:** Dividend policy stability and shareholder yield metrics with 3-year rolling consistency measures.

**Design:** Enhances basic dividend metrics with policy stability (payout ratio volatility), buyback consistency, share dilution tracking, and historical yield normalization.

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `fin_g7_has_data` | HYGIENE | Boolean: fundamental data available |
| 2 | `fin_g7_activity` | HYGIENE | Activity score (0-1) |
| 3 | `fin_g7_days_since_update` | HYGIENE | Days since last data update |

### Core Dividend Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `fin_g7_payout_ratio` | RISK | |Dividends Paid| / Net Income (dividend sustainability) |
| 5 | `fin_g7_dividend_yield_proxy` | REGIME | Payout ratio as dividend yield proxy |

### Policy Stability (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `fin_g7_dividend_policy_stability` | RISK | Std dev of payout ratio over 12 quarters (LOW = stable policy) |
| 7 | `fin_g7_buyback_consistency` | REGIME | Fraction of last 12 quarters with net buybacks > 0 |
| 8 | `fin_g7_share_dilution_3y` | RISK | (Shares_t - Shares_t-12) / Shares_t-12 (positive = dilution) |

### Historical Context (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `fin_g7_yield_zscore_5y` | RISK | Dividend yield proxy z-score vs 20-quarter history |
| 10 | `fin_g7_shares_outstanding` | HYGIENE | Common stock shares outstanding (raw value) |

---

### Interpretation Guide

| Condition | Shareholder Return Policy | Trading Implication |
|-----------|---------------------------|---------------------|
| `payout_ratio` 30-60% | Balanced policy | Sustainable dividends + growth reinvestment |
| `payout_ratio` > 80% | High payout | Mature, limited growth, income stock |
| `payout_ratio` > 100% | Unsustainable | Dividend cut risk |
| `payout_ratio` < 20% | Low payout | Growth focus, reinvestment priority |
| `dividend_policy_stability` < 0.05 | Stable policy | Predictable dividend (utilities, REITs) |
| `dividend_policy_stability` > 0.2 | Volatile policy | Erratic dividends, cyclical or distress |
| `buyback_consistency` > 0.7 | Active buybacks | Capital allocation via repurchases |
| `buyback_consistency` < 0.3 | Infrequent buybacks | Dividend-focused or investment phase |
| `share_dilution_3y` > 0.1 | Significant dilution | Equity raises, option grants, M&A |
| `share_dilution_3y` < -0.05 | Net buybacks | Shareholder-friendly capital allocation |
| `share_dilution_3y` near 0 | Neutral | No major equity activity |
| `yield_zscore_5y` > 2 | High yield vs history | Attractive income or distress signal |
| `yield_zscore_5y` < -2 | Low yield vs history | Dividend cut or reinvestment shift |

**Cross-Family Linkage:** Payout ratio used with ROE (fin_g0) to compute sustainable growth rate.

---

## finbert Family — Full Feature Reference

**Source:** GDELT news + FinBERT transformer model (yiyanghkust/finbert-tone)  
**Columns:** 6 (2 sentiment features + 4 governance)  
**Philosophy:** Daily sentiment aggregation from news headlines using financial domain-tuned BERT.

**Design:** 
- Loads news from unified GDELT cache (parquet format)
- Applies FinBERT sentiment model to daily headline aggregations
- Returns sentiment scores with model confidence
- Supports gap_thresh parameter for stale data filtering

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `finbert_has_data` | HYGIENE | Boolean: news data available |
| 2 | `finbert_activity` | HYGIENE | Activity score (0-1) |
| 3 | `finbert_days_since_update` | HYGIENE | Days since last data update |
| 4 | `finbert_confidence` | HYGIENE | Model confidence score (0-1) |

### Sentiment Features (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `finbert_score` | PREDICTIVE | Sentiment score from FinBERT model (-1 to +1, where -1=negative, 0=neutral, +1=positive) |
| 6 | `finbert_neutral` | REGIME | Probability of neutral sentiment (0-1) |

---

### Interpretation Guide

| Condition | Sentiment State | Trading Implication |
|-----------|-----------------|---------------------|
| `finbert_score` > 0.5 | Strong positive | Bullish news flow, potential momentum |
| `finbert_score` 0.1 to 0.5 | Moderately positive | Positive bias, watch for continuation |
| `finbert_score` -0.1 to 0.1 | Neutral | No clear directional signal |
| `finbert_score` -0.5 to -0.1 | Moderately negative | Negative bias, caution |
| `finbert_score` < -0.5 | Strong negative | Bearish news, potential downside |
| `finbert_neutral` > 0.7 | High neutral probability | News lacks strong directional content |
| `finbert_neutral` < 0.3 | Low neutral probability | Polarized sentiment (strong positive or negative) |
| `finbert_confidence` > 0.8 | High confidence | Model certain about sentiment classification |
| `finbert_confidence` < 0.5 | Low confidence | Ambiguous news, mixed signals |
| `finbert_score` spike + high confidence | Event-driven | Major news catalyst (earnings, M&A, regulatory) |
| `finbert_score` reversal after extreme | Sentiment exhaustion | Potential contrarian signal |

**Data Coverage:** Loads from cache/gdelt/{SYMBOL}_gdelt_unified.parquet  
**Model:** HuggingFace yiyanghkust/finbert-tone (financial domain BERT)  
**Fallback:** Legacy JSON cache or yfinance news (limited 7-day window)

**Usage Notes:**
- Sentiment scores are daily aggregations (multiple headlines per day)
- High neutral probability indicates non-directional news (corporate governance, routine filings)
- Combine with `finbert_confidence` to filter high-quality signals
- Use gap_thresh parameter to exclude stale news periods

---

## garch_iv Family — Full Feature Reference

**Source:** Daily price data (EODHD preferred, fallback to universal fetcher)  
**Columns:** 22 (19 features + 3 governance)  
**Philosophy:** GARCH(1,1) volatility forecasts with regime detection and model diagnostics.

**Design:** Fits rolling GARCH(1,1) models (252-day windows) to extract multi-horizon volatility forecasts, persistence metrics, and regime shift indicators. Combines legacy realized volatility proxies with sophisticated model-based features.

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `garch_iv_has_data` | HYGIENE | Boolean: price data available |
| 2 | `garch_iv_activity` | HYGIENE | Activity score (0-1) |
| 3 | `garch_iv_days_since_update` | HYGIENE | Days since last data update |

### Legacy Features (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `garch_iv_garch30_minus_garch180` | RISK | Realized vol spread: RV(30d) - RV(180d) (short-term regime shift) |
| 5 | `garch_iv_skew_proxy_downside_minus_upside` | RISK | Downside vol - upside vol (60d window, tail risk proxy) |

### GARCH Forecasts (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `garch_iv_garch_1d` | RISK | 1-day ahead volatility forecast from GARCH(1,1) model |
| 7 | `garch_iv_garch_5d` | RISK | 5-day ahead volatility forecast |
| 8 | `garch_iv_garch_20d` | RISK | 20-day ahead volatility forecast (medium-term) |

### Model Parameters (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `garch_iv_garch_persistence` | REGIME | alpha + beta from GARCH(1,1) (near 1.0 = highly persistent regime) |
| 10 | `garch_iv_garch_long_run_variance` | REGIME | Steady-state volatility: omega / (1 - alpha - beta) |
| 11 | `garch_iv_garch_log_likelihood` | HYGIENE | Model fit quality (sudden drop = regime shift or structural break) |

### Derived Regime Features (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `garch_iv_garch_ratio_1d_20d` | RISK | Short-term vs medium-term forecast ratio (spike = near-term shock) |
| 13 | `garch_iv_garch_spike_flag` | PREDICTIVE | Binary: 1 if garch_1d > 1.5 * garch_20d (regime shock detector) |
| 14 | `garch_iv_garch_zscore` | RISK | Z-score of garch_1d vs 60-day distribution |
| 15 | `garch_iv_garch_residual_vol` | RISK | Std dev of standardized residuals (20d) - model uncertainty |
| 16 | `garch_iv_garch_standardized_residual` | HYGIENE | Returns / conditional volatility (last value from model) |

### High-Alpha Features (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 17 | `garch_iv_garch_vol_of_vol` | PREDICTIVE | Std dev of garch_1d over 20 days (regime instability indicator) |
| 18 | `garch_iv_garch_short_long_ratio` | RISK | garch_1d / long_run_variance (deviation from steady state) |
| 19 | `garch_iv_garch_vol_norm_20d` | RISK | garch_20d / realized_vol_20d (model vs actual calibration) |
| 20 | `garch_iv_garch_vol_momentum` | PREDICTIVE | 10-day slope of garch_1d (volatility trend) |
| 21 | `garch_iv_garch_shock_indicator` | PREDICTIVE | Binary: 1 if |standardized_residual| > 2.0 (outlier event) |

---

### Interpretation Guide

| Condition | Volatility Regime | Trading Implication |
|-----------|-------------------|---------------------|
| `garch_1d` < 0.15 | Low volatility | Calm markets, potential vol expansion ahead |
| `garch_1d` 0.15-0.30 | Normal volatility | Typical market conditions |
| `garch_1d` > 0.40 | High volatility | Stress regime, risk-off |
| `garch_persistence` > 0.95 | Highly persistent | Regime slow to change (mean-reversion slower) |
| `garch_persistence` < 0.85 | Low persistence | Regime can shift quickly (fast mean-reversion) |
| `garch_spike_flag` = 1 | Regime shock | Near-term event (earnings, news, macro) |
| `garch_ratio_1d_20d` > 1.5 | Short-term spike | Temporary shock, expect reversion |
| `garch_ratio_1d_20d` < 0.8 | Unusually calm | Below medium-term trend |
| `garch_zscore` > 2 | Extreme vol | Vol expansion, defensive positioning |
| `garch_zscore` < -2 | Extreme calm | Vol compression, potential breakout |
| `garch_vol_of_vol` > 0.05 | Regime instability | Uncertainty about volatility itself |
| `garch_vol_of_vol` < 0.02 | Stable regime | Predictable volatility environment |
| `garch_vol_momentum` > 0 | Rising volatility | Build hedges, reduce exposure |
| `garch_vol_momentum` < 0 | Falling volatility | Vol crush, option selling opportunities |
| `garch_shock_indicator` = 1 | Outlier event | Price gap, news catalyst, liquidity shock |
| `garch_short_long_ratio` > 1.3 | Above steady state | Elevated vol, mean reversion bias |
| `garch_short_long_ratio` < 0.8 | Below steady state | Compressed vol, expansion risk |
| `garch_vol_norm_20d` > 1.2 | Model overestimating | GARCH forecasting higher than realized |
| `garch_vol_norm_20d` < 0.8 | Model underestimating | GARCH forecasting lower than realized |
| `garch_log_likelihood` sudden drop | Regime shift | Model fit deteriorating, structural break |

**Model:** GARCH(1,1) with 252-day rolling estimation windows  
**Requirements:** Minimum 252 days of price history for GARCH fitting  
**Fallback:** If `arch` library unavailable, only legacy features (RV spread, skew proxy) computed

---

## index_constituents Family — Full Feature Reference

**Source:** EODHD Fundamentals API (HistoricalTickerComponents for GSPC.INDX, DJI.INDX)  
**Columns:** 24 (20 features + 4 governance)  
**Philosophy:** Index membership events with look-ahead shifted +1 session (tradable timing).

**Design:** 
- Extracts add/remove events from HistoricalTickerComponents
- Membership state change applied on next session after event date (tradable timing)
- Tracks S&P 500 (GSPC) and Dow Jones (DJI) by default
- Rolling 5d/20d add/remove flows for market-wide index rebalancing pressure

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `index_constituents_has_data` | HYGIENE | Boolean: fundamentals data available for tracked indices |
| 2 | `index_constituents_activity` | HYGIENE | Activity score (0-1) |
| 3 | `index_constituents_days_since_update` | HYGIENE | Days since last data update |
| 4 | `index_constituents_confidence` | HYGIENE | Data confidence score |

### Data Availability (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `index_constituents_has_hist_gspc` | HYGIENE | Boolean: S&P 500 historical components available |
| 6 | `index_constituents_has_hist_dji` | HYGIENE | Boolean: Dow Jones historical components available |

### Current Membership (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `index_constituents_member_gspc` | REGIME | Binary: currently in S&P 500 (1 = yes, 0 = no) |
| 8 | `index_constituents_member_dji` | REGIME | Binary: currently in Dow Jones Industrial Average |
| 9 | `index_constituents_num_indices` | REGIME | Count of indices this symbol belongs to (0-2) |

### Addition Events (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `index_constituents_added_gspc` | PREDICTIVE | Binary pulse: 1 on session after S&P 500 addition (index inclusion effect) |
| 11 | `index_constituents_added_dji` | PREDICTIVE | Binary pulse: 1 on session after Dow Jones addition |
| 12 | `index_constituents_days_since_add_gspc` | REGIME | Days since last S&P 500 addition (9999 if never added) |
| 13 | `index_constituents_days_since_add_dji` | REGIME | Days since last Dow Jones addition |

### Removal Events (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 14 | `index_constituents_removed_gspc` | PREDICTIVE | Binary pulse: 1 on session after S&P 500 removal (index deletion effect) |
| 15 | `index_constituents_removed_dji` | PREDICTIVE | Binary pulse: 1 on session after Dow Jones removal |
| 16 | `index_constituents_days_since_remove_gspc` | REGIME | Days since last S&P 500 removal (9999 if never removed) |
| 17 | `index_constituents_days_since_remove_dji` | REGIME | Days since last Dow Jones removal |

### Index Weights (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 18 | `index_constituents_weight_gspc` | RISK | S&P 500 weight % (current snapshot, NOT forward-filled) |
| 19 | `index_constituents_weight_dji` | RISK | Dow Jones weight % (current snapshot) |

### Aggregate Flows (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 20 | `index_constituents_add_flow_5d` | PREDICTIVE | Rolling 5-day sum of addition events across tracked indices |
| 21 | `index_constituents_add_flow_20d` | PREDICTIVE | Rolling 20-day sum of addition events (rebalancing wave) |
| 22 | `index_constituents_remove_flow_5d` | PREDICTIVE | Rolling 5-day sum of removal events across tracked indices |
| 23 | `index_constituents_remove_flow_20d` | PREDICTIVE | Rolling 20-day sum of removal events |

---

### Interpretation Guide

| Condition | Index Event | Trading Implication |
|-----------|-------------|---------------------|
| `added_gspc` = 1 | S&P 500 inclusion | Index fund buying (day 1), potential momentum |
| `added_dji` = 1 | Dow Jones inclusion | Dow fund buying, liquidity increase |
| `removed_gspc` = 1 | S&P 500 deletion | Index fund selling pressure, vol spike |
| `removed_dji` = 1 | Dow Jones deletion | Dow fund selling, potential distress |
| `member_gspc` = 1 | In S&P 500 | Large-cap status, index fund ownership |
| `member_dji` = 1 | In Dow 30 | Blue-chip status, high profile |
| `num_indices` = 2 | In multiple indices | High institutional ownership, liquid |
| `num_indices` = 0 | Not in tracked indices | Mid/small-cap, lower index flows |
| `days_since_add_gspc` < 60 | Recent addition | Post-inclusion drift (3-6 months) |
| `days_since_add_gspc` > 1000 | Long-term member | Stable index ownership |
| `days_since_remove_gspc` < 60 | Recent removal | Post-deletion selling tail |
| `add_flow_5d` > 5 | Rebalancing wave | Market-wide index additions (quarterly rebal) |
| `remove_flow_5d` > 5 | Removal cluster | Market-wide deletions (sector rotation) |
| `weight_gspc` > 2.0 | Top S&P holding | Mega-cap, huge index flows (AAPL, MSFT) |
| `weight_gspc` < 0.05 | Small S&P weight | Minimal index impact |
| `weight_dji` > 5.0 | Top Dow holding | Price-weighted Dow: high $ price dominates |

**Look-Ahead Policy:** Membership changes applied on next session after event date (tradable timing)  
**Tracked Indices:** S&P 500 (GSPC.INDX), Dow Jones (DJI.INDX) by default  
**Weight Caveat:** Weights are current snapshots (last session), NOT historical time series

---

## insider_form4 Family — Full Feature Reference

**Source:** EODHD Insider Transactions API (SEC Form 4 filings)  
**Columns:** 32 (29 features + 3 governance)  
**Philosophy:** No-lookahead insider transaction aggregates with +1 session shift from filing date.

**Design:** 
- Aggregates Form 4 filings by filing_date (reportDate from SEC)
- Signals applied on first tradable session AFTER filing date (+1 session shift)
- Separates buy (P) vs sell (S) transactions
- Tracks executive (CEO) transactions separately
- Computes rolling intensities (5d/20d/63d windows)
- Market cap normalization for transaction magnitude (best-effort)

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `insider_form4_has_data` | HYGIENE | Boolean: EODHD insider API access available |
| 2 | `insider_form4_activity` | HYGIENE | Activity score (0-1) |
| 3 | `insider_form4_days_since_update` | HYGIENE | Days since last data update |

### Daily Transaction Aggregates (15)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `insider_form4_net_shares_1d` | PREDICTIVE | Buy shares - Sell shares (on effective session) |
| 5 | `insider_form4_net_value_1d` | PREDICTIVE | Buy value - Sell value ($) |
| 6 | `insider_form4_buy_shares_1d` | PREDICTIVE | Total shares purchased by insiders |
| 7 | `insider_form4_sell_shares_1d` | PREDICTIVE | Total shares sold by insiders |
| 8 | `insider_form4_buy_value_1d` | PREDICTIVE | Total $ value of insider buys |
| 9 | `insider_form4_sell_value_1d` | RISK | Total $ value of insider sells |
| 10 | `insider_form4_num_trades_1d` | REGIME | Total number of Form 4 filings |
| 11 | `insider_form4_num_buy_trades_1d` | PREDICTIVE | Number of buy transactions |
| 12 | `insider_form4_num_sell_trades_1d` | REGIME | Number of sell transactions |
| 13 | `insider_form4_num_unique_insiders_1d` | REGIME | Count of distinct insiders filing |
| 14 | `insider_form4_num_unique_buy_insiders_1d` | PREDICTIVE | Count of distinct buyers |
| 15 | `insider_form4_num_unique_sell_insiders_1d` | REGIME | Count of distinct sellers |
| 16 | `insider_form4_exec_buy_value_1d` | PREDICTIVE | Buy value by CEO/Chief Executive ($) |
| 17 | `insider_form4_exec_sell_value_1d` | RISK | Sell value by CEO ($) |
| 18 | `insider_form4_exec_net_value_1d` | PREDICTIVE | CEO net buy value ($) |

### Derived Signals (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 19 | `insider_form4_cluster_buy_1d` | PREDICTIVE | Binary: 1 if multiple insiders (≥3) buy on same day (coordinated signal) |
| 20 | `insider_form4_days_since_buy` | REGIME | Days since last insider buy (9999 if never) |
| 21 | `insider_form4_days_since_sell` | REGIME | Days since last insider sell (9999 if never) |

### Rolling Intensities (12)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 22 | `insider_form4_net_value_5d` | PREDICTIVE | 5-day rolling sum of net_value_1d |
| 23 | `insider_form4_buy_value_5d` | PREDICTIVE | 5-day rolling sum of buy_value_1d |
| 24 | `insider_form4_sell_value_5d` | RISK | 5-day rolling sum of sell_value_1d |
| 25 | `insider_form4_num_trades_5d` | REGIME | 5-day rolling sum of num_trades_1d |
| 26 | `insider_form4_net_value_20d` | PREDICTIVE | 20-day rolling sum of net_value_1d |
| 27 | `insider_form4_buy_value_20d` | PREDICTIVE | 20-day rolling sum of buy_value_1d |
| 28 | `insider_form4_sell_value_20d` | RISK | 20-day rolling sum of sell_value_1d |
| 29 | `insider_form4_num_trades_20d` | REGIME | 20-day rolling sum of num_trades_1d |
| 30 | `insider_form4_net_value_63d` | PREDICTIVE | 63-day rolling sum of net_value_1d |
| 31 | `insider_form4_buy_value_63d` | PREDICTIVE | 63-day rolling sum of buy_value_1d |
| 32 | `insider_form4_sell_value_63d` | RISK | 63-day rolling sum of sell_value_1d |
| 33 | `insider_form4_num_trades_63d` | REGIME | 63-day rolling sum of num_trades_1d |

### Normalized Metrics (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 34 | `insider_form4_net_value_20d_z252` | PREDICTIVE | Z-score of net_value_20d vs 252-day distribution |
| 35 | `insider_form4_has_mcap` | HYGIENE | Boolean: market cap data available for normalization |
| 36 | `insider_form4_net_value_pct_mcap_1d` | PREDICTIVE | net_value_1d / market_cap (transaction size vs company size) |

---

### Interpretation Guide

| Condition | Insider Signal | Trading Implication |
|-----------|----------------|---------------------|
| `net_value_1d` > $1M | Large insider buy | Bullish insider conviction (information advantage) |
| `net_value_1d` < -$1M | Large insider sell | Bearish signal (but could be portfolio rebalancing) |
| `cluster_buy_1d` = 1 | Coordinated buying | Multiple insiders buying = strong bullish signal |
| `exec_buy_value_1d` > 0 | CEO buying | Highest conviction insider signal (CEO knows best) |
| `exec_sell_value_1d` > 0 | CEO selling | Caution, but often benign (compensation, taxes) |
| `num_unique_buy_insiders_1d` ≥ 3 | Broad insider buying | Consensus bullish view among insiders |
| `num_unique_sell_insiders_1d` ≥ 3 | Broad insider selling | Consensus bearish view |
| `net_value_20d` > $5M | Sustained buying | 20-day accumulation by insiders |
| `net_value_20d` < -$5M | Sustained selling | 20-day distribution by insiders |
| `net_value_20d_z252` > 2 | Extreme buying | Insider buying at 252-day highs |
| `net_value_20d_z252` < -2 | Extreme selling | Insider selling at 252-day highs |
| `net_value_pct_mcap_1d` > 0.1% | Material transaction | Transaction > 0.1% of market cap (significant) |
| `days_since_buy` < 5 | Recent buying | Fresh insider buy (within last week) |
| `days_since_sell` > 180 | No recent selling | Insiders holding (bullish) |
| `buy_value_5d` > sell_value_5d` | Near-term bullish | More buying than selling this week |

**Look-Ahead Policy:** Filing date shifted +1 session (tradable timing)  
**Sparsity:** Many days will be zero (no filings)  
**Transaction Codes:** P = Purchase, S = Sale (standard SEC Form 4 codes)  
**Executive Filter:** CEO / Chief Executive Officer title variants only

---

## marketcap_history Family — Full Feature Reference

**Source:** EODHD Historical Market Capitalization API (weekly points, forward-filled to daily)  
**Columns:** 9 (8 features + 1 governance)  
**Philosophy:** Size dynamics with +1 session leakage-safe shift and fundamentals-based fallback.

**Design:** 
- Primary: EODHD weekly market cap points (often starts ~2021)
- Fallback: Close price × shares_outstanding from quarterly fundamentals (full history)
- All points shifted +1 session for leakage safety
- Forward-filled to daily (ffill_limit=10 to avoid stale data)
- Log returns for statistical stability

### Governance Columns (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `marketcap_history_has_data` | HYGIENE | Boolean: market cap data available (>0) |

### Core Market Cap Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 2 | `marketcap_history_mcap` | REGIME | Market capitalization ($) - raw value |
| 3 | `marketcap_history_log_mcap` | REGIME | Log(market cap) - for statistical stability |

### Size Change Metrics (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `marketcap_history_mcap_chg_20d` | RISK | Log(mcap_t) - log(mcap_t-20) - 20-day size change |
| 5 | `marketcap_history_mcap_chg_63d` | RISK | Log(mcap_t) - log(mcap_t-63) - 63-day size change (quarter) |
| 6 | `marketcap_history_mcap_chg_252d` | RISK | Log(mcap_t) - log(mcap_t-252) - 252-day size change (year) |

### Volatility Metrics (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `marketcap_history_mcap_vol_63d` | RISK | Rolling 63-day std of daily log market cap changes |

### Liquidity Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `marketcap_history_float_turnover` | RISK | Volume / shares_float (daily turnover ratio) |
| 9 | `marketcap_history_turnover_z_252d` | RISK | Z-score of float_turnover vs 252-day distribution |

---

### Interpretation Guide

| Condition | Size Regime | Trading Implication |
|-----------|-------------|---------------------|
| `log_mcap` > 11 (e−11 ≈ $60B) | Mega-cap | Highly liquid, index-heavy, lower alpha |
| `log_mcap` 9-11 ($8B-$60B) | Large-cap | S&P 500 territory, institutional ownership |
| `log_mcap` 7-9 ($1B-$8B) | Mid-cap | Growth potential, moderate liquidity |
| `log_mcap` < 7 (<$1B) | Small-cap | High alpha potential, illiquidity risk |
| `mcap_chg_20d` > 0.2 | Rapid expansion | +22% in 20 days, momentum or bubble |
| `mcap_chg_20d` < -0.2 | Rapid contraction | -18% in 20 days, distress or correction |
| `mcap_chg_63d` > 0.5 | Major expansion | +65% in quarter, hyper-growth or acquisition |
| `mcap_chg_252d` > 1.0 | Doubled in year | 2.7x annual return, extreme momentum |
| `mcap_chg_252d` < -0.5 | Halved in year | -39% drawdown, distress or secular decline |
| `mcap_vol_63d` > 0.05 | High volatility | Size unstable, price-driven or dilution events |
| `mcap_vol_63d` < 0.02 | Low volatility | Stable size, predictable growth |
| `float_turnover` > 0.05 | High turnover | 5% of float trades daily, very liquid |
| `float_turnover` < 0.005 | Low turnover | <0.5% daily, illiquid or tightly held |
| `turnover_z_252d` > 2 | Turnover spike | Unusual volume (news, event, rebalancing) |
| `turnover_z_252d` < -2 | Turnover drought | Unusually low volume (holiday, neglect) |

**Fallback Strategy:** If EODHD historical-market-cap unavailable or starts late, uses Close × shares_outstanding from quarterly fundamentals  
**Leakage Policy:** Market cap points shifted +1 session after original date  
**Forward-Fill:** Weekly points forward-filled to daily (max 10 sessions to avoid stale data)  
**Turnover Caveat:** Requires shares_float from EODHD; fallback uses shares_outstanding as proxy

---

## microstructure Family — Full Feature Reference

**Source:** Daily OHLCV from EODHD (primary) or UniversalDataFetcher (fallback)  
**Columns:** 34 (31 features + 3 governance)  
**Philosophy:** Raw mathematical microstructure features derived from OHLCV bars—NO ML, NO HF complexity.

**Design:** 
- **Scale-free**: All features normalized by price, volume, or ATR
- **Continuous**: No binary flags (gradient-friendly for deep learning)
- **Mathematically pure**: Direct OHLCV transformations
- **Intraday fallback**: If intraday data available (≤30 days), uses minute bars; otherwise daily proxy
- **Quote enhancement**: Last-row quote snapshot from EODHD (bid/ask/size) for near-real-time windows

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `microstructure_has_data` | HYGIENE | Boolean: microstructure data available |
| 2 | `microstructure_activity` | HYGIENE | Activity score (0-1) |
| 3 | `microstructure_days_since_update` | HYGIENE | Days since last data update |

### Range & Volatility Ratios (8)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `micro_true_range` | RISK | True range (max of H-L, |H-C_prev|, |L-C_prev|) |
| 5 | `micro_atr_ratio` | RISK | ATR(14) / Close (volatility regime indicator) |
| 6 | `micro_range_pct` | RISK | (High - Low) / Close (daily range as % of price) |
| 7 | `micro_body_pct` | PREDICTIVE | |Close - Open| / Close (candle body size) |
| 8 | `micro_wick_top` | PREDICTIVE | (High - max(Open,Close)) / Close (upper wick) |
| 9 | `micro_wick_bottom` | PREDICTIVE | (min(Open,Close) - Low) / Close (lower wick) |
| 10 | `micro_shadow_ratio` | PREDICTIVE | (Wick_top + Wick_bottom) / (Body + ε) (wick dominance) |
| 11 | `micro_range_scaled` | RISK | (High - Low) / (ATR(14) + ε) (volatility normalization) |

### Volume & Liquidity Proxies (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `micro_volume_zscore` | PREDICTIVE | (Volume - μ_20d) / σ_20d (volume surprise) |
| 13 | `micro_volume_surge` | PREDICTIVE | Volume / μ_20d (relative volume) |
| 14 | `micro_volume_liquidity` | RISK | Volume × Range (liquidity proxy) |
| 15 | `micro_turnover` | RISK | Volume × Close (dollar turnover) |
| 16 | `micro_amihud` | RISK | |Return| / (Volume + ε) × 1e6 (illiquidity measure) |
| 17 | `micro_hl_volume_corr` | REGIME | Corr_20d(Range, Volume) (volume-volatility relationship) |

### Order Flow Proxies (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 18 | `micro_ofi_proxy` | PREDICTIVE | sign(Close - Open) × Volume (order flow imbalance) |
| 19 | `micro_signed_volume` | PREDICTIVE | Volume × Return_1d (signed volume) |
| 20 | `micro_pressure_proxy` | PREDICTIVE | (Close - Open) / (Range + ε) (buying pressure) |
| 21 | `micro_demand_supply_ratio` | PREDICTIVE | (Close - Low) / (High - Close + ε) (bid/ask dominance) |
| 22 | `micro_liquidity_imbalance` | PREDICTIVE | demand_supply_ratio - 0.5 (centered imbalance) |

### Volatility & Price Impact (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 23 | `micro_impact_ratio` | RISK | Range / (Volume + ε) × 1e9 (price impact per unit volume) |
| 24 | `micro_impact_volatility` | RISK | RV_5d / (Volume + ε) × 1e9 (volatility impact) |
| 25 | `micro_spread_proxy` | RISK | impact_ratio × 100 (bid-ask spread proxy) |
| 26 | `micro_vol_of_vol` | RISK | RollingStd_5d(RV_5d) (volatility of volatility) |
| 27 | `micro_intraday_vol_proxy` | RISK | Range / (Open + ε) (intraday volatility) |

### Staleness (Inactivity) Proxies (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 28 | `micro_stale_tick` | HYGIENE | Binary: 1 if High == Low (zero range) |
| 29 | `micro_zero_range_flag` | HYGIENE | Alias for stale_tick |
| 30 | `micro_low_liquidity_flag` | HYGIENE | Binary: 1 if Volume < 30% of μ_20d |

### Overnight Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 31 | `micro_overnight_gap` | PREDICTIVE | Open_t - Close_t-1 (overnight gap) |
| 32 | `micro_overnight_vol` | RISK | |Overnight_gap| / Close_t-1 (gap magnitude) |
| 33 | `micro_intraday_vs_overnight_vol` | RISK | Range / (|Gap| + ε) (intraday vs overnight ratio) |
| 34 | `micro_gap_direction` | PREDICTIVE | sign(Overnight_gap) (gap direction) |

### Optional Quote Features (9)

**Note:** Only populated on last row for near-real-time windows (end date within 2 days of today)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 35 | `micro_quote_bid_price` | HYGIENE | Top-of-book bid price (EODHD delayed quote) |
| 36 | `micro_quote_ask_price` | HYGIENE | Top-of-book ask price |
| 37 | `micro_quote_bid_size` | HYGIENE | Bid size (shares) |
| 38 | `micro_quote_ask_size` | HYGIENE | Ask size (shares) |
| 39 | `micro_quote_mid_price` | HYGIENE | (Bid + Ask) / 2 |
| 40 | `micro_quote_spread_abs` | RISK | Ask - Bid (absolute spread) |
| 41 | `micro_quote_spread_bps` | RISK | 10,000 × Spread / Mid (spread in basis points) |
| 42 | `micro_quote_imbalance` | PREDICTIVE | (Bid_size - Ask_size) / (Bid_size + Ask_size) |
| 43 | `micro_quote_has_data` | HYGIENE | Binary: 1 if quote data available |

---

### Interpretation Guide

| Condition | Microstructure State | Trading Implication |
|-----------|----------------------|---------------------|
| `amihud` > 100 | High illiquidity | Wide spreads, high impact cost |
| `amihud` < 10 | High liquidity | Tight spreads, low impact cost |
| `ofi_proxy` > 0 | Net buying pressure | Demand exceeds supply |
| `ofi_proxy` < 0 | Net selling pressure | Supply exceeds demand |
| `volume_surge` > 2 | Abnormal volume | News event, institutional activity |
| `volume_surge` < 0.5 | Below-average volume | Low participation, illiquid |
| `pressure_proxy` > 0.5 | Strong buying | Close near high |
| `pressure_proxy` < -0.5 | Strong selling | Close near low |
| `demand_supply_ratio` > 2 | Bid dominance | Buyers willing to pay up |
| `demand_supply_ratio` < 0.5 | Ask dominance | Sellers pressing |
| `spread_proxy` > 50 | Wide spread | Illiquid, high transaction costs |
| `spread_proxy` < 10 | Tight spread | Liquid, low transaction costs |
| `vol_of_vol` spike | Regime instability | Volatility regime changing |
| `stale_tick` = 1 | No trading | Halted or extremely illiquid |
| `low_liquidity_flag` = 1 | Illiquid session | Caution: high impact cost |
| `overnight_gap` > 2% | Major gap | News overnight, revaluation |
| `quote_imbalance` > 0.3 | Bid-heavy order book | Buying interest, support |
| `quote_imbalance` < -0.3 | Ask-heavy order book | Selling pressure, resistance |

**Data Sources:**
- **Primary**: EODHD daily OHLCV
- **Intraday (≤30d)**: EODHD 5m bars or AlphaVantage 1m/5m/15m bars
- **Quote Snapshot**: EODHD delayed quotes (last row only, near-real-time windows)

**Feature Categories:**
1. Range & Volatility Ratios (8)
2. Volume & Liquidity Proxies (6)
3. Order Flow Proxies (5)
4. Volatility & Price Impact (5)
5. Staleness Proxies (3)
6. Overnight Features (4)
7. Quote Features (9, optional)

---

## ml_framework Family — Full Feature Reference

**Source:** EODHD Technical Indicators API  
**Columns:** 13 (10 features + 3 governance)  
**Philosophy:** Pure EODHD-sourced technical indicators—NO local derivation, ensuring consistent external benchmarking.

**Design:** 
- All indicators fetched from EODHD Technical Indicators API
- 365-day lookback for rolling calculations
- No local computation (ensures reproducibility)
- Restricted to requested date range after fetching long history

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `ml_framework_has_data` | HYGIENE | Boolean: EODHD indicators available |
| 2 | `ml_framework_activity` | HYGIENE | Activity score (0-1) |
| 3 | `ml_framework_days_since_update` | HYGIENE | Days since last data update |

### Moving Averages (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `ml_sma_20` | REGIME | 20-day Simple Moving Average (trend anchor) |
| 5 | `ml_sma_50` | REGIME | 50-day Simple Moving Average (medium-term trend) |
| 6 | `ml_ema_12` | PREDICTIVE | 12-day Exponential Moving Average (fast EMA) |
| 7 | `ml_ema_26` | PREDICTIVE | 26-day Exponential Moving Average (slow EMA) |

### Bollinger Bands (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `ml_bbands_upper` | RISK | Upper Bollinger Band (20d, 2σ) |
| 9 | `ml_bbands_middle` | REGIME | Middle Bollinger Band (20d SMA) |
| 10 | `ml_bbands_lower` | RISK | Lower Bollinger Band (20d, 2σ) |

### Momentum & Volatility (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `ml_rsi_14` | PREDICTIVE | 14-day Relative Strength Index (0-100, overbought/oversold) |
| 12 | `ml_atr_14` | RISK | 14-day Average True Range (volatility measure) |

### MACD (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 13 | `ml_macd` | PREDICTIVE | MACD line (EMA_12 - EMA_26) |
| 14 | `ml_macd_signal` | PREDICTIVE | MACD signal line (9-day EMA of MACD) |
| 15 | `ml_macd_divergence` | PREDICTIVE | MACD divergence (MACD - Signal, histogram) |

---

### Interpretation Guide

| Condition | Indicator Signal | Trading Implication |
|-----------|------------------|---------------------|
| Price > `sma_50` | Above medium-term trend | Bullish regime |
| Price < `sma_50` | Below medium-term trend | Bearish regime |
| `ema_12` > `ema_26` | Fast EMA above slow | Bullish momentum |
| `ema_12` < `ema_26` | Fast EMA below slow | Bearish momentum |
| Price > `bbands_upper` | Above upper band | Overbought, potential reversal |
| Price < `bbands_lower` | Below lower band | Oversold, potential bounce |
| Price in (`bbands_lower`, `bbands_upper`) | Within bands | Normal range |
| `rsi_14` > 70 | Overbought | Potential pullback |
| `rsi_14` < 30 | Oversold | Potential rally |
| `rsi_14` 40-60 | Neutral zone | No clear signal |
| `atr_14` increasing | Rising volatility | Regime change, risk increasing |
| `atr_14` decreasing | Falling volatility | Consolidation, low risk |
| `macd` > `macd_signal` | MACD cross above signal | Bullish crossover, buy signal |
| `macd` < `macd_signal` | MACD cross below signal | Bearish crossover, sell signal |
| `macd_divergence` > 0 | Positive histogram | Bullish momentum strengthening |
| `macd_divergence` < 0 | Negative histogram | Bearish momentum strengthening |
| `macd_divergence` crossing 0 | Histogram zero cross | Momentum inflection point |

**Technical Indicator Parameters:**
- **SMA**: Simple Moving Average (20, 50 periods)
- **EMA**: Exponential Moving Average (12, 26 periods)
- **Bollinger Bands**: 20-day SMA ± 2 standard deviations
- **RSI**: Relative Strength Index (14-day period)
- **ATR**: Average True Range (14-day period)
- **MACD**: Fast=12, Slow=26, Signal=9

**Data Consistency:** All indicators sourced from EODHD API ensure reproducibility across different environments (no local calculation variance).

**Lookback Strategy:** 365-day lookback ensures sufficient history for all rolling calculations, then restricted to requested date range.

---

## multiasset Family — Full Feature Reference

**Source:** Tiingo Price Data (SPY, QQQ, IWM, ACWI)  
**Columns:** 15 (12 features + 3 governance)  
**Philosophy:** Clean equity benchmark exposures for structural risk measurement. Separates US equity (SPY/QQQ/IWM), global (ACWI), and style factors (market/growth-value/size) derived from PCA.

**Design:** Full time series with 120-day rolling windows. Orthogonal to cross_asset (which includes VIX/rates/credit).

### Overview

The multiasset family provides institutional-grade equity benchmark correlation and beta features. Alpha sources:
- **Beta regimes** — High SPY beta → reduce in risk-off
- **Style tilts** — QQQ vs IWM divergence → growth/value rotation
- **Spread dynamics** — Return spread vs SPY → decoupling signals
- **PCA factors** — Interpretable style exposures from benchmark decomposition

**Benchmark Coverage:**
- **SPY** (S&P 500): Market exposure
- **QQQ** (Nasdaq 100): Growth/tech tilt
- **IWM** (Russell 2000): Size tilt (small cap)
- **ACWI** (MSCI All Country World): Global risk exposure

**Window:** 120-day rolling correlations and betas

### Governance Columns (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `multiasset_has_data` | HYGIENE | Boolean: benchmark data available |
| 2 | `multiasset_activity` | HYGIENE | Activity score (0-1) |
| 3 | `multiasset_days_since_update` | HYGIENE | Days since last update |

### US Equity Benchmarks (7)

**SPY (S&P 500) Features:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `multiasset_corr_spy_120` | RISK | 120-day rolling correlation with S&P 500 |
| 5 | `multiasset_beta_spy_120` | RISK | 120-day rolling beta to S&P 500 (systematic risk) |
| 6 | `multiasset_spread_spy_20` | PREDICTIVE | 20-day rolling mean return spread (stock - SPY) |

**QQQ (Nasdaq 100) Features:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `multiasset_corr_qqq_120` | RISK | 120-day rolling correlation with Nasdaq 100 |
| 8 | `multiasset_beta_qqq_120` | RISK | 120-day rolling beta to Nasdaq 100 (growth/tech exposure) |

**IWM (Russell 2000) Features:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `multiasset_corr_iwm_120` | RISK | 120-day rolling correlation with Russell 2000 |
| 10 | `multiasset_beta_iwm_120` | RISK | 120-day rolling beta to Russell 2000 (small cap exposure) |

### Global Equity Factor (2)

**ACWI (MSCI All Country World) Features:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `multiasset_corr_acwi_120` | RISK | 120-day rolling correlation with ACWI |
| 12 | `multiasset_beta_acwi_120` | RISK | 120-day rolling beta to ACWI (global risk exposure) |

### Equity Style Factors — PCA-Derived (3)

**Generated only when all 4 benchmarks (SPY/QQQ/IWM/ACWI) available:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 13 | `multiasset_equity_factor_market` | RISK | PC1: Overall market exposure (SPY-dominant, explains most variance) |
| 14 | `multiasset_equity_factor_growth_value` | REGIME | PC2: Growth vs value tilt (QQQ vs SPY divergence) |
| 15 | `multiasset_equity_factor_size` | REGIME | PC3: Large vs small cap (SPY vs IWM divergence) |

---

### Interpretation Guide

| Condition | Market State | Trading Implication |
|-----------|--------------|---------------------|
| `beta_spy_120` > 1.5 | High systematic risk | Reduce exposure in risk-off regimes |
| `beta_spy_120` < 0.5 | Low market sensitivity | Defensive/idiosyncratic alpha |
| `corr_spy_120` < 0.3 | Decoupled from market | Stock-specific drivers dominate |
| `corr_spy_120` > 0.8 | High market correlation | Macro-driven, systemic risk |
| `spread_spy_20` > 0.01 | Outperforming market | Relative strength signal |
| `spread_spy_20` < -0.01 | Underperforming market | Relative weakness signal |
| `beta_qqq_120` > `beta_spy_120` | Growth tilt | Stock follows tech/growth more than market |
| `beta_iwm_120` > `beta_spy_120` | Size tilt | Stock follows small cap more than large |
| `corr_acwi_120` > `corr_spy_120` | Global exposure | International factors matter |
| `equity_factor_growth_value` > 0 | Growth tilt | Stock aligned with QQQ vs SPY |
| `equity_factor_growth_value` < 0 | Value tilt | Stock aligned with SPY vs QQQ |
| `equity_factor_size` > 0 | Large cap tilt | Stock moves with large caps |
| `equity_factor_size` < 0 | Small cap tilt | Stock moves with small caps |

---

### Design Notes

**Data Quality:**
- Young ETFs (e.g., ACWI pre-2008) are gracefully handled — features only computed when sufficient overlap exists
- Minimum 120 overlapping days required for valid rolling window
- Columns dropped if < 120 overlapping rows with symbol

**PCA Factors:**
- Rolling PCA on 4 benchmarks (SPY/QQQ/IWM/ACWI)
- PC1: Market factor (explains most variance, SPY-dominant)
- PC2: Growth vs Value (QQQ vs SPY divergence)
- PC3: Size factor (large vs small cap, SPY vs IWM)
- Only computed when all 4 benchmarks available

**Provenance:**
- Source: Tiingo price data
- Method: Equity benchmark exposure analysis
- Full time series (not snapshot)

---

## options Family — Full Feature Reference

**Source:** EODHD Options API  
**Columns:** 24 (23 features + 1 governance)  
**Philosophy:** Raw options chain snapshot for Stage A learning. NO directional scores — just clean metrics (IV, volume, OI, moneyness, term structure).

**Coverage:** US equities with active options markets  
**Update:** Snapshot only (latest available chain, not historical)  
**Leakage Policy:** Only snapshot date marked as `has_data=1`, all other dates set to 0 (no forward/backfill)

### Overview

The options family provides raw options chain metrics for detecting:
- **Implied volatility regimes** — ATM IV, call/put IV skew
- **Market sentiment** — Put/call volume and OI ratios
- **Liquidity conditions** — Bid-ask spreads, strike availability
- **Event proximity** — Days to nearest expiration

**NOT included:**
- Greeks (delta, gamma, vega, theta) → computed in `options_anchoring` family
- Directional signals → Stage A learns these from raw features
- Historical time series → snapshot only

**Data Structure:**
- Snapshot date: Real options data with `has_data=1`
- All other dates: Zeros with `has_data=0` (no propagation)
- Prevents leakage of future options chain into historical windows

### Governance Columns (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `options_has_data` | HYGIENE | Binary: 1 on snapshot date, 0 otherwise |

### Implied Volatility Features (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 2 | `options_atm_iv` | RISK | At-the-money implied volatility (avg of 5 nearest strikes) |
| 3 | `options_call_iv_avg` | RISK | Average call implied volatility across all strikes |
| 4 | `options_put_iv_avg` | RISK | Average put implied volatility across all strikes |
| 5 | `options_iv_spread` | RISK | Put IV - Call IV (put/call skew, fear premium) |

### Volume & Open Interest Features (6)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `options_call_volume` | REGIME | Total call volume across all strikes/expiries |
| 7 | `options_put_volume` | REGIME | Total put volume across all strikes/expiries |
| 8 | `options_put_call_volume_ratio` | REGIME | Put volume / Call volume (sentiment indicator) |
| 9 | `options_call_oi` | REGIME | Total call open interest |
| 10 | `options_put_oi` | REGIME | Total put open interest |
| 11 | `options_put_call_oi_ratio` | REGIME | Put OI / Call OI (structural positioning) |

### Pricing Features (7)

**Call Pricing:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `options_call_bid_avg` | HYGIENE | Average call bid price |
| 13 | `options_call_ask_avg` | HYGIENE | Average call ask price |
| 14 | `options_call_last_avg` | HYGIENE | Average call last traded price |

**Put Pricing:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 15 | `options_put_bid_avg` | HYGIENE | Average put bid price |
| 16 | `options_put_ask_avg` | HYGIENE | Average put ask price |
| 17 | `options_put_last_avg` | HYGIENE | Average put last traded price |

**Spread:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 18 | `options_bid_ask_spread_pct` | RISK | Average bid-ask spread as % of mid price (liquidity proxy) |

### Moneyness Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 19 | `options_strikes_available` | HYGIENE | Total number of unique strike prices |
| 20 | `options_otm_call_pct` | REGIME | % of call strikes that are out-of-the-money (strike > spot) |
| 21 | `options_otm_put_pct` | REGIME | % of put strikes that are out-of-the-money (strike < spot) |

### Term Structure Features (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 22 | `options_expiries_available` | HYGIENE | Number of unique expiration dates available |
| 23 | `options_nearest_expiry_days` | REGIME | Calendar days to nearest expiration |
| 24 | `options_confidence` | HYGIENE | Data confidence score (based on chain completeness) |

---

### Interpretation Guide

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `atm_iv` > 50% | High implied volatility | Elevated option prices, event/uncertainty |
| `atm_iv` < 20% | Low implied volatility | Cheap options, complacency |
| `iv_spread` > 5% | Put IV premium | Fear/downside protection demand |
| `iv_spread` < -5% | Call IV premium | Upside speculation/covered call selling |
| `put_call_volume_ratio` > 1.5 | Heavy put buying | Bearish sentiment or hedging |
| `put_call_volume_ratio` < 0.7 | Heavy call buying | Bullish sentiment or speculation |
| `put_call_oi_ratio` > 1.2 | Structural put positioning | Long-term hedges in place |
| `put_call_oi_ratio` < 0.8 | Structural call positioning | Long-term bullish bets |
| `bid_ask_spread_pct` > 10% | Wide spreads | Illiquid options, high transaction costs |
| `bid_ask_spread_pct` < 2% | Tight spreads | Liquid options market |
| `otm_call_pct` > 60% | Many upside strikes | Wide strike coverage, liquid market |
| `otm_put_pct` > 60% | Many downside strikes | Wide strike coverage |
| `nearest_expiry_days` < 7 | Near-term expiry | Weekly options, event-driven |
| `expiries_available` > 10 | Rich term structure | Active options market |

---

### Design Notes

**Snapshot Architecture:**
- EODHD provides latest options chain only (no historical chains)
- Single snapshot date marked with `has_data=1`
- All other dates filled with zeros and `has_data=0`
- Prevents temporal leakage into training windows

**ATM IV Calculation:**
- Uses 5 strikes nearest to current spot price
- Averages their implied volatilities
- More robust than single ATM strike

**Moneyness:**
- OTM calls: strike > current price
- OTM puts: strike < current price
- Percentage of total strikes in each category

**Data Quality:**
- Returns `None` if no options chain available
- Replaces inf/-inf with NaN, then fills with 0
- Confidence score based on chain completeness

**Telemetry:**
- Source: `eodhd_options_api`
- Feature type: `raw_options_metrics`
- No directional signals — Stage A learns patterns

---
## `options_anchoring` Family (Full Feature Reference)

**Purpose**: Institutional-grade options anchoring with decay weighting (Goldman/JPM/Susquehanna standard)

**Category**: Base Family  
**Data Source**: EODHD Options API (daily business-day cadence)  
**Fetch Frequency**: Daily (business days) - critical for IV/skew value  
**Cache**: Resumable on-disk parquet snapshots (`data/cache/options_anchoring/{symbol}_weekly_snapshots.parquet`)  
**Decay Weighting**: FAST (3d half-life), MEDIUM (5d), SLOW (20d)  
**Column Count**: 12 (11 features + 1 governance)

---

### Governance Columns (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `options_anchoring_days_since_update` | HYGIENE | Days since last successful options snapshot fetch |

### A. IV ANCHORING (3) - FAST DECAY (half-life 3 days)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 2 | `options_anchoring_iv_anchor_pct` | RISK | Z-score of ATM IV vs 20-day mean/std (regime detection) |
| 3 | `options_anchoring_iv_percentile_30d` | RISK | Percentile rank of current ATM IV in last 30 days (0-100) |
| 4 | `options_anchoring_iv_percentile_1yr` | RISK | Percentile rank of current ATM IV in last 252 days (0-100) |

### B. SKEW ANCHORING (3) - FAST DECAY (half-life 3 days)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `options_anchoring_iv_skew_anchor` | RISK | OTM put IV - OTM call IV (put/call skew, fear premium) |
| 6 | `options_anchoring_iv_skew_zscore` | RISK | Z-score of skew vs 30-day distribution |
| 7 | `options_anchoring_risk_reversal_25d` | RISK | 25-delta put IV - 25-delta call IV (Goldman/JPM standard) |

### C. EXPECTED MOVE ANCHORING (2) - FAST DECAY (half-life 3 days)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `options_anchoring_expected_move_pct` | PREDICTIVE | Straddle-implied move as % of spot (market's 1σ expectation) |
| 9 | `options_anchoring_em_vs_real_vol_ratio` | REGIME | Expected move / realized volatility (fwd-looking vs backward) |

### D. VOLUME POSITIONING (1) - MEDIUM DECAY (half-life 5 days)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 10 | `options_anchoring_put_call_vol_ratio_anchor` | REGIME | EMA(3) of put/call volume ratio (sentiment tracker) |

### E. OI POSITIONING (1) - SLOW DECAY (half-life 20 days)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `options_anchoring_put_call_oi_ratio_anchor` | REGIME | Put OI / call OI (structural positioning, slow moving) |

### F. DATA QUALITY (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 12 | `options_anchoring_confidence` | HYGIENE | Snapshot quality score (0-1, based on chain completeness) |

---

### Interpretation Guide

**IV Anchoring Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `iv_anchor_pct` > +2σ | IV extremely high vs recent | Expensive options, consider selling premium |
| `iv_anchor_pct` < -2σ | IV extremely low vs recent | Cheap options, consider buying protection |
| `iv_percentile_30d` > 90 | IV near 30-day highs | Short-term panic/event premium |
| `iv_percentile_1yr` > 90 | IV near 1-year highs | Sustained regime of uncertainty |
| `iv_percentile_30d` < 10 | IV near 30-day lows | Short-term complacency |

**Skew Anchoring Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `iv_skew_anchor` > +5% | Put IV premium | Fear, downside protection demand |
| `iv_skew_anchor` < -5% | Call IV premium | Greed, upside speculation |
| `iv_skew_zscore` > +2σ | Skew extremely wide | Panic selling/hedging spike |
| `risk_reversal_25d` > +8% | 25Δ RR expensive | Institutional fear (Goldman/JPM reference) |
| `risk_reversal_25d` < -3% | 25Δ RR cheap | Institutional complacency |

**Expected Move Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `expected_move_pct` > 10% | Large implied move | High uncertainty/event ahead |
| `expected_move_pct` < 3% | Small implied move | Low uncertainty, tight range expected |
| `em_vs_real_vol_ratio` > 1.5 | Forward vol > realized | Options pricing future uncertainty |
| `em_vs_real_vol_ratio` < 0.7 | Forward vol < realized | Options underpricing risk |

**Positioning Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `put_call_vol_ratio_anchor` > 1.5 | Heavy put volume | Bearish flow, active hedging |
| `put_call_vol_ratio_anchor` < 0.7 | Heavy call volume | Bullish flow, speculation |
| `put_call_oi_ratio_anchor` > 1.3 | Structural put bias | Long-term protective positioning |
| `put_call_oi_ratio_anchor` < 0.8 | Structural call bias | Long-term bullish bets |

---

### Design Notes

**Decay Weighting Philosophy:**
- **FAST (3-day half-life)**: IV/skew/expected move are highly reactive to news/events
- **MEDIUM (5-day)**: Volume ratios persist for several sessions but fade
- **SLOW (20-day)**: OI ratios represent structural positioning, slow to change
- Formula: `weight = 0.5^(days_since / halflife)`

**Institutional Standards:**
- **ATM IV**: Average of 5 nearest strikes (more robust than single strike)
- **Skew**: OTM puts (95% moneyness) vs OTM calls (105% moneyness)
- **Risk Reversal**: 25-delta approximation at ±10% moneyness (Goldman/JPM/Susquehanna standard)
- **Expected Move**: Straddle cost × 0.85 (market convention for 1σ)

**Rolling Metrics:**
- **Z-scores**: 20-day window for IV, 30-day for skew
- **Percentiles**: 30-day and 252-day (1 year) windows
- **EMA smoothing**: Span=3 for volume ratios to reduce noise

**Caching Strategy:**
- Resumable parquet snapshots prevent redundant API calls
- Atomic writes via temp files ensure cache integrity
- Configurable fetch frequency (`EODHD_OPTIONS_ANCHOR_FREQ`, default='B')
- Max samples cap (`EODHD_OPTIONS_ANCHOR_MAX_SAMPLES`, default=600)

**Leakage Prevention:**
- Business-day-aligned snapshots only
- No intraday data
- Historical lookback uses only past snapshots

**RealOptionsProvider Integration:**
- yfinance fallback for real options chains
- Monthly expiry selection (third Friday ≥7 days away)
- ATM ± 1 strike filtering for consistent analysis

**Data Quality:**
- Returns stub with zeros if no snapshots available
- Replaces inf/-inf with NaN
- Confidence score based on chain completeness per snapshot

**Telemetry:**
- Source: `eodhd_options_api_with_yfinance_fallback`
- Feature type: `institutional_options_anchoring`
- Decay model: `exponential_halflife_3_5_20`

---

## `regime` Family (Full Feature Reference)

**Purpose**: Institutional-grade regime classification with temporal context

**Category**: Base Family  
**Data Source**: Tiingo price data (OHLC daily bars)  
**Method**: Logistic activation on trend/momentum/slope metrics  
**Regime Types**: Bull (2), Neutral (1), Bear (0)  
**Column Count**: 9 (9 features, no separate governance)

---

### A. REGIME PROBABILITIES (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `regime_bull_probability` | REGIME | Bull regime probability (0-1, logistic activation) |
| 2 | `regime_bear_probability` | REGIME | Bear regime probability (0-1, logistic activation) |
| 3 | `regime_neutral_probability` | REGIME | Neutral regime probability (0-1, residual 1 - bull - bear) |

### B. REGIME LABEL (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 4 | `regime_label` | REGIME | Explicit regime classification (0=bear, 1=neutral, 2=bull) |

### C. TEMPORAL CONTEXT (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `regime_duration` | REGIME | Days since current regime started (capped at 250) |
| 6 | `regime_change_flag` | REGIME | Binary: 1 if regime changed in last 5 days, 0 otherwise |

### D. TREND METRICS (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `regime_trend_ratio` | PREDICTIVE | Fast MA (20d) / Slow MA (60d) trend strength |
| 8 | `regime_trend_ratio_zscore` | PREDICTIVE | Z-score of trend_ratio vs 1-year history (extreme detector) |
| 9 | `regime_volatility_20d` | RISK | 20-day realized volatility (annualized %) |

---

### Interpretation Guide

**Regime Probabilities:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `bull_probability` > 0.8 | Strong bull regime | High-confidence uptrend, favor long positions |
| `bull_probability` 0.5-0.8 | Moderate bull | Uptrend with caution, reduce leverage |
| `neutral_probability` > 0.6 | Range-bound market | Mean-reversion strategies, avoid trend following |
| `bear_probability` > 0.8 | Strong bear regime | High-confidence downtrend, defensive positioning |
| `bear_probability` 0.5-0.8 | Moderate bear | Downtrend forming, reduce exposure |

**Regime Label (Discrete):**

| Label | Regime | Interpretation |
|-------|--------|----------------|
| 2 | Bull | 20d MA > 60d MA, positive slope, momentum |
| 1 | Neutral | Mixed signals, range-bound, transition |
| 0 | Bear | 20d MA < 60d MA, negative slope, selling |

**Temporal Context:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `regime_duration` < 10 days | Fresh regime | New trend, high uncertainty, wait for confirmation |
| `regime_duration` 10-30 days | Established regime | Trend gaining conviction, safe to follow |
| `regime_duration` 30-100 days | Mature regime | Trend mature, consider partial profit-taking |
| `regime_duration` > 100 days | Exhaustion risk | Trend old, watch for reversal signals |
| `regime_change_flag` = 1 | Recent transition | Regime just changed, volatility spike expected |

**Trend Metrics:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `trend_ratio` > +0.10 | Strong uptrend | Fast MA 10%+ above slow MA, bullish |
| `trend_ratio` +0.02 to +0.10 | Moderate uptrend | Positive slope, gradual appreciation |
| `trend_ratio` -0.02 to +0.02 | Neutral/choppy | MAs converged, range-bound |
| `trend_ratio` -0.10 to -0.02 | Moderate downtrend | Negative slope, gradual decline |
| `trend_ratio` < -0.10 | Strong downtrend | Fast MA 10%+ below slow MA, bearish |
| `trend_ratio_zscore` > +3σ | Extreme overbought | Trend stretched vs 1yr history, reversal risk |
| `trend_ratio_zscore` < -3σ | Extreme oversold | Trend stretched downward, bounce candidate |
| `volatility_20d` > 40% | High vol regime | Elevated uncertainty, reduce position sizes |
| `volatility_20d` < 15% | Low vol regime | Complacency, potential volatility breakout ahead |

**Combined Gating Signals (Purpose):**

| Scenario | Signal Combination | Recommended Action |
|----------|-------------------|-------------------|
| Fresh bear, extreme trend | `bear_probability` > 0.8, `regime_duration` < 10, `trend_ratio_zscore` < -3 | **DO NOT FADE** - Wait for stabilization |
| Mature bull, neutral trend | `bull_probability` 0.5-0.7, `regime_duration` > 100, `trend_ratio_zscore` near 0 | Consider profit-taking, trend exhaustion |
| Recent change + high vol | `regime_change_flag` = 1, `volatility_20d` > 30% | Reduce leverage, regime uncertain |
| Established bull, low vol | `bull_probability` > 0.7, `regime_duration` 20-60, `volatility_20d` < 20% | Safe to deploy capital, stable uptrend |

---

### Design Notes

**Regime Classification Algorithm:**

1. **Base Metrics Calculation:**
   - MA Fast: 20-day rolling mean of close prices
   - MA Slow: 60-day rolling mean of close prices
   - Slope: 5-day rolling mean of MA_fast changes (momentum proxy)
   - Momentum 60: 60-day percentage change
   - Volatility 20d: 20-day rolling std of returns × √252 (annualized)

2. **Activation Function (Logistic):**
   ```
   activation = 4.0 × trend_ratio + 2.0 × slope + momentum_60
   bull_probability = 1 / (1 + exp(-activation))
   bear_probability = 1 / (1 + exp(activation))
   neutral_probability = 1 - (bull + bear)  [clipped 0-1]
   ```

3. **Regime Label Assignment:**
   - Regime = argmax(bull_prob, neutral_prob, bear_prob)
   - Mapped: {bear: 0, neutral: 1, bull: 2}

**Temporal Tracking:**
- **Regime Duration**: Days counter reset on label change, capped at 250 days
- **Regime Change Flag**: Rolling 5-day window, binary flag if any change detected

**Z-Score Calculation:**
- **Window**: 252 trading days (1 year)
- **Minimum periods**: 60 days for valid calculation
- **Formula**: `(trend_ratio - μ_252) / (σ_252 + 1e-6)`

**Data Requirements:**
- Minimum: 252 days of price history for zscore stability
- Returns `None` if insufficient data or missing close prices
- Uses Tiingo daily OHLC (NYSE trading hours)

**Leakage Prevention:**
- All metrics computed from past prices only
- No forward-looking information
- Forward-fill handled at aggregator level (not in handler)

**Use Case - Gating Signals:**
- **Purpose**: "Don't fade brand-new bear regime when trend extreme"
- **Example**: If `regime_duration` < 10 and `trend_ratio_zscore` < -3, avoid buying the dip (extreme downtrend just started)
- **Example**: If `regime_change_flag` = 1 and `volatility_20d` > 35%, reduce leverage (transition volatility spike)

**Telemetry:**
- Source: `tiingo_price_data`
- Method: `regime_classification_institutional`
- Features: 9 (no separate governance columns)

---

## `short_interest` Family (Full Feature Reference)

**Purpose**: Short interest tracking with squeeze probability analysis

**Category**: Base Family  
**Data Sources**: NASDAQ API (primary, 12+ months history), EODHD fundamentals (fallback), Yahoo Finance, Finviz scraper, FINRA provider (last resort)  
**Update Frequency**: Bi-monthly from NASDAQ (15th and end of month)  
**Forward-Fill**: Yes (bi-monthly settlement data forward-filled to daily)  
**Auto-Caching**: All historical snapshots cached for zscore calculation  
**Column Count**: 21 (20 features + 1 governance)

---

### Governance Columns (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `short_interest_has_data` | HYGIENE | Binary: 1 on/after first report date, 0 before |

### Core Short Interest Metrics (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 2 | `short_interest_percent` | RISK | Short interest as % of shares outstanding |
| 3 | `short_interest_ratio` | RISK | Short interest / average daily volume (days to cover) |
| 4 | `short_interest_days_to_cover` | RISK | Days to cover at average volume (alias of ratio) |
| 5 | `short_interest_float_short_pct` | RISK | Short interest as % of float (primary metric) |

### Change & Momentum Metrics (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `short_interest_change_1m` | PREDICTIVE | Month-over-month % change in short interest |
| 7 | `short_interest_change_3m` | PREDICTIVE | 3-month % change in short interest |
| 8 | `short_interest_momentum` | PREDICTIVE | Rolling trend in short interest (EMA of changes) |

### Statistical Context (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `short_interest_zscore_1y` | REGIME | Z-score of current short % vs 1-year history |
| 10 | `short_interest_pct_zscore_3y` | REGIME | Z-score of float_short_pct vs 3-year history |

### Positioning Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `short_interest_short_to_oi_ratio` | REGIME | Short interest / total options open interest |
| 12 | `short_interest_short_vs_institutional` | REGIME | Short interest vs institutional ownership ratio |

### Squeeze Risk Metrics (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 13 | `short_interest_squeeze_risk_flag` | PREDICTIVE | Binary: 1 if squeeze conditions met, 0 otherwise |
| 14 | `short_interest_squeeze_risk_score` | PREDICTIVE | Composite squeeze risk score (0-100) |
| 15 | `short_interest_squeeze_probability` | PREDICTIVE | Probability of short squeeze event (0-1) |
| 16 | `short_interest_shares_on_loan_pct` | RISK | % of shares currently on loan (borrow demand) |

### Borrow Rate Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 17 | `short_interest_borrow_rate` | RISK | Annual borrow rate for shorting stock (%) |
| 18 | `short_interest_borrow_rate_zscore_3y` | REGIME | Z-score of borrow rate vs 3-year history |

### Confidence & Scoring (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 19 | `short_interest_confidence` | HYGIENE | Data quality score (0-1, based on source) |
| 20 | `short_interest_conf` | HYGIENE | Alias of confidence for consistency |
| 21 | `short_interest_score_raw` | HYGIENE | Raw composite score before normalization |
| 22 | `short_interest_score` | HYGIENE | Normalized composite score (0-100) |

---

### Interpretation Guide

**Core Short Interest Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `float_short_pct` > 30% | Extreme short interest | High squeeze potential, crowded short |
| `float_short_pct` 20-30% | High short interest | Elevated risk of sharp rally |
| `float_short_pct` 10-20% | Moderate short interest | Normal skepticism, watch for catalysts |
| `float_short_pct` < 10% | Low short interest | Limited squeeze risk |
| `days_to_cover` > 10 | Illiquid shorts | Hard to cover, squeeze amplification |
| `days_to_cover` 5-10 | Moderate coverage time | Multi-day squeeze potential |
| `days_to_cover` < 5 | Easy to cover | Quick exit possible, muted squeeze |

**Change & Momentum Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `change_1m` > +20% | Rapidly increasing shorts | Bears piling in, potential capitulation setup |
| `change_1m` +5% to +20% | Gradual short buildup | Growing skepticism, monitor |
| `change_1m` -5% to +5% | Stable positioning | No significant shift |
| `change_1m` -20% to -5% | Gradual short covering | Bears reducing exposure |
| `change_1m` < -20% | Rapid short covering | Squeeze or catalyst-driven exit |
| `momentum` > 0 | Increasing short trend | Bears gaining confidence |
| `momentum` < 0 | Decreasing short trend | Bears losing confidence, covering |

**Statistical Context:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `zscore_1y` > +2σ | Unusually high shorts | Extreme bearish sentiment vs recent history |
| `zscore_1y` +1σ to +2σ | Elevated shorts | Above-average bearish positioning |
| `zscore_1y` -1σ to +1σ | Normal range | Typical short interest levels |
| `zscore_1y` < -2σ | Unusually low shorts | Extreme bullish sentiment or lack of skepticism |
| `pct_zscore_3y` > +3σ | 3-year extreme | Historic crowding, major squeeze risk |

**Squeeze Risk Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `squeeze_risk_flag` = 1 | Conditions met | High probability event, monitor closely |
| `squeeze_risk_score` > 80 | Extreme squeeze risk | Multiple red flags, explosive potential |
| `squeeze_risk_score` 60-80 | High squeeze risk | Significant catalyst could trigger |
| `squeeze_risk_score` 40-60 | Moderate risk | Watch for setup completion |
| `squeeze_probability` > 0.7 | High probability | 70%+ chance of squeeze event |
| `squeeze_probability` 0.4-0.7 | Moderate probability | Setup forming, needs catalyst |

**Borrow Rate Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `borrow_rate` > 15% | Extreme borrow cost | Hard-to-borrow, desperate shorts |
| `borrow_rate` 5-15% | High borrow cost | Elevated demand to short |
| `borrow_rate` < 5% | Normal borrow cost | Easy to short, no scarcity |
| `borrow_rate_zscore_3y` > +2σ | Historic high cost | Unprecedented borrow demand |
| `shares_on_loan_pct` > 80% | Nearly all float borrowed | Extreme utilization, squeeze catalyst |

**Combined Squeeze Setup:**

| Scenario | Signal Combination | Recommended Action |
|----------|-------------------|-------------------|
| Classic squeeze setup | `float_short_pct` > 25%, `days_to_cover` > 8, `borrow_rate` > 10% | **High alert** - Monitor for positive catalyst |
| Accelerating shorts | `float_short_pct` > 20%, `change_1m` > +15%, `momentum` > 0 | Bears capitulating setup, wait for turn |
| Historic crowding | `pct_zscore_3y` > +3σ, `squeeze_probability` > 0.6 | Extreme positioning, any good news triggers |
| Covering in progress | `change_1m` < -15%, `momentum` < 0, `borrow_rate` declining | Squeeze may be underway, momentum trade |

---

### Design Notes

**Data Source Priority (Cascading):**

1. **NASDAQ API** (primary):
   - 12-24 months of historical data
   - Official FINRA bi-monthly settlement reports (15th + end of month)
   - Highest data quality (marked as 'HIGH')
   - Auto-caches all historical snapshots for zscore calculation

2. **EODHD Fundamentals** (fallback):
   - Current + prior month short interest
   - From Technicals and SharesStats sections
   - Fields: SharesShort, SharesShortPriorMonth, ShortRatio, ShortPercent, ShortPercentFloat
   - Quality: 'HIGH' if real data, 'ESTIMATED' if calculated

3. **Yahoo Finance** (yfinance):
   - Current snapshot via `stock.info`
   - Fields: sharesShort, sharesShortPriorMonth, shortRatio, shortPercentOfFloat, dateShortInterest
   - Quality: 'HIGH'

4. **Finviz Scraper** (free, no API key):
   - Web scraping of Finviz.com
   - Fields: Short Float, Short Ratio, Short Interest
   - Requires no authentication

5. **FINRA Provider** (last resort):
   - Direct FINRA data if available
   - Only if all above sources fail

**Feature Calculation:**

- **Metrics-to-Features Pipeline**: `_metrics_to_features()` converts each report into 20+ features
- **Historical Time Series**: All reports converted to DataFrame, indexed by date
- **Daily Resampling**: Bi-monthly reports forward-filled to daily frequency
- **Pre-History Zeroing**: All values zero before first report date (honest coverage)
- **has_data Flag**: Binary indicator showing valid data region

**Squeeze Risk Algorithm:**

Composite scoring based on thresholds:
- **High short ratio**: >20% of float
- **Extreme short ratio**: >30% of float
- **High days to cover**: >5 days
- **Extreme days to cover**: >10 days
- **High borrow rate**: >5%
- **Extreme borrow rate**: >15%
- **High utilization**: >80% shares on loan
- **Extreme utilization**: >95% shares on loan

Score combines multiple factors weighted by severity.

**Auto-Caching Strategy:**

- **Cache Location**: Local persistent storage via `short_interest_cache`
- **Purpose**: Accumulate historical snapshots for zscore calculation (1-year, 3-year windows)
- **Trigger**: Every time new data fetched from any source, snapshot is cached
- **Benefits**: Enables statistical context (zscores) even with single-snapshot APIs (EODHD, Yahoo)

**Leakage Prevention:**

- **End-date filtering**: Reports after requested `end` date are excluded from feature calculation
- **No backfill**: Pre-history region (before first report) set to zero, not fabricated
- **Forward-fill only**: Bi-monthly data forward-filled (shorts persist), never backfilled

**Data Quality Handling:**

- **All-NaN check**: If all numeric features are NaN, emit stub with zeros
- **Inf replacement**: Replace inf/-inf with NaN, then fill with 0
- **Schema stability**: Consistent 21-column output (20 features + has_data)

**Stub Emissions (Deterministic):**

When data unavailable, returns stub DataFrame with:
- All features set to 0.0
- `has_data` = 0.0
- Status in telemetry: 'dormant:provider_unavailable', 'dormant:no_data', 'dormant:no_data_in_range', or 'error'

**Telemetry:**
- Source: `ShortInterestAnalyzer` (multi-source: NASDAQ/EODHD/Yahoo/Finviz/FINRA)
- Status: 'ok', 'dormant:*', or 'error'
- Features: 21 columns (20 features + 1 governance)

---

## `subsidiary` Family (Full Feature Reference)

**Purpose**: Organizational complexity from quarterly R&D and operating expense data

**Category**: Base Family  
**Data Source**: EODHD Quarterly Income Statement (researchDevelopment, totalOperatingExpenses)  
**Update Frequency**: Quarterly (forward-filled to daily)  
**Purpose**: Proxies for organizational scale, complexity, innovation investment, expansion/contraction  
**Column Count**: 12 (8 features + 4 governance)

---

### Governance Columns (4)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `subsidiary_has_data` | HYGIENE | Binary: 1 if quarterly data loaded, 0 otherwise |
| 2 | `subsidiary_activity` | HYGIENE | Activity indicator (0-1) |
| 3 | `subsidiary_days_since_update` | HYGIENE | Days since last quarterly report (999 if no data) |
| 4 | `subsidiary_confidence` | HYGIENE | Data quality score (0-1) |

### Core Spending Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 5 | `subsidiary_rd_spending` | REGIME | Quarterly R&D spending (absolute USD) |
| 6 | `subsidiary_opex_spending` | REGIME | Total operating expenses (absolute USD) |

### Intensity & Complexity (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 7 | `subsidiary_rd_intensity` | PREDICTIVE | R&D as % of operating expenses (innovation focus) |
| 8 | `subsidiary_complexity_score` | REGIME | Log(1 + OpEx) - scale-invariant complexity proxy |

### Quarter-over-Quarter Growth (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 9 | `subsidiary_rd_qoq_growth` | PREDICTIVE | QoQ % change in R&D spending (expansion/contraction) |
| 10 | `subsidiary_opex_qoq_growth` | PREDICTIVE | QoQ % change in operating expenses |

### Year-over-Year Growth (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `subsidiary_rd_yoy_growth` | PREDICTIVE | YoY % change in R&D spending (4-quarter comparison) |
| 12 | `subsidiary_opex_yoy_growth` | PREDICTIVE | YoY % change in operating expenses |

---

### Interpretation Guide

**Core Spending Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `rd_spending` increasing | Rising R&D investment | Innovation push, future growth bet |
| `rd_spending` decreasing | Cutting R&D | Cost reduction, margin defense, short-term focus |
| `opex_spending` > $1B | Large-scale operations | Complex organization, multiple divisions |
| `opex_spending` < $100M | Lean operations | Focused business, limited overhead |

**Intensity & Complexity:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `rd_intensity` > 20% | High R&D focus | Tech/biotech company, innovation-driven |
| `rd_intensity` 10-20% | Moderate R&D | Balanced growth/efficiency |
| `rd_intensity` < 10% | Low R&D | Mature business, commoditized products |
| `rd_intensity` < 5% | Minimal R&D | Low-tech, cost-focused, cyclical |
| `complexity_score` > 22 | Very complex org | Log(OpEx) > 22 ≈ $4B+ OpEx, large conglomerate |
| `complexity_score` 18-22 | Moderate complexity | Mid-cap complexity, multiple segments |
| `complexity_score` < 18 | Simple org | Small-cap, focused business model |

**Quarter-over-Quarter Growth:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `rd_qoq_growth` > +20% | Rapid R&D expansion | Aggressive innovation, product pipeline building |
| `rd_qoq_growth` +5% to +20% | Steady R&D growth | Normal expansion, consistent investment |
| `rd_qoq_growth` -5% to +5% | Stable R&D | Predictable innovation budget |
| `rd_qoq_growth` -20% to -5% | R&D pullback | Prioritization shift, margin pressure |
| `rd_qoq_growth` < -20% | R&D cuts | Cost crisis, strategic pivot, distress |
| `opex_qoq_growth` > +15% | Rapid expansion | Hiring, M&A integration, geographic expansion |
| `opex_qoq_growth` < -15% | Major restructuring | Layoffs, divestitures, cost cutting |

**Year-over-Year Trends:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `rd_yoy_growth` > +30% | Sustained R&D push | Multi-year innovation cycle, competitive response |
| `rd_yoy_growth` < -20% | Persistent R&D decline | Strategic shift away from innovation |
| `opex_yoy_growth` > `revenue_yoy_growth` | Margin compression | Operating leverage deteriorating |
| `opex_yoy_growth` < `revenue_yoy_growth` | Margin expansion | Operating leverage improving |

**Combined Signals:**

| Scenario | Signal Combination | Interpretation |
|----------|-------------------|----------------|
| Growth mode | `rd_qoq_growth` > +15%, `opex_qoq_growth` > +10% | Aggressive expansion, investing for growth |
| Efficiency drive | `rd_qoq_growth` < 0%, `opex_qoq_growth` < 0%, margins improving | Cost discipline, profitability focus |
| Innovation pivot | `rd_intensity` increasing, `rd_qoq_growth` > +20% | Strategic bet on new products/markets |
| Mature phase | `rd_intensity` < 10%, `opex_qoq_growth` near 0% | Stable operations, limited growth investment |
| Distress | `rd_qoq_growth` < -20%, `opex_qoq_growth` < -15% | Crisis mode, survival cuts |

---

### Design Notes

**Data Source Details:**

- **EODHD API**: Quarterly Income Statement
  - Field: `researchDevelopment` (R&D spending)
  - Field: `totalOperatingExpenses` (total OpEx)
- **Frequency**: Quarterly reports (filed 30-45 days after quarter end)
- **Forward-Fill**: Quarterly values forward-filled to daily frequency

**Feature Calculations:**

1. **R&D Intensity**: `(rd_spending / opex_spending) × 100`
   - Normalizes R&D investment by company size
   - Cross-sectionally comparable (% metric)

2. **Complexity Score**: `log(1 + opex_spending)`
   - Log transform for scale invariance
   - Larger companies have higher complexity scores
   - Smooth metric (no sudden jumps from small changes)

3. **QoQ Growth**: `((current_quarter - prior_quarter) / prior_quarter) × 100`
   - 1-period percentage change
   - Detects short-term shifts in spending

4. **YoY Growth**: `((current_quarter - same_quarter_last_year) / same_quarter_last_year) × 100`
   - 4-period (1-year) percentage change
   - Removes seasonal effects
   - Identifies sustained trends

**Data Quality Requirements:**

- Minimum: 4 quarters of data to calculate YoY growth
- If fewer than 4 quarters, returns stub with zeros and `has_data=0`
- Replaces inf/-inf with NaN, then forward-fills and back-fills
- `days_since_update` set to 999 if no data

**Leakage Prevention:**

- Quarterly data released 30-45 days after quarter end
- Forward-fill only (no backfill) to avoid lookahead
- Legitimate lag: Reports are known historical facts when filed

**Organizational Complexity Proxy:**

This family is named "subsidiary" for historical reasons but actually measures **organizational complexity** via operational metrics:
- **R&D spending** → Innovation capacity, product pipeline depth
- **Operating expenses** → Scale of operations, employee count proxy
- **Growth rates** → Expansion/contraction dynamics
- **Intensity** → Strategic focus (innovation vs efficiency)

True subsidiary relationship data would require SEC Exhibit 21 parsing (not currently implemented). Current implementation uses EODHD quarterly financials as complexity proxies.

**Alternative Data (Not Implemented):**

- SEC Exhibit 21 subsidiary lists (requires web scraping)
- GLEIF Level-2 Who-owns-Whom data (requires API integration)
- Segment reporting (available in EODHD but not parsed)

**Stub Emissions:**

When data unavailable, returns stub DataFrame with:
- All features set to 0.0
- `has_data` = 0.0
- `days_since_update` = 999.0
- Status in telemetry: 'dormant:missing_api_key', 'dormant:no_data', or 'error'

**Telemetry:**
- Source: `eodhd_quarterly_income_statement`
- Metrics: `['rd', 'opex', 'rd_intensity', 'rd_qoq_growth', 'opex_qoq_growth', 'rd_yoy_growth', 'opex_yoy_growth', 'complexity_score']`
- Type: `quarterly_opex_rd`

---

## `tft_features` Family (Full Feature Reference)

**Purpose**: TFT-style temporal patterns for forecasting (feature engineering only - NO ML model execution)

**Category**: Base Family  
**Data Source**: EODHD price data (OHLC daily bars)  
**Lookback**: 365 days for proper rolling calculations  
**Philosophy**: Multi-scale patterns, volatility regimes, seasonality, stability, dynamic confidence  
**Column Count**: 17 (17 features, no separate governance)

**IMPORTANT**: This family does **NOT** run an actual Temporal Fusion Transformer model. It performs pure feature engineering to create inputs that TFT models (or other forecasting models) would find useful. The actual TFT model implementation exists separately in `src/dcf_lab/models/transformer_tft.py` but is not invoked by this family.

---

### A. MULTI-SCALE TREND (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `tft_features_trend_short_strength` | PREDICTIVE | 5-10 day momentum bursts: (MA_10.diff(5) / close) |
| 2 | `tft_features_trend_long_strength` | PREDICTIVE | 20-30 day structural alignment: (MA_30.diff(10) / close) |
| 3 | `tft_features_trend_consistency` | REGIME | Fraction of positive returns in last 20 days (1=clean uptrend, 0=downtrend, 0.5=choppy) |
| 4 | `tft_features_trend_signal_to_noise` | PREDICTIVE | \|trend\| / volatility - strong trend relative to noise |
| 5 | `tft_features_trend_importance` | PREDICTIVE | Legacy blended trend importance: (abs(short) + abs(long)) / 2 |

### B. VOLATILITY REGIME & DYNAMICS (5)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 6 | `tft_features_vol_short` | RISK | 5-10 day realized volatility |
| 7 | `tft_features_vol_long` | RISK | 20-30 day baseline volatility |
| 8 | `tft_features_vol_regime_zscore` | REGIME | (current vol - median vol) / std vol - detects unusually high/low vol |
| 9 | `tft_features_vol_trend` | RISK | Is vol rising or calming? vol_long.diff(10) |
| 10 | `tft_features_volatility_importance` | RISK | Legacy volatility importance (alias of vol_long) |

### C. SEASONALITY & CALENDAR EFFECTS (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 11 | `tft_features_weekday_effect` | REGIME | Rolling average return by weekday (6-month window) |
| 12 | `tft_features_month_phase` | REGIME | Normalized day-of-month: -1 (start), 0 (mid), +1 (end) |
| 13 | `tft_features_seasonality_importance` | REGIME | Legacy seasonality: abs(sine-based pattern × returns).rolling(20).mean() |

### D. STABILITY / PREDICTABILITY (3)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 14 | `tft_features_stability_short` | REGIME | 5-10 day stability: 1 / (1 + abs(returns).rolling(10).mean()) |
| 15 | `tft_features_stability_long` | REGIME | 20-30 day stability: 1 / (1 + abs(returns).rolling(30).mean()) |
| 16 | `tft_features_stability_score` | REGIME | Legacy stability score (alias of stability_long) |

### E. DYNAMIC CONFIDENCE (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 17 | `tft_features_CONFIDENCE` | HYGIENE | Dynamic confidence: 0.3 + 0.4×sigmoid(-\|vol_regime_zscore\|) + 0.3×sigmoid(trend_snr - 0.5)×stability_long |

---

### Interpretation Guide

**Multi-Scale Trend Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `trend_short_strength` > +0.05 | Strong short-term momentum | 5-10 day burst, potential continuation |
| `trend_short_strength` < -0.05 | Short-term weakness | 5-10 day sell-off, reversal candidate |
| `trend_long_strength` > +0.03 | Sustained uptrend | 20-30 day structural strength |
| `trend_long_strength` < -0.03 | Sustained downtrend | 20-30 day structural weakness |
| `trend_consistency` > 0.7 | Clean uptrend | 70%+ positive days in last 20 |
| `trend_consistency` < 0.3 | Clean downtrend | 70%+ negative days |
| `trend_consistency` 0.4-0.6 | Choppy/range-bound | No clear directional bias |
| `trend_signal_to_noise` > 3.0 | High signal | Trend strong relative to noise, predictable |
| `trend_signal_to_noise` < 1.0 | Low signal | Noise dominates, unpredictable |

**Volatility Regime Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `vol_short` > 0.03 | High short-term vol | Daily swings > 3% (annualized ~47%) |
| `vol_short` < 0.01 | Low short-term vol | Daily swings < 1% (annualized ~16%) |
| `vol_regime_zscore` > +2σ | Extreme high vol | Crash/panic regime, vol spike |
| `vol_regime_zscore` +1σ to +2σ | Elevated vol | Above-average uncertainty |
| `vol_regime_zscore` -1σ to +1σ | Normal vol | Typical volatility range |
| `vol_regime_zscore` < -2σ | Extreme low vol | Complacency, potential breakout ahead |
| `vol_trend` > 0 | Rising volatility | Uncertainty increasing, caution |
| `vol_trend` < 0 | Calming volatility | Uncertainty decreasing, stabilizing |

**Seasonality & Calendar:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `weekday_effect` > +0.01 | Bullish weekday | This day of week tends positive (6-month avg) |
| `weekday_effect` < -0.01 | Bearish weekday | This day of week tends negative |
| `month_phase` near -1 | Start of month | Month-start effects (inflows, rebalancing) |
| `month_phase` near +1 | End of month | Month-end effects (window dressing, outflows) |

**Stability Signals:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `stability_short` > 0.8 | High short-term stability | Low daily moves, predictable |
| `stability_short` < 0.5 | Low short-term stability | High daily moves, chaotic |
| `stability_long` > 0.7 | High long-term stability | Smooth trends, low noise |
| `stability_long` < 0.5 | Low long-term stability | Erratic behavior, high noise |

**Dynamic Confidence:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `CONFIDENCE` > 0.8 | High confidence | Normal vol, strong trend signal, stable behavior |
| `CONFIDENCE` 0.5-0.8 | Moderate confidence | Acceptable conditions for forecasting |
| `CONFIDENCE` < 0.5 | Low confidence | Extreme vol or low signal-to-noise, uncertain |

**Combined Regime Detection:**

| Scenario | Signal Combination | Interpretation |
|----------|-------------------|----------------|
| Strong trending | `trend_consistency` > 0.7, `trend_signal_to_noise` > 2, `stability_long` > 0.6 | Clean trend, high predictability, favorable for momentum |
| High volatility event | `vol_regime_zscore` > +2, `stability_short` < 0.5 | Crash/panic, avoid forecasting |
| Complacent market | `vol_regime_zscore` < -2, `stability_long` > 0.7 | Low vol, smooth trends, breakout risk |
| Choppy range | `trend_consistency` 0.4-0.6, `trend_signal_to_noise` < 1.5 | No trend, high noise, mean-reversion |

---

### Design Notes

**TFT Philosophy:**

This family mimics inputs that **Temporal Fusion Transformers** (TFT) care about:
- **Multi-scale trends**: Short-term bursts vs long-term structural alignment
- **Volatility regimes**: Is current vol normal or extreme?
- **Seasonality**: Calendar effects (weekday, month-end)
- **Stability**: How predictable is the price action?
- **Dynamic confidence**: Model's self-assessment of forecast reliability

**Rolling Window Definitions:**

- **Short-term**: 5-10 days (captures momentum bursts, intraday patterns)
- **Long-term**: 20-30 days (captures structural trends, regime stability)
- **Lookback**: 252 days (1 year) for vol regime z-score
- **Weekday effect**: 126 days (6 months) for seasonal patterns

**Signal-to-Noise Calculation:**

```
trend_signal_to_noise = |MA_30.diff(10)| / (rolling_std_20 × close + 1e-8)
```

- High values (>3): Strong trend dominates noise
- Low values (<1): Noise dominates trend

**Volatility Regime Z-Score:**

```
vol_regime_zscore = (vol_long - median_vol_252) / (std_vol_252 + 1e-8)
```

- Detects extreme vol regimes (crashes, complacency)
- Median/std over 1-year window (min 60 periods)

**Stability Metrics:**

```
stability = 1 / (1 + abs(returns).rolling(window).mean())
```

- High stability (→1): Small daily moves, smooth trends
- Low stability (→0.5): Large daily moves, erratic

**Dynamic Confidence Formula:**

```
base_conf = sigmoid(-|vol_regime_zscore|)  # Penalize extreme vol
trend_conf = sigmoid(trend_signal_to_noise - 0.5)  # Reward strong trends
CONFIDENCE = 0.3 + 0.4×base_conf + 0.3×trend_conf×stability_long
```

- Combines: normal vol regime + strong trend signal + stability
- Range: [0.0, 1.0], clipped and default 0.5
- Warmup penalty: First 30 days × 0.5

**Seasonality Details:**

- **Weekday Effect**: For each date, compute average return for that weekday in last 126 days (6 months)
  - Example: If today is Monday, average all Monday returns in last 6 months
  - Captures day-of-week patterns

- **Month Phase**: Normalized position in month
  - Day 1-15: -1 to 0 (start of month)
  - Day 16-31: 0 to +1 (end of month)
  - Captures month-end rebalancing, window dressing

**Data Requirements:**

- **Minimum**: 30 days of price history (returns `None` if less)
- **Optimal**: 365+ days for stable vol regime z-scores and seasonality
- **Lookback**: Fetches 365 extra days before `start` for proper rolling calculations

**Leakage Prevention:**

- All metrics computed from past prices only
- No forward-looking information
- Returns calculated with `.pct_change()` (t / t-1 - 1)

**Legacy Features (Compatibility):**

Some features marked "legacy" for backward compatibility:
- `trend_importance`: Average of abs(short) + abs(long)
- `volatility_importance`: Alias of vol_long
- `seasonality_importance`: Sine-based approximation
- `stability_score`: Alias of stability_long

These maintain compatibility with older model versions.

**Use Case - Forecasting Input:**

This family is designed as **input features for time-series forecasting models** (NOT running the models themselves):
- **TFT (Temporal Fusion Transformers)**: Multi-scale attention over time-varying features (model exists in `src/dcf_lab/models/transformer_tft.py` but not invoked here)
- **N-BEATS**: Block-based temporal decomposition
- **DeepAR**: Autoregressive RNN with seasonality
- **Prophet**: Additive trend + seasonality components

**Key Distinction**: This family generates **features** that forecasting models consume. The actual model training/inference happens elsewhere in the pipeline (e.g., `quantile_forecast`, `arima_forecast` families or separate forecasting workflows).

**Telemetry:**
- Source: `EODHD`
- TFT version: `v2`
- Lookback days: `365`
- Features: 17 columns

---

## `peer_screener_context` Family (Full Feature Reference)

**Purpose**: Cross-sectional peer ranks/percentiles using cached fundamentals (sector/industry relative positioning)

**Category**: Base Family  
**Data Source**: Cached per-symbol fin_g6 (valuation multiples) + EODHD fundamentals metadata (sector/industry)  
**Grouping**: Sector + Industry (EODHD fundamentals General->Sector/Industry)  
**Leakage Policy**: All peer metrics shifted +1 NYSE session before ranking  
**Cache**: Shared cross-symbol cache under local_cache root (`_shared_cross_sectional/h{horizon}/peer_screener_context/`)  
**Column Count**: 24 (23 features + 1 governance)

---

### Governance Columns (1)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 1 | `peer_screener_context_has_data` | HYGIENE | Binary: 1 if peer data loaded, 0 otherwise |

### Sector Peer Context (6)

**Valuation Percentiles (Sector):**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 2 | `peer_screener_context_sector_pe_ratio_pct` | REGIME | Percentile rank of P/E ratio within sector (0-1, higher=expensive) |
| 3 | `peer_screener_context_sector_pe_ratio_cheap_pct` | PREDICTIVE | Cheapness percentile (0-1, higher=cheaper, inverted from pe_ratio_pct) |
| 4 | `peer_screener_context_sector_pb_ratio_pct` | REGIME | Percentile rank of P/B ratio within sector |
| 5 | `peer_screener_context_sector_pb_ratio_cheap_pct` | PREDICTIVE | Cheapness percentile for P/B |
| 6 | `peer_screener_context_sector_ev_ebitda_pct` | REGIME | Percentile rank of EV/EBITDA within sector |
| 7 | `peer_screener_context_sector_ev_ebitda_cheap_pct` | PREDICTIVE | Cheapness percentile for EV/EBITDA |

### Industry Peer Context (6)

**Valuation Percentiles (Industry - more granular than sector):**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 8 | `peer_screener_context_industry_pe_ratio_pct` | REGIME | Percentile rank of P/E ratio within industry |
| 9 | `peer_screener_context_industry_pe_ratio_cheap_pct` | PREDICTIVE | Cheapness percentile within industry |
| 10 | `peer_screener_context_industry_pb_ratio_pct` | REGIME | Percentile rank of P/B ratio within industry |
| 11 | `peer_screener_context_industry_pb_ratio_cheap_pct` | PREDICTIVE | Cheapness percentile for P/B within industry |
| 12 | `peer_screener_context_industry_ev_ebitda_pct` | REGIME | Percentile rank of EV/EBITDA within industry |
| 13 | `peer_screener_context_industry_ev_ebitda_cheap_pct` | PREDICTIVE | Cheapness percentile for EV/EBITDA within industry |

### Peer Count Metrics (2)

| # | Column | Role | Description |
|---|--------|------|-------------|
| 14 | `peer_screener_context_sector_peer_count` | HYGIENE | Number of valid sector peers on this date |
| 15 | `peer_screener_context_industry_peer_count` | HYGIENE | Number of valid industry peers on this date |

### Universe Context (Z-Scores - 5 metrics)

**Cross-Universe Standardized Metrics:**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 16 | `peer_screener_context_universe_momentum_z` | PREDICTIVE | Z-score of 3-month momentum vs universe (log transform) |
| 17 | `peer_screener_context_universe_valuation_z` | PREDICTIVE | Z-score of valuation vs universe (neg_log transform, higher=cheaper) |
| 18 | `peer_screener_context_universe_options_iv_z` | RISK | Z-score of ATM IV vs universe |
| 19 | `peer_screener_context_universe_options_skew_z` | RISK | Z-score of IV skew vs universe |
| 20 | `peer_screener_context_universe_short_interest_z` | RISK | Z-score of short interest % vs universe |

### Universe Context (Percentiles - 5 metrics)

**Cross-Universe Percentile Ranks (0-1):**

| # | Column | Role | Description |
|---|--------|------|-------------|
| 21 | `peer_screener_context_universe_momentum_pct` | PREDICTIVE | Percentile rank of momentum (0=worst, 1=best) |
| 22 | `peer_screener_context_universe_valuation_pct` | PREDICTIVE | Percentile rank of cheapness (0=expensive, 1=cheap) |
| 23 | `peer_screener_context_universe_options_iv_pct` | RISK | Percentile rank of IV (0=low vol, 1=high vol) |
| 24 | `peer_screener_context_universe_options_skew_pct` | RISK | Percentile rank of skew (0=low fear, 1=high fear) |

---

### Interpretation Guide

**Sector/Industry Percentiles:**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `sector_pe_ratio_cheap_pct` > 0.8 | Cheap vs sector | Potential value opportunity within sector |
| `sector_pe_ratio_cheap_pct` < 0.2 | Expensive vs sector | Crowded trade or momentum |
| `industry_pe_ratio_cheap_pct` > 0.8 | Cheap vs industry | More refined value signal (granular peers) |
| `industry_ev_ebitda_cheap_pct` > 0.8 | Cheap on EV/EBITDA | Potential M&A target or distressed |
| `industry_peer_count` < 3 | Thin peer group | Percentile unreliable, fall back to sector |
| `sector_peer_count` < 3 | No valid peers | Isolated stock, use universe context only |

**Universe Context (Z-Scores):**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `universe_momentum_z` > +2σ | Strong momentum vs universe | Leader, potential continuation |
| `universe_momentum_z` < -2σ | Weak momentum vs universe | Laggard, potential reversal candidate |
| `universe_valuation_z` > +2σ | Cheap vs universe | Deep value, contrarian bet |
| `universe_valuation_z` < -2σ | Expensive vs universe | Growth premium or bubble |
| `universe_options_iv_z` > +2σ | High IV vs peers | Idiosyncratic risk/event |
| `universe_options_skew_z` > +2σ | Extreme skew vs peers | Isolated fear/hedging demand |

**Universe Context (Percentiles):**

| Metric | Signal | Trading Implication |
|--------|--------|---------------------|
| `universe_momentum_pct` > 0.9 | Top decile momentum | Trend follower's target |
| `universe_momentum_pct` < 0.1 | Bottom decile momentum | Mean-reversion candidate |
| `universe_valuation_pct` > 0.9 | Top decile cheapness | Deep value screener hit |
| `universe_valuation_pct` < 0.1 | Bottom decile cheapness | Expensive/growth |
| `universe_options_iv_pct` > 0.9 | Top decile IV | Event risk or earnings volatility |

**Combined Signals:**

| Sector Signal | Industry Signal | Universe Signal | Interpretation |
|---------------|----------------|-----------------|----------------|
| Cheap (>0.8) | Cheap (>0.8) | Cheap (>0.9) | **Triple consensus value** - strong signal |
| Cheap (>0.8) | Expensive (<0.2) | Neutral (0.4-0.6) | Industry rotation, sector-level theme |
| Expensive (<0.2) | Expensive (<0.2) | Expensive (<0.1) | **Triple consensus expensive** - avoid or short |
| Neutral | Neutral | Extreme momentum (>0.9) | Universe-level leadership, sector-agnostic |

---

### Design Notes

**Grouping Hierarchy:**
- **Industry** (most granular): Direct competitors (e.g., "Software - Application")
- **Sector** (broader): Industry group (e.g., "Technology")
- **Universe** (broadest): All symbols in candidate universe (DEFAULT_CANDIDATE_UNIVERSE or env override)

**Percentile Calculation:**
- Row-wise rank across peer columns (axis=1)
- `higher_is_better=True` for raw percentile
- `higher_is_better=False` for "cheap" variants (inverted)
- Neutral default (0.5) for missing/degenerate groups avoids encoding missing as extremes

**Leakage Prevention:**
- All peer metrics shifted +1 NYSE session before ranking
- Prevents same-day lookahead from filings/snapshots
- Ensures cross-sectional ranks use only known information

**Minimum Peer Thresholds:**
- **Industry**: `PEER_SCREENER_CONTEXT_MIN_GROUP_PEERS` (default=3)
- **Sector**: Same threshold, used as fallback when industry too thin
- **Universe**: Adaptive threshold based on max observed peer count (avoids gating in partial-universe runs)

**Universe Discovery:**
1. **Explicit override**: `PEER_SCREENER_CONTEXT_UNIVERSE` env var (comma-separated)
2. **Default**: `DEFAULT_CANDIDATE_UNIVERSE` from `universe_selector.py`
3. **Fallback**: Target symbol only if no universe available

**Live Fetch Capability:**
- If peer symbol not in local_cache, can fetch on-demand via `build_panel`
- Controlled by `PEER_SCREENER_CONTEXT_LIVE_FETCH` (default=1)
- Memoized per-(symbol,family) to avoid redundant builds
- Re-masks data where `{family}_has_data==0` to prevent zero-filled rows from counting as valid peers

**Shared Cache Strategy:**
- Wide cross-symbol matrix cached once per (start,end,horizon)
- Path: `{cache_root}/_shared_cross_sectional/h{horizon}/peer_screener_context/v{version}/{start}_{end}.parquet`
- Reused across all symbols to avoid recomputation
- Universe context computed per-symbol and uses separate stats cache

**Universe Stats Cache:**
- Small per-date universe mean/std/count for 5 metrics
- Avoids writing/reading enormous wide matrices
- Path: `{cache_root}/_shared_cross_sectional/h{horizon}/peer_screener_context/v{version}/{start}_{end}_universe_stats.parquet`

**Metric Transforms:**
- **Momentum** (`dcf_mom_3m`): Log transform for symmetry
- **Valuation** (`fin_g6_pe_ratio`): Negative log transform (higher=cheaper)
- **Options IV** (`options_atm_iv`): Identity (raw values)
- **Options Skew** (`options_iv_spread`): Identity
- **Short Interest** (`short_interest_float_short_pct`): Identity

**Data Quality:**
- Returns stub (zeros + neutral 0.5 percentiles) if no cache_root or target missing
- Sector/industry metadata from EODHD fundamentals (`get_profile`)
- Fallback to sector percentiles when industry peer count < min threshold
- Z-scores clipped to ±8σ to prevent extreme outliers

**Cache Version:**
- `PEER_SCREENER_CONTEXT_SHARED_CACHE_VERSION = 2`
- Bump to invalidate old caches when rank/percentile logic changes

**Telemetry:**
- `source`: "shared_cache" or "computed"
- `base_family`: "fin_g6" (valuation multiples)
- `grouping`: "sector+industry (EODHD fundamentals)"
- `leakage`: "peer metrics shifted +1 NYSE session(s)"
- `universe_source`: "env", "DEFAULT_CANDIDATE_UNIVERSE", or "target_only"
- `min_peers_configured`, `min_peers_effective`: Adaptive gating thresholds

---