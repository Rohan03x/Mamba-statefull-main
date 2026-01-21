#!/usr/bin/env python3
"""
Lag configuration registry for all feature families.

Defines which features get lagged and by how many periods for Stage A optimization.
Each family has pattern-based rules mapping feature name patterns to lag periods.

Usage:
    from src.features.lag_config import get_lag_config, apply_lags
    
    lag_rules = get_lag_config('ml_framework')
    df_with_lags = apply_lags(df, lag_rules)
"""

from typing import Dict, List, Tuple
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# LAG CONFIGURATION REGISTRY
# ============================================================================

LAG_CONFIG: Dict[str, List[Tuple[List[str], List[int]]]] = {
    'ml_framework': [
        # Signals (price vs MA/EMA, indicators)
        (['price_vs_ma_', 'price_vs_ema_', 'bb_width', 'bb_position', 'macd', 'macd_signal', 'rsi'], [1, 5]),
        # Raw MA/EMA levels
        (['ma_', 'ema_'], [5]),
        # Returns
        (['return_', 'log_return_'], [1, 5, 10]),
        # Volatility
        (['volatility_', 'volatility_annualized_'], [5, 20]),
        # Higher moments
        (['skewness_', 'kurtosis_'], [5]),
        # Volume MAs
        (['volume_ma_'], [5, 20]),
        # Volume ratios & price-volume
        (['volume_ratio_', 'price_volume', 'volume_weighted_price'], [1, 5]),
        # Price levels/ranges
        (['price_ratio_', 'price_position_', 'percentile_'], [1, 5]),
        # Confidence
        (['CONFIDENCE'], [1, 5]),
        # Autocorr
        (['autocorr_lag_'], [1]),
        # Temporal features get NO lags
        (['day_of_year', 'week_of_year', 'quarter', 
          'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end'], []),
    ],
    
    'microstructure': [
        # Liquidity & impact
        (['micro_amihud', 'micro_spread_proxy', 'micro_impact_ratio', 'micro_impact_volatility',
          'micro_liquidity_imbalance', 'micro_volume_liquidity', 'micro_turnover'], [1, 3]),
        (['micro_low_liquidity_flag'], [1]),
        # Order flow & pressure
        (['micro_ofi_proxy', 'micro_pressure_proxy', 'micro_signed_volume', 'micro_demand_supply_ratio'], [1, 3]),
        # Volatility & ranges
        (['micro_intraday_vol_proxy', 'micro_overnight_vol', 'micro_intraday_vs_overnight_vol',
          'micro_vol_of_vol', 'micro_true_range', 'micro_atr_ratio', 'micro_range_pct',
          'micro_range_scaled', 'micro_body_pct', 'micro_shadow_ratio', 'micro_wick_top', 'micro_wick_bottom'], [1, 3]),
        (['micro_zero_range_flag'], [1]),
        # Gaps & overnight
        (['micro_overnight_gap', 'micro_gap_direction'], [1, 3]),
        # Volume dynamics
        (['micro_volume_surge', 'micro_volume_zscore', 'micro_hl_volume_corr'], [1, 3]),
        (['micro_stale_tick'], [1]),
        # Confidence
        (['CONFIDENCE'], [1, 3]),
    ],
    
    'garch_iv': [
        # === BASELINE VOL METRICS (4 features) ===
        # Volatility levels - track if vol has been persistently high/low
        (['short_vol', 'long_vol', 'vol_ratio'], [1, 5]),
        # Instantaneous shocks - short persistence over 1-3 days
        (['last_shock'], [1, 3]),
        
        # === VOL TERM STRUCTURE (3 features) ===
        # Slow regime drift signals, not noisy daily toggles
        (['mid_vol_60', 'vol_slope_short_mid', 'vol_slope_mid_long'], [5, 20]),
        
        # === VOL-OF-VOL STABILITY (2 features) ===
        # You care if volatility-of-volatility is high for weeks, not just today
        (['vol_of_vol_20', 'vol_of_vol_ratio'], [5, 20]),
        
        # === GARCH STATE (5 features) ===
        # Structural state of the vol process
        (['garch_cond_var', 'garch_persistence', 'garch_half_life', 'garch_long_run_var'], [5, 20]),
        # Sequence of "big standardized shocks"
        (['garch_std_resid_sq'], [1, 3]),
        
        # === ASYMMETRY/LEVERAGE (2 features) ===
        # Binary flag; 1 lag is enough
        (['neg_shock_dummy'], [1]),
        # Slow-moving measure of leverage effect
        (['shock_asymmetry_20'], [5, 20]),
        
        # === REGIME & Z-SCORES (6 features) ===
        # Normalized regime indicators
        (['short_vol_z_252', 'long_vol_z_252'], [5, 20]),
        # Binary regime states - want "has this been in regime for a few days?"
        (['vol_low_regime', 'vol_high_regime', 'vol_spike_flag', 'vol_crush_flag'], [1, 5]),
    ],
    
    'cboe_term': [
        # === BASIC TERM SLOPES (3 features) ===
        # VIX term structure shifts are slower
        (['vxst_vix_term_slope', 'vix_vxv_term_slope', 'vix_vxmt_term_slope'], [1, 5]),
        
        # === NORMALIZED & CURVATURE (2 features) ===
        (['normalized_term_slope', 'vix_term_curvature'], [1, 5]),
        
        # === SHOCK & PANIC (2 features) ===
        # Short-term panic tends to last a couple of days
        (['front_back_spread', 'panic_premium'], [1, 5]),
        
        # === REGIME FLAGS (3 features) ===
        (['contango_flag', 'backwardation_flag', 'curve_shape_flag'], [1]),
        
        # === RATIOS & YIELD (3 features) ===
        # Variants of the same concept: term structure positioning
        (['vix_roll_yield', 'vix_ratio_term', 'vix_contango_strength'], [1, 5]),
        
        # === VOL RISK PREMIUM (2 features) ===
        # Slow risk premium / "fear vs realized" relationship
        (['vol_risk_premium', 'vol_risk_premium_pct'], [5, 20]),
    ],
    
    'correlation': [
        # ========================================================================
        # CORRELATION ALPHA ENGINE LAG STRATEGY: TACTICAL MIXED-FREQUENCY
        # ========================================================================
        # 7 alpha engines + traditional correlations = 120-140 features
        # Lag expansion: ~120 → ~240 features (2.0x expansion)
        
        # === 1️⃣ TRADITIONAL ROLLING CORRELATIONS (15 base features) ===
        # CORR_20_*, CORR_60_* → [1, 5] (lag yesterday's structural correlation + 1-week)
        # CORR_5_* → NO LAGS (too noisy, already have spreads/momentum)
        (['CORR_20_', 'CORR_60_'], [1, 5]),
        (['CORR_5_'], []),  # No lags on 5-day window
        
        # === BREAK FLAGS (30 features) ===
        # Did the regime break persist from yesterday?
        (['_break_low', '_break_high'], [1]),
        
        # === 2️⃣ ALPHA ENGINE 1: LAG_CORR_* (20 features) → NO LAGS ===
        # Already lagged correlations by definition (LAG_CORR_1_*, LAG_CORR_2_*, etc.)
        # Extra lags = redundant feature bloat
        (['LAG_CORR_'], []),
        
        # === 3️⃣ ALPHA ENGINE 2: CORRELATION MOMENTUM/TREND (15 features) ===
        # *_TREND features (e.g., CORR_20_SPY_TREND)
        # Slope already computed; [5] lag gives "trend of trend" without noise
        (['_TREND'], [5]),
        
        # === 4️⃣ ALPHA ENGINE 3: CORRELATION VOLATILITY (15 features) ===
        # *_VOL features (e.g., CORR_20_SPY_VOL)
        # Stability of relationship → weekly + ~1-month drift matter
        (['_VOL'], [5, 20]),
        
        # === 5️⃣ ALPHA ENGINE 4: CORRELATION SPREADS (10 features) ===
        # CORR_SPREAD_5_20_*, CORR_SPREAD_20_60_*
        # Already cross-window; [5] lag shows curve flattening/steepening over time
        (['CORR_SPREAD_'], [5]),
        
        # === 6️⃣ ALPHA ENGINE 5: CROSS-FEATURE CORRELATIONS (3 features) ===
        # CORR_RETURN_VOL_20, CORR_RETURN_RANGE_10, CORR_VOL_VOLATILITY_20
        # Structural microstructure links; weekly + monthly lags capture regime change
        (['CORR_RETURN_VOL', 'CORR_RETURN_RANGE', 'CORR_VOL_VOLATILITY'], [5, 20]),
        
        # === 7️⃣ ALPHA ENGINE 6: VIX LAG CORRELATIONS (4 features) → NO LAGS ===
        # CORR_20_VIX_lag1, CORR_20_VIX_lag2, CORR_60_VIX_lag1, CORR_60_VIX_lag2
        # Already explicitly about lagged VIX; meta-lags add very little
        (['VIX_lag'], []),
        
        # === 8️⃣ ALPHA ENGINE 7: ACF FEATURES (6 features) ===
        # ACF_RET_1, ACF_RET_2, ACF_ABSRET_1, ACF_ABSRET_2, ACF_VOL_1, ACF_VOL_2
        # Memory descriptors; [5] lag shows if memory structure itself is changing
        (['ACF_'], [5]),
    ],
    
    'options': [
        # === SPARSE SNAPSHOT DATA - MINIMAL LAGS ===
        # Only lag [1] for the most stable features to avoid propagating missingness
        
        # === IV (4 features) ===
        (['atm_iv', 'call_iv_avg', 'put_iv_avg', 'iv_spread'], [1]),
        
        # === VOLUME & OI (6 features) ===
        (['call_volume', 'put_volume', 'put_call_volume_ratio',
          'call_oi', 'put_oi', 'put_call_oi_ratio'], [1]),
        
        # === PRICING (7 features) ===
        # No lags for bid/ask/last - too noisy and sparse
        # (['call_bid_avg', 'call_ask_avg', 'call_last_avg',
        #   'put_bid_avg', 'put_ask_avg', 'put_last_avg', 'bid_ask_spread_pct'], []),
        
        # === MONEYNESS & TERM STRUCTURE (5 features) ===
        # Already fairly stable, lag [1] at most
        (['strikes_available', 'otm_call_pct', 'otm_put_pct',
          'nearest_expiry_days', 'expiries_available'], [1]),
    ],
    
    'short_interest': [
        # === SHORT INTEREST LEVELS (5 features) → [5, 20] ===
        # Level-like features: short_interest_pct, short_ratio, borrow_rate,
        # short_vs_institutional, days_to_cover_zscore
        # Updates weekly/biweekly - care about trend over weeks-months
        (['short_interest_pct', 'short_interest_ratio', 'short_ratio',
          'borrow_rate', 'short_vs_institutional', 
          'days_to_cover_zscore'], [5, 20]),
        
        # === SHORT INTEREST DYNAMICS (3 features) → [5] ===
        # Change/dynamics: short_interest_change, short_squeeze_risk, short_momentum
        # Trend of trend - [5] lag shows if dynamics are accelerating/decelerating
        (['short_interest_change', 'short_squeeze_risk', 'short_momentum',
          'change_1m', 'change_3m', 'squeeze_risk_score', 'squeeze_probability'], [5]),
    ],
    
    'earnings': [
        # === NO LAGS - EVENT FAMILY ===
        # Earnings is event-based. History already encoded in features:
        # - days_since_earnings
        # - pre_earnings_drift, post_earnings_drift
        # - earnings_momentum
        # - earnings_volatility, earnings_quality_score
        # - beat_streak, beat_rate_3y
        # - revision_breadth, estimate_dispersion
        # Adding lags would create duplicate values between quarterly events
        # Rely on: pre/post drift, momentum, volatility of surprises, beat metrics
    ],
    
    'dividends': [
        # === NO LAGS - EVENT/TIME-BASED FAMILY ===
        # Dividends is event-driven. History already embedded in features:
        # - dividend_growth_rate
        # - dividend_consistency
        # - dividend_sustainability
        # - dividend_aristocrat_flag
        # - days_to_ex_dividend
        # Generic daily lags add noise beyond what's already captured
        # Note: dividend_yield lags covered in FIN_G7 (dividend_yield_proxy)
    ],
    
    'subsidiary': [
        # === SUBSIDIARY STRUCTURE (25 features) → [60, 252] ===
        # Very structural; changes rarely (quarterly/annual updates)
        # Count, complexity, geographic diversification metrics
        # Lags: previous quarter + previous year state
        (['subsidiary_count', 'country_count', 'sector_count', 'region_count',
          'complexity_score', 'foreign_exposure_pct', 'domestic_exposure_pct',
          'hhi_score', 'depth_score', 'global_sprawl_score',
          'top_country_pct', 'entropy', 'top_region_pct', 'top_sector_pct',
          'emerging_market_exposure_pct', 'developed_market_exposure_pct',
          'fx_risk_score', 'political_risk_score',
          'avg_employees', 'revenue_coverage_pct',
          'data_completeness_pct', 'missing_regions_flag',
          'international_pct', 'geographic_diversification', 
          'subsidiary_growth', 'consolidated_complexity'], [60, 252]),
    ],
    
    'alternative_signals': [
        # A. Price Anomalies - fast, transient patterns
        # Gaps, overnight returns, trend acceleration, opening/closing dynamics
        (['gap_', 'overnight_return', 'trend_acceleration', 
          'opening_reversal', 'closing_ramp'], [1, 3, 5]),
        
        # B. Volume Anomalies - fast, short-lived liquidity stress signals
        # Volume z-scores, divergences, liquidity stress, opening surges
        (['volume_zscore', 'volume_divergence', 'liquidity_stress', 
          'opening_volume_surge'], [1, 3, 5]),
        
        # C. Earnings & Seasonality - temporal encodings, NO lags needed
        # Calendar patterns (day-of-week, month-end), earnings timing flags
        # Already cyclical/time-indexed by construction
        (['seasonality_', 'earnings_calendar_', 'is_month_end', 
          'is_quarter_end', 'earnings_week_flag'], []),
        
        # D. Free Sentiment - slower moving, persistence matters
        # Google Trends, Reddit mentions, news volume
        (['sentiment_', 'google_trends', 'reddit_mentions', 'news_volume'], [1, 3, 5, 10]),
        
        # E. Realized Volatility - conservative lags
        # Realized vol estimators and ratios
        (['realized_vol_', 'vol_ratio_'], [1, 5, 20]),
        
        # F. Intraday Anomalies - NEW: time-of-day patterns
        # Session momentum, midday drift, intraday vol ratios
        # Fast-decaying intraday effects
        (['midday_drift', 'session_momentum', 'intraday_volatility_ratio',
          'time_weighted_returns', 'liquidity_hourglass'], [1, 3, 5]),
    ],
    
    'quantile_forecast': [
        # a) Stationary inputs - all continuous signals, small+medium lags to see if structure is changing/stabilising
        # Log returns
        (['r1', 'r5', 'r10', 'r20'], [1, 3, 5]),
        # Volatility measures
        (['realized_vol_5', 'realized_vol_10', 'realized_vol_20', 'high_low_range_pct', 'intraday_range'], [1, 3, 5]),
        # Rolling moments
        (['rolling_mean_5', 'rolling_skew_20', 'rolling_kurt_20'], [1, 3, 5]),
        # Price vs trend (relative positioning)
        (['close_vs_sma10', 'close_vs_sma20', 'close_vs_sma50'], [1, 3, 5]),
        # Volume structure
        (['volume_zscore', 'volume_change_pct'], [1, 3, 5]),
        
        # b) Regime indicators - binary/flag-like, lag [1] only to see persistence without correlation explosion
        (['low_vol_flag', 'high_vol_flag', 'breakout_flag'], [1]),
        
        # c) Output feature - crucial for model to know if expected return has been consistently up/down
        (['quantile_forecast_expected_return'], [1, 3, 5]),
    ],
    
    'calibration': [
        # Continuous calibration metrics - slow-quality regime indicators
        # Change slowly; want to know if calibration has been good or deteriorating over weeks
        (['calibration_score', 'brier_score', 'log_score', 'pit_uniformity', 
          'coverage_ratio', 'interval_width', 'sharpness', 'resolution'], [5, 20]),
        
        # Calibration quality indicators - flags reflect persistent miscalibration, weekly lag enough
        (['underconfident_flag', 'overconfident_flag', 'bias_positive', 'bias_negative'], [5]),
        
        # Metadata - state/ID fields get no lags (except sample_size which can have [20] for long-term tracking)
        (['sample_size'], [20]),
        (['requires_recalibration', 'calibration_version'], []),  # No lags for state/ID fields
    ],
    
    'online_learning': [
        # Original performance metrics - track if performance consistently better/worse and drift frequency
        (['online_learning_mae', 'online_learning_rmse', 'online_learning_mape', 
          'online_learning_direction_accuracy', 'online_learning_samples'], [1, 5]),
        # Counters - drift events, updates, retrains
        (['online_learning_drift_events', 'online_learning_incremental_updates', 
          'online_learning_partial_retrains'], [1, 5]),
        
        # Drift & update flags - detect if drift is currently on and persisted from yesterday
        (['online_learning_drift_flag', 'online_learning_partial_retrain_flag', 
          'online_learning_model_updated_flag'], [1]),
        
        # MAX ALPHA features - "structure of uncertainty" signals
        # Quantile spreads (e.g., q90_q10_spread, q75_q25_spread)
        (['_spread', 'q90_q10', 'q75_q25', 'q95_q05'], [1, 3, 5]),
        # Asymmetry indicators (upper_tail, lower_tail, skew from quantiles)
        (['asymmetry', '_tail_', 'upper_tail', 'lower_tail', 'quantile_skew'], [1, 3, 5]),
        # Distribution moments from quantiles
        (['quantile_mean', 'quantile_std', 'quantile_kurtosis'], [1, 3, 5]),
        # Calibration-enhanced confidence
        (['calibration_enhanced', 'confidence_score', 'adjusted_confidence'], [1, 3, 5]),
        # Adaptive learning rate signals
        (['learning_rate', 'adaptation_speed', 'update_magnitude'], [1, 3, 5]),
        # Regime-aware interval features
        (['regime_aware', 'interval_width', 'coverage_rate', 'sharpness'], [1, 3, 5]),
        # Prediction error patterns
        (['prediction_error', 'forecast_bias', 'residual_'], [1, 3, 5]),
    ],
    
    'arima_forecast': [
        # a) Core ARIMA outputs - must-have lags to detect if ARIMA consistently bullish/bearish and residuals blowing out
        # Point forecast for horizon days
        (['arima_forecast_'], [1, 3]),
        # Residuals and uncertainty
        (['arima_residuals', 'arima_residual_std'], [1, 3]),
        # Confidence intervals
        (['arima_confidence_upper', 'arima_confidence_lower'], [1, 3]),
        
        # b) Model diagnostics/selection - slow-moving, medium lag captures drift if needed
        # Fitted values and coefficients
        (['arima_fitted_values', 'arima_ma_coef', 'arima_ar_coef'], [5]),
        # Model selection criteria
        (['arima_aic', 'arima_bic'], [5]),
    ],
    
    'tft_features': [
        # TFT already encodes long history; minimal lags to avoid "double-loop" of temporal structure
        # All TFT features get [1] lag only - one-step memory of TFT's own state
        # This includes: multi-scale trend, volatility regime, seasonality, stability, confidence
        (['trend_short_strength', 'trend_long_strength', 'trend_consistency', 
          'trend_signal_to_noise', 'trend_importance'], [1]),
        (['vol_short', 'vol_long', 'vol_regime_zscore', 'vol_trend', 'volatility_importance'], [1]),
        (['weekday_effect', 'month_phase', 'seasonality_importance'], [1]),
        (['stability_short', 'stability_long', 'stability_score'], [1]),
        (['CONFIDENCE'], [1]),
        # Optional [5] lag for long-horizon features (commented out for v1 - keep lean)
        # (['trend_long_strength', 'vol_long', 'stability_long'], [5]),
    ],
    
    'multiasset': [
        # ========================================================================
        # STRUCTURAL EQUITY EXPOSURE LAG STRATEGY: MEDIUM-TERM DRIFT
        # ========================================================================
        # 120-day structural exposures move SLOWLY - need medium lags for drift
        # 12 features total → 33 features with lags (2.75x expansion)
        
        # === A. US EQUITY BENCHMARKS (7 features) → [10, 20] ===
        # 120-day correlations/betas change slowly; 10-20 day lags show structural change
        # SPY: corr, beta, spread (3)
        # QQQ: corr, beta (2)
        # IWM: corr, beta (2)
        (['multiasset_corr_spy_120', 'multiasset_beta_spy_120', 'multiasset_spread_spy_20',
          'multiasset_corr_qqq_120', 'multiasset_beta_qqq_120',
          'multiasset_corr_iwm_120', 'multiasset_beta_iwm_120'], [10, 20]),
        
        # === B. GLOBAL EQUITY FACTOR (2 features) → [10, 20] ===
        # ACWI: corr, beta (global vs US exposure drift)
        (['multiasset_corr_acwi_120', 'multiasset_beta_acwi_120'], [10, 20]),
        
        # === C. EQUITY STYLE FACTORS (3 features) → [20] ===
        # PCA factors already summarize structure; one long lag sufficient
        # Market, growth/value, size factors
        (['multiasset_equity_factor_market', 
          'multiasset_equity_factor_growth_value',
          'multiasset_equity_factor_size'], [20]),
    ],
    
    'regime': [
        # ========================================================================
        # STOCK-LEVEL TREND REGIME LAG STRATEGY: FAST BUT NOT INTRADAY
        # ========================================================================
        # Fairly fast vs macro, but still not intraday noise
        # 9 features total → 20 features with lags (2.22x expansion)
        
        # === A. REGIME PROBABILITIES (3 features) → [1, 5] ===
        # Want to know if prob persistently skewed to bull/bear for a week
        (['regime_bull_probability', 'regime_bear_probability', 
          'regime_neutral_probability'], [1, 5]),
        
        # === D. TREND METRICS (3 features) → [1, 5] ===
        # Short + 1-week lag captures "trend increasing/decreasing" and vol regime shifts
        (['regime_trend_ratio', 'regime_trend_ratio_zscore', 
          'regime_volatility_20d'], [1, 5]),
        
        # === C. TEMPORAL CONTEXT - CHANGE FLAG (1 feature) → [1] ===
        # Did we just flip regime recently?
        (['regime_change_flag'], [1]),
        
        # === B. REGIME LABEL (1 feature) → NO LAGS ===
        # === C. TEMPORAL CONTEXT - DURATION (1 feature) → NO LAGS ===
        # Categorical (0/1/2) lags don't add much
        # Duration is already cumulative (time aggregate)
        (['regime_label', 'regime_duration'], []),
    ],
    
    'cross_asset': [
        # ========================================================================
        # TACTICAL MACRO SHOCK PROPAGATION LAG STRATEGY: FAST + SLOW MIX
        # ========================================================================
        # 20-60d tactical macro land: [1,5] for fast dynamics, [5,20] for structural
        # 18 features total → 48 features with lags (2.67x expansion)
        
        # === FAST TACTICAL DYNAMICS → [1, 5] ===
        # A. ETF correlations (3): shock-sensitive, short + 1-week persistence
        # B. Beta dynamics (3): beta level, change rate, volatility
        # C. VIX relationships (3): core risk-on/off
        # D. Lead/lag (2): leadership stability over days
        # H. Composite risk factor (1): tactical risk barometer
        (['spy_corr_20d', 'qqq_corr_20d', 'sector_etf_corr_20d',
          'beta_20d', 'beta_change_rate', 'beta_volatility_20d',
          'vix_corr_20d', 'vix_spread_indicator', 'realized_vol_vs_spy_corr',
          'spy_leads_stock_5d', 'stock_leads_spy_5d',
          'risk_onoff_factor'], [1, 5]),
        
        # === BINARY REGIME SHIFTS → [1] ===
        # B. Beta sign flip flag: binary regime shift indicator
        (['beta_sign_flip_flag'], [1]),
        
        # === SLOW MACRO/CREDIT/FX DRIFT → [5, 20] ===
        # E. Rate sensitivity (2): duration/short-rate relationships drift slowly
        # F. Credit spreads (2): credit regimes are slow
        # G. FX risk-on/off (1): macro/FX tilt evolves slowly
        (['tnx_corr_20d', 'irx_corr_20d',
          'credit_spread_level', 'asset_corr_hyg_60',
          'asset_corr_uup_60'], [5, 20]),
    ],
    
    'options_anchoring': [
        # === ANCHORED TEMPORAL CONTEXT - LAGS ARE VALUABLE ===
        # Extremes often persist a few days around events: [1, 3]
        
        # === IV ANCHORING (3 features) ===
        (['iv_anchor_pct', 'iv_percentile_30d', 'iv_percentile_1yr'], [1, 3]),
        
        # === SKEW ANCHORING (3 features) ===
        # Skew extremes often persist a few days around events
        (['iv_skew_anchor', 'iv_skew_zscore', 'risk_reversal_25d'], [1, 3]),
        
        # === EXPECTED MOVE ANCHORING (2 features) ===
        (['expected_move_pct', 'em_vs_real_vol_ratio'], [1, 3]),
        
        # === POSITIONING/SENTIMENT ANCHORING (2 features) ===
        # Relatively smooth; [1,3] is enough to see sustained put-heavy hedging
        (['put_call_vol_ratio_anchor', 'put_call_oi_ratio_anchor'], [1, 3]),
    ],
    
    'macro_tst_hf': [
        # ========================================================================
        # INSTITUTIONAL-GRADE MACRO LAG STRATEGY: SLOW-DRIFT WEEKLY/MONTHLY
        # ========================================================================
        # Macro is SLOW - care about weekly/monthly drift, not day-to-day noise
        # 56 features total → 144 features with lags (2.57x expansion)
        
        # === A. FUNDAMENTAL ECONOMIC INDICATORS (8 features) → [20, 60] ===
        # Quarterly/monthly/slow rates - 1 or 5 days would be near-duplicates
        # GDP, unemployment, inflation (CPI/PCE), Fed funds, 10Y/2Y yields, curve slope
        (['macro_gdp_growth', 'macro_unemployment_rate', 
          'macro_inflation_cpi', 'macro_inflation_pce',
          'macro_fed_funds_rate', 'macro_treasury_10y', 'macro_treasury_2y',
          'macro_yield_curve_slope'], [20, 60]),
        
        # === C. STRUCTURAL & FLOW INDICATORS (11 features) → [20, 60] ===
        # Slow structural series - medium/long drift matters
        # Labor, capacity, retail sales, industrial production, sentiment, PMIs,
        # trade, profits, real yields, mortgage rates
        (['macro_labor_participation', 'macro_capacity_utilization',
          'macro_retail_sales_growth', 'macro_industrial_production',
          'macro_consumer_sentiment', 'macro_pmi_manufacturing', 'macro_pmi_services',
          'macro_trade_balance', 'macro_corporate_profits',
          'macro_real_yields', 'macro_mortgage_rates'], [20, 60]),
        
        # === B. MARKET-BASED INDICATORS (5 features) → [5, 20] ===
        # Daily-ish series but macro-flavored - weekly + ~1-month drift sufficient
        # VIX, credit spreads, dollar, oil, housing
        (['macro_vix', 'macro_credit_spread', 'macro_dollar_index',
          'macro_oil_price', 'macro_housing_starts'], [5, 20]),
        
        # === D. DERIVED MACRO SIGNALS (10 features) → [5, 20] ===
        # Already "deltas/indices" - weekly & monthly lags capture cycle changes
        # Growth/inflation momentum, policy stance, curve steepness, credit cycle,
        # liquidity, fiscal, external balance, financial conditions, recession risk
        (['macro_growth_momentum', 'macro_inflation_momentum',
          'macro_policy_stance', 'macro_yield_curve_steepness',
          'macro_credit_cycle', 'macro_liquidity_conditions',
          'macro_fiscal_impulse', 'macro_external_balance',
          'macro_financial_conditions', 'macro_recession_risk'], [5, 20]),
        
        # === H. FALLBACK PROXIES (8 features) → [5, 20] ===
        # Treat like their "real" counterparts (market-based/derived)
        (['macro_fallback_vix_proxy', 'macro_fallback_credit_proxy',
          'macro_fallback_dollar_proxy', 'macro_fallback_oil_proxy',
          'macro_fallback_sentiment_proxy', 'macro_fallback_pmi_proxy',
          'macro_fallback_housing_proxy', 'macro_fallback_inflation_proxy'], [5, 20]),
        
        # === E. HF TIMESERIESTRANSFORMER EMBEDDINGS (8 features) → [5] ===
        # Already encode 30-day lookback - only need light sense of change
        # One 1-week lag shows "macro state moved in embedding space"
        (['macro_tst_embedding_'], [5]),
        
        # === F. HF INFERENCE SIGNALS (4 features) → [5] ===
        # Care about forecast drifting over weeks, not intraday blips
        # GDP forecast, inflation forecast, recession prob, regime cluster
        (['macro_tst_forecast_gdp', 'macro_tst_forecast_inflation',
          'macro_tst_forecast_recession', 'macro_tst_regime_cluster'], [5]),
        
        # === G. LEGACY/FALLBACK INDICATORS (2 features) → [1] ===
        # Binary-ish regime flags - mainly want persistence ("still in crisis/expansion?")
        (['macro_crisis_indicator', 'macro_expansion_phase'], [1]),
    ],
    
    'dcf': [
        # === DCF SIGNAL/DELTA FEATURES (4 features) → [5, 20, 60] ===
        # Signal-like: dcf_discount_pct, dcf_premium_pct, dcf_reversion_signal, dcf_momentum
        # Moves with price + model updates - want short, medium, and quarterly lags
        (['dcf_discount_pct', 'dcf_premium_pct', 'dcf_reversion_signal', 'dcf_momentum',
          'discount_', 'premium_', 'reversion_', 'momentum_'], [5, 20, 60]),
        
        # === DCF STATE FEATURES (4 features) → [20, 60] ===
        # State-like: dcf_fair_value, dcf_zscore, dcf_percentile, dcf_confidence
        # Valuation layer positioning - medium and quarterly lags
        (['dcf_fair_value', 'dcf_zscore', 'dcf_percentile', 'dcf_confidence',
          'fair_value', 'zscore', 'percentile'], [20, 60]),
    ],
    
    'finbert': [
        # FinBERT sentiment score - main signal, persistence matters over 1-5 days
        (['finbert_sentiment_score'], [1, 3, 5]),
        # Confidence - persistence over 1-3 days (are we consistently confident?)
        (['finbert_sentiment_confidence'], [1, 3]),
        # Positive/negative components - optional, just 1-day lag to see flips
        (['finbert_sentiment_positive', 'finbert_sentiment_negative'], [1]),
    ],
    
    'earnings_transcript_hf': [
        # Overall sentiment score and confidence - quarterly events with multi-day decay
        (['score', 'conf'], [1, 5]),
        # Prepared vs Q&A split - detect tone shift between sections
        (['score_prepared', 'score_qa'], [1, 5]),
        # Uncertainty and risk scores - track persistence of cautious language
        (['uncertainty_score', 'risk_score'], [1, 5]),
        # QoQ/YoY deltas - already historical comparisons, NO lags needed
        (['score_delta_qoq', 'score_delta_yoy'], []),
    ],
    
    'doc_embedding_novelty_hf': [
        # Main novelty signal - regime changes unfold over days/weeks
        (['score', 'conf'], [1, 5, 20]),
        # Event/article volume - detect sustained coverage vs one-off spikes
        (['n_events', 'n_articles'], [1, 5]),
        # Cluster novelties - sector-specific regime shifts persist
        (['macro_novelty', 'geopolitical_novelty', 'regulatory_novelty',
          'energy_novelty', 'conflict_novelty', 'tech_novelty'], [1, 5, 20]),
        # Theme metadata - current state, not historical
        (['theme_weight', 'baseline_mean'], []),
        # Spike and persistence flags - already capture temporal patterns
        (['novelty_spike_flag', 'novelty_persistence_5d'], []),
        # Top theme numeric proxy
        (['top_theme_numeric'], [1, 5]),
    ],
    
    # ========================================================================
    # FUNDAMENTAL FAMILIES: QUARTERLY BALANCE SHEET + INCOME STATEMENT
    # ========================================================================
    # Balance sheet items update quarterly - lags at ~60d (prev quarter) and 252d (prev year)
    # Price-based ratios move daily - mix of short (5d), medium (20d), and quarterly (60d) lags
    
    # === FIN_G2: Leverage & Solvency (4 features) ===
    # Slow structural balance sheet risk
    # Lags: [60, 252] = previous quarter + previous year state
    'fin_g2': [
        (['debt_to_equity', 'debt_to_assets', 'interest_coverage', 'equity_multiplier'], [60, 252]),
    ],
    
    # === FIN_G3: Efficiency & Activity (5 features) ===
    # Turnover ratios change, but still at reporting cadence
    # Lags: [60, 252] = previous quarter + previous year state
    'fin_g3': [
        (['asset_turnover', 'inventory_turnover', 'receivables_turnover', 
          'payables_turnover', 'working_capital_turnover'], [60, 252]),
    ],
    
    # === FIN_G4: Cash Flow Metrics (6 features) ===
    # Cash flow & FCF structure = slow, but very informative
    # Lags: [60, 252] = previous quarter + previous year state
    'fin_g4': [
        (['operating_cash_flow', 'free_cash_flow', 'fcf_to_revenue', 
          'fcf_to_net_income', 'cash_conversion_cycle', 'capex_to_revenue'], [60, 252]),
    ],
    
    # === FIN_G5: Valuation Ratios (8 features) ===
    # From aggregator_panel.py _h_fin_g6: pe_ratio, pb_ratio, ps_ratio, pcf_ratio,
    # peg_ratio, ev_to_ebitda, ev_to_revenue, price_to_fcf
    # These involve price, so they move daily. Want both short + medium lags.
    # Lags: [5, 20, 60] = 1 week, 1 month, 1 quarter
    'fin_g5': [
        (['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
          'peg_ratio', 'ev_to_ebitda', 'ev_to_revenue', 'price_to_fcf'], [5, 20, 60]),
    ],
    
    # === FIN_G6: Growth Metrics (6 features) ===
    # From aggregator_panel.py _h_fin_g5: revenue_growth_yoy, earnings_growth_yoy,
    # eps_growth_quarterly, revenue_growth_quarterly, book_value_growth_yoy, tangible_book_growth_yoy
    # YoY & quarterly growth: mostly per-report changes
    # Lags: [60, 252] = previous quarter + previous year growth context
    'fin_g6': [
        (['revenue_growth_yoy', 'earnings_growth_yoy',
          'eps_growth_quarterly', 'revenue_growth_quarterly',
          'book_value_growth_yoy', 'tangible_book_growth_yoy'], [60, 252]),
    ],
    
    # === FIN_G7: Dividend & Shareholder Metrics (5 features) ===
    # From aggregator_panel.py _h_fin_g7: dividend_yield, dividend_payout_ratio (payout_ratio),
    # buyback_yield, total_shareholder_yield, shares_outstanding_change
    # Mixed: some price-based (yield), some slow (payout, buybacks)
    'fin_g7': [
        # Price-influenced: yield and total shareholder yield
        (['dividend_yield', 'total_shareholder_yield'], [5, 20, 60]),
        # Structural ratios: payout, buyback, share count changes
        (['dividend_payout_ratio', 'payout_ratio', 'buyback_yield', 'shares_outstanding_change'], [60, 252]),
    ],
}


# ============================================================================
# LAG APPLICATION FUNCTIONS
# ============================================================================

def get_lag_config(family: str) -> List[Tuple[List[str], List[int]]]:
    """
    Get lag configuration for a specific family.
    
    Args:
        family: Family name (e.g., 'ml_framework', 'microstructure')
    
    Returns:
        List of (patterns, lags) tuples
    """
    return LAG_CONFIG.get(family, [])


def apply_lags(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """
    Apply lags to a DataFrame based on family configuration.
    
    DISABLED: Returns original DataFrame without any lags for performance.
    
    Args:
        df: Input DataFrame with feature columns
        family: Family name to look up lag config
    
    Returns:
        DataFrame with original features only (no lags)
    """
    # LAGS DISABLED - return original DataFrame immediately
    logger.debug(f"Lags disabled for {family}, returning raw features only")
    return df
    
    # Original lag code disabled below
    if df is None or df.empty:
        return df
    
    lag_rules = get_lag_config(family)
    if not lag_rules:
        logger.debug(f"No lag config for family '{family}', returning original DataFrame")
        return df
    
    lagged_frames = [df]
    total_lags_added = 0
    existing_columns = set(df.columns)

    # Build a set of "clean" column names (family prefix removed) so we can
    # distinguish semantic *_lag{N} columns from derived lag columns.
    # If a column ends with _lagN AND the corresponding base name exists,
    # we treat it as a derived lag and never lag it again.
    clean_names: List[str] = []
    for col in df.columns:
        col_lower = str(col).lower()
        clean_names.append(col_lower.replace(f"{family.lower()}_", ""))
    clean_name_set = set(clean_names)
    
    # Track (column, lag) pairs to avoid duplicates across pattern groups
    lagged_pairs = set()
    
    for patterns, lags in lag_rules:
        if not lags:  # Empty lag list = no lags for these patterns
            continue
        
        # Find matching columns
        matching_cols = []
        for col in df.columns:
            col_lower = col.lower()
            # Remove family prefix for pattern matching
            col_clean = col_lower.replace(f"{family.lower()}_", "")

            # Guard: never create lags-of-lags. Only skip *_lag{N} columns when they
            # look like a derived lag of another feature that exists in this frame.
            m = re.search(r"^(.*)_lag(\d+)$", col_clean)
            if m:
                base_name = m.group(1)
                if base_name in clean_name_set:
                    continue

            if any(pattern.lower() in col_clean for pattern in patterns):
                matching_cols.append(col)
        
        if not matching_cols:
            continue
        
        # Apply each lag
        for lag in lags:
            # Only lag columns that haven't been lagged at this lag already
            cols_to_lag = [col for col in matching_cols if (col, lag) not in lagged_pairs]
            if not cols_to_lag:
                continue
            
            # Deduplicate preserving order
            seen = set()
            unique_cols = []
            for col in cols_to_lag:
                if col not in seen:
                    seen.add(col)
                    unique_cols.append(col)
            
            if not unique_cols:
                continue

            # Avoid creating duplicate column names (e.g., base already contains *_lag20)
            keep_cols: List[str] = []
            keep_names: List[str] = []
            for col in unique_cols:
                new_name = f"{col}_lag{lag}"
                if new_name in existing_columns:
                    continue
                keep_cols.append(col)
                keep_names.append(new_name)

            if not keep_cols:
                continue

            lagged = df[keep_cols].shift(lag)
            lagged.columns = keep_names
            lagged_frames.append(lagged)
            total_lags_added += len(keep_cols)
            existing_columns.update(keep_names)
            
            # Mark these (column, lag) pairs as processed
            for col in unique_cols:
                lagged_pairs.add((col, lag))
    
    if len(lagged_frames) > 1:
      result = pd.concat(lagged_frames, axis=1)
      # Parquet writers reject duplicate column labels.
      if result.columns.duplicated().any():
        result = result.loc[:, ~result.columns.duplicated(keep="last")]
      logger.info(
        f"Applied lags to {family}: {len(df.columns)} base features → "
        f"{len(result.columns)} total ({total_lags_added} lagged features added)"
      )
      return result

    return df


def get_all_families_with_lags() -> List[str]:
    """Return list of all families that have lag configurations."""
    return list(LAG_CONFIG.keys())


def get_lag_summary(family: str) -> Dict[str, int]:
    """
    Get summary statistics for a family's lag configuration.
    
    Returns:
        Dict with keys: 'pattern_groups', 'max_lag', 'min_lag', 'total_lag_steps'
    """
    lag_rules = get_lag_config(family)
    if not lag_rules:
        return {
            'pattern_groups': 0,
            'max_lag': 0,
            'min_lag': 0,
            'total_lag_steps': 0,
        }
    
    all_lags = []
    for _, lags in lag_rules:
        all_lags.extend(lags)
    
    return {
        'pattern_groups': len(lag_rules),
        'max_lag': max(all_lags) if all_lags else 0,
        'min_lag': min(all_lags) if all_lags else 0,
        'total_lag_steps': len(set(all_lags)),
    }


__all__ = [
    'LAG_CONFIG',
    'get_lag_config',
    'apply_lags',
    'get_all_families_with_lags',
    'get_lag_summary',
]
