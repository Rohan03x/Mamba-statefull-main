# Feature families and features (from parquet schemas)

Generated at `2026-01-12T14:28:26Z`.

Artifacts scanned:

- `cache/features/AAPL_h63_merged.parquet`
  - meta: `cache/features/AAPL_h63_merged.meta.json`
  - provenance: `cache/features/AAPL_h63_merged.provenance.json`
- `cache/features/AAPL_h63_feast_phase2_base_v1.parquet`
- `cache/features/AAPL_h63_feast_phase2_full_v1.parquet`

## Full per-family feature caches (AAPL h=63)

Source: `data/local_cache/<symbol>_h<horizon>/*_train_features.parquet`

Families: **38**
Total columns (sum of per-family schemas): **714**

### <base>

- date

### alternative_signals (45)

- alternative_signals_beta_20d
- alternative_signals_beta_change_rate
- alternative_signals_beta_vix_interaction
- alternative_signals_buy_volume_proxy
- alternative_signals_close_to_close_volatility
- alternative_signals_closing_ramp
- alternative_signals_conf
- alternative_signals_days_since_last_earnings
- alternative_signals_days_to_next_earnings
- alternative_signals_earnings_runup_10d
- alternative_signals_gap_down_pct
- alternative_signals_gap_up_pct
- alternative_signals_gap_vs_vix_interaction
- alternative_signals_google_trends_score
- alternative_signals_high_low_volatility_ratio
- alternative_signals_intraday_range_pct
- alternative_signals_intraday_range_z
- alternative_signals_intraday_volatility_ratio
- alternative_signals_liquidity_stress_pct
- alternative_signals_mean_reversion_signal
- alternative_signals_news_volume_change
- alternative_signals_news_volume_count
- alternative_signals_news_volume_z
- alternative_signals_open_to_close_volatility
- alternative_signals_opening_reversal
- alternative_signals_opening_volume_surge
- alternative_signals_overnight_return
- alternative_signals_overnight_return_z
- alternative_signals_post_earnings_drift_5d
- alternative_signals_qqq_correlation_20d
- alternative_signals_relative_volume_20d
- alternative_signals_rv_10d
- alternative_signals_rv_20d
- alternative_signals_rv_5d
- alternative_signals_rv_ratio_5_20
- alternative_signals_rv_z_20
- alternative_signals_score
- alternative_signals_score_raw
- alternative_signals_sector_etf_correlation_20d
- alternative_signals_spy_correlation_20d
- alternative_signals_trend_acceleration
- alternative_signals_turn_of_month_flag
- alternative_signals_volume_price_divergence
- alternative_signals_volume_trend_10d
- alternative_signals_volume_z_20d

### arima_forecast (18)

- arima_forecast_1d
- arima_forecast_5d
- arima_forecast_ar_accel
- arima_forecast_ar_level
- arima_forecast_ar_resid_vol
- arima_forecast_ar_trend
- arima_forecast_arima_abs_residual
- arima_forecast_arima_innovation
- arima_forecast_arima_log_likelihood
- arima_forecast_arima_momentum_indicator
- arima_forecast_arima_persistence
- arima_forecast_arima_residual_t
- arima_forecast_arima_residual_zscore
- arima_forecast_arima_uncertainty_proxy
- arima_forecast_conf
- arima_forecast_confidence
- arima_forecast_score
- arima_forecast_score_raw

### calibration (24)

- calibration_confidence
- calibration_interval_coverage
- calibration_interval_error
- calibration_interval_expected
- calibration_mean_calibration_error
- calibration_overall_score
- calibration_q05_coverage
- calibration_q05_error
- calibration_q10_coverage
- calibration_q10_error
- calibration_q25_coverage
- calibration_q25_error
- calibration_q50_coverage
- calibration_q50_error
- calibration_q75_coverage
- calibration_q75_error
- calibration_q90_coverage
- calibration_q90_error
- calibration_q95_coverage
- calibration_q95_error
- calibration_q99_coverage
- calibration_q99_error
- calibration_requires_recalibration
- calibration_sample_size

### cboe_term (19)

- cboe_term_conf
- cboe_term_front_back_spread
- cboe_term_normalized_term_slope
- cboe_term_panic_premium
- cboe_term_panic_premium_change
- cboe_term_score
- cboe_term_score_raw
- cboe_term_vix_contango_strength
- cboe_term_vix_curvature_change
- cboe_term_vix_ratio_term
- cboe_term_vix_roll_yield
- cboe_term_vix_term_curvature
- cboe_term_vix_term_slope_change_1d
- cboe_term_vix_term_slope_change_5d
- cboe_term_vix_vxmt_term_slope
- cboe_term_vix_vxv_term_slope
- cboe_term_vol_risk_premium
- cboe_term_vol_risk_premium_pct
- cboe_term_vxst_vix_term_slope

### correlation (31)

- correlation_acf_absret_1
- correlation_acf_ret_1
- correlation_acf_ret_5
- correlation_acf_vol_1
- correlation_conf
- correlation_confidence
- correlation_corr_20_qqq
- correlation_corr_20_qqq_trend
- correlation_corr_20_sectorxlk
- correlation_corr_20_spy
- correlation_corr_20_spy_trend
- correlation_corr_20_spy_vol
- correlation_corr_20_vix_lag1
- correlation_corr_20_vix_lag2
- correlation_corr_20_vxx
- correlation_corr_20_vxx_trend
- correlation_corr_20_vxx_vol
- correlation_corr_60_spy
- correlation_corr_60_vix_lag1
- correlation_corr_60_vix_lag2
- correlation_corr_decoupling_z
- correlation_corr_return_range_10
- correlation_corr_return_vol_20
- correlation_corr_spread_20_60_spy
- correlation_corr_vol_volatility_20
- correlation_lag_corr_1_spy
- correlation_lag_corr_1_vxx
- correlation_lag_corr_2_vxx
- correlation_lag_corr_5_spy
- correlation_score
- correlation_score_raw

### cross_asset (25)

- cross_asset_asset_corr_hyg_60
- cross_asset_asset_corr_uup_60
- cross_asset_beta_20d
- cross_asset_beta_change_rate
- cross_asset_beta_sign_flip_flag
- cross_asset_beta_volatility_20d
- cross_asset_beta_volatility_change
- cross_asset_conf
- cross_asset_coupling_change
- cross_asset_credit_spread_level
- cross_asset_irx_corr_20d
- cross_asset_irx_corr_change_5d
- cross_asset_qqq_corr_20d
- cross_asset_realized_vol_vs_spy_corr
- cross_asset_risk_onoff_factor
- cross_asset_score
- cross_asset_score_raw
- cross_asset_sector_etf_corr_20d
- cross_asset_spy_corr_20d
- cross_asset_spy_leads_stock_5d
- cross_asset_stock_leads_spy_5d
- cross_asset_tnx_corr_20d
- cross_asset_tnx_corr_change_5d
- cross_asset_vix_corr_20d
- cross_asset_vix_spread_indicator

### dcf (19)

- dcf_confidence
- dcf_mom_12m
- dcf_mom_1m
- dcf_mom_3m
- dcf_mom_sharped
- dcf_mom_vol_adjusted
- dcf_price_regime
- dcf_price_to_fairvalue_1y
- dcf_price_to_fairvalue_3m
- dcf_scenario_position
- dcf_scenario_skew
- dcf_scenario_spread
- dcf_terminal_value_pct
- dcf_trend_slope_1m
- dcf_trend_stability
- dcf_value_momentum_ratio
- dcf_vol_adjusted_value
- dcf_zscore_1m
- dcf_zscore_3m

### dividends (12)

- dividends_conf
- dividends_confidence
- dividends_days_to_ex_dividend
- dividends_dividend_amount
- dividends_dividend_event_intensity
- dividends_dividend_frequency
- dividends_dividend_yield_est
- dividends_dividend_yield_zscore
- dividends_ex_dividend_window_strength
- dividends_has_data
- dividends_score
- dividends_score_raw

### doc_embedding_novelty_hf (16)

- baseline_mean
- conflict_novelty
- doc_embedding_novelty_hf_conf
- doc_embedding_novelty_hf_score
- doc_embedding_novelty_hf_score_raw
- energy_novelty
- geopolitical_novelty
- macro_novelty
- n_articles
- n_events
- novelty_persistence_5d
- novelty_spike_flag
- regulatory_novelty
- tech_novelty
- theme_weight
- top_theme_numeric

### earnings (14)

- earnings_beat_rate_3y
- earnings_beat_streak
- earnings_confidence
- earnings_days_since_earnings
- earnings_days_to_next_earnings
- earnings_eps_growth_qoq
- earnings_eps_growth_yoy
- earnings_eps_surprise_pct
- earnings_event_decay
- earnings_has_data
- earnings_miss_streak
- earnings_revision_breadth
- earnings_surprise_percent
- earnings_surprise_volatility

### earnings_transcript_hf (4)

- earnings_transcript_hf_conf
- earnings_transcript_hf_has_data
- earnings_transcript_hf_score
- earnings_transcript_hf_score_raw

### fin_g1 (9)

- fin_g1_cash_ratio
- fin_g1_conf
- fin_g1_current_ratio
- fin_g1_has_data
- fin_g1_liquidity_trend_3y
- fin_g1_liquidity_zscore_5y
- fin_g1_quick_ratio
- fin_g1_score
- fin_g1_score_raw

### fin_g2 (15)

- fin_g2_conf
- fin_g2_debt_to_assets
- fin_g2_debt_to_equity
- fin_g2_equity_multiplier
- fin_g2_has_data
- fin_g2_interest_burden
- fin_g2_interest_coverage
- fin_g2_leverage_trend_3y
- fin_g2_leverage_zscore_5y
- fin_g2_long_term_debt
- fin_g2_net_debt_to_ebitda
- fin_g2_net_debt_to_fcf
- fin_g2_score
- fin_g2_score_raw
- fin_g2_total_debt

### fin_g3 (12)

- fin_g3_asset_turnover
- fin_g3_conf
- fin_g3_dios
- fin_g3_dpos
- fin_g3_dsos
- fin_g3_has_data
- fin_g3_inventory_turnover
- fin_g3_payables_turnover
- fin_g3_receivables_turnover
- fin_g3_score
- fin_g3_score_raw
- fin_g3_turnover_volatility_3y

### fin_g4 (12)

- fin_g4_accruals_ratio
- fin_g4_capex_to_revenue
- fin_g4_cfo_to_net_income
- fin_g4_conf
- fin_g4_fcf_margin
- fin_g4_fcf_to_net_income
- fin_g4_fcf_to_revenue
- fin_g4_free_cash_flow
- fin_g4_has_data
- fin_g4_operating_cash_flow
- fin_g4_score
- fin_g4_score_raw

### fin_g5 (11)

- fin_g5_cf_growth_yoy
- fin_g5_conf
- fin_g5_ebitda_growth_yoy
- fin_g5_eps_cagr_3y
- fin_g5_growth_volatility_3y
- fin_g5_has_data
- fin_g5_margin_expansion
- fin_g5_revenue_cagr_3y
- fin_g5_revenue_growth_yoy
- fin_g5_score
- fin_g5_score_raw

### fin_g6 (12)

- fin_g6_conf
- fin_g6_dividend_yield
- fin_g6_ev_ebitda
- fin_g6_ev_ebitda_zscore_5y
- fin_g6_has_data
- fin_g6_pb_ratio
- fin_g6_pb_ratio_zscore_5y
- fin_g6_pe_ratio
- fin_g6_pe_ratio_zscore_5y
- fin_g6_ps_ratio
- fin_g6_score
- fin_g6_score_raw

### fin_g7 (10)

- fin_g7_conf
- fin_g7_dividend_policy_stability
- fin_g7_dividend_yield_proxy
- fin_g7_has_data
- fin_g7_payout_ratio
- fin_g7_score
- fin_g7_score_raw
- fin_g7_share_dilution_3y
- fin_g7_shares_outstanding
- fin_g7_yield_zscore_5y

### finbert (6)

- finbert_conf
- finbert_confidence
- finbert_has_data
- finbert_neutral
- finbert_score
- finbert_score_raw

### forecast_hf (3)

- forecast_hf_conf
- forecast_hf_score
- forecast_hf_score_raw

### fundamental_val_hf (3)

- fundamental_val_hf_conf
- fundamental_val_hf_score
- fundamental_val_hf_score_raw

### garch_iv (21)

- garch_iv_conf
- garch_iv_garch30_minus_garch180
- garch_iv_garch_1d
- garch_iv_garch_20d
- garch_iv_garch_5d
- garch_iv_garch_log_likelihood
- garch_iv_garch_long_run_variance
- garch_iv_garch_persistence
- garch_iv_garch_ratio_1d_20d
- garch_iv_garch_residual_vol
- garch_iv_garch_shock_indicator
- garch_iv_garch_short_long_ratio
- garch_iv_garch_spike_flag
- garch_iv_garch_standardized_residual
- garch_iv_garch_vol_momentum
- garch_iv_garch_vol_norm_20d
- garch_iv_garch_vol_of_vol
- garch_iv_garch_zscore
- garch_iv_score
- garch_iv_score_raw
- garch_iv_skew_proxy_downside_minus_upside

### macro_regime_hf (3)

- macro_regime_hf_conf
- macro_regime_hf_score
- macro_regime_hf_score_raw

### macro_tst_hf (41)

- derived_credit_spread
- derived_credit_spread_change
- derived_credit_spread_z
- derived_curve_slope_change
- derived_derived_debt_to_gdp_change
- derived_derived_gdp_growth_accel
- derived_derived_inflation_accel
- derived_derived_real_rate_change
- derived_derived_trade_balance_change
- derived_derived_unemployment_change
- derived_gold_change
- derived_interact_curve_bank
- derived_interact_oil_energy
- derived_interact_rate_growth
- derived_interact_vix_beta
- derived_oil_change
- derived_rates_2y_change_1d
- derived_rates_2y_change_5d
- derived_vix_return_1d
- derived_vix_return_5d
- derived_vix_spike_flag
- derived_yield_curve_slope
- hf_regime
- hf_volatility
- l1_gdp_growth_annual
- l1_inflation_cpi_annual
- l1_unemployment_rate
- l2_govt_debt_pct_gdp
- l2_net_trade_balance
- l3_gold
- l3_hyg_credit
- l3_irx_3mo
- l3_lqd_ig
- l3_oil
- l3_real_interest_rate
- l3_tnx_10y
- l3_tnx_2y
- l3_vix
- macro_tst_hf_conf
- macro_tst_hf_score
- macro_tst_hf_score_raw

### microstructure (35)

- microstructure_conf
- microstructure_confidence
- microstructure_micro_amihud
- microstructure_micro_atr_ratio
- microstructure_micro_body_pct
- microstructure_micro_demand_supply_ratio
- microstructure_micro_gap_direction
- microstructure_micro_hl_volume_corr
- microstructure_micro_impact_ratio
- microstructure_micro_impact_volatility
- microstructure_micro_intraday_vol_proxy
- microstructure_micro_intraday_vs_overnight_vol
- microstructure_micro_liquidity_imbalance
- microstructure_micro_low_liquidity_flag
- microstructure_micro_ofi_proxy
- microstructure_micro_overnight_gap
- microstructure_micro_overnight_vol
- microstructure_micro_pressure_proxy
- microstructure_micro_range_pct
- microstructure_micro_range_scaled
- microstructure_micro_shadow_ratio
- microstructure_micro_signed_volume
- microstructure_micro_spread_proxy
- microstructure_micro_stale_tick
- microstructure_micro_true_range
- microstructure_micro_turnover
- microstructure_micro_vol_of_vol
- microstructure_micro_volume_liquidity
- microstructure_micro_volume_surge
- microstructure_micro_volume_zscore
- microstructure_micro_wick_bottom
- microstructure_micro_wick_top
- microstructure_micro_zero_range_flag
- microstructure_score
- microstructure_score_raw

### ml_framework (81)

- ml_framework_autocorr_lag_1
- ml_framework_autocorr_lag_5
- ml_framework_bb_lower_20
- ml_framework_bb_lower_50
- ml_framework_bb_position_20
- ml_framework_bb_position_50
- ml_framework_bb_upper_20
- ml_framework_bb_upper_50
- ml_framework_bb_width_20
- ml_framework_bb_width_50
- ml_framework_conf
- ml_framework_confidence
- ml_framework_day_of_year
- ml_framework_ema_12
- ml_framework_ema_26
- ml_framework_ema_50
- ml_framework_is_quarter_end
- ml_framework_is_quarter_start
- ml_framework_is_year_end
- ml_framework_is_year_start
- ml_framework_kurtosis_20d
- ml_framework_kurtosis_50d
- ml_framework_log_return_1d
- ml_framework_log_return_5d
- ml_framework_ma_10
- ml_framework_ma_20
- ml_framework_ma_200
- ml_framework_ma_5
- ml_framework_ma_50
- ml_framework_macd
- ml_framework_macd_signal
- ml_framework_percentile_10_20d
- ml_framework_percentile_10_50d
- ml_framework_percentile_25_20d
- ml_framework_percentile_25_50d
- ml_framework_percentile_75_20d
- ml_framework_percentile_75_50d
- ml_framework_percentile_90_20d
- ml_framework_percentile_90_50d
- ml_framework_price_position_10d
- ml_framework_price_position_20d
- ml_framework_price_position_50d
- ml_framework_price_ratio_10d
- ml_framework_price_ratio_20d
- ml_framework_price_ratio_50d
- ml_framework_price_ratio_5d
- ml_framework_price_volume
- ml_framework_price_vs_ema_12
- ml_framework_price_vs_ema_26
- ml_framework_price_vs_ema_50
- ml_framework_price_vs_ma_10
- ml_framework_price_vs_ma_20
- ml_framework_price_vs_ma_200
- ml_framework_price_vs_ma_5
- ml_framework_price_vs_ma_50
- ml_framework_quarter
- ml_framework_return_10d
- ml_framework_return_1d
- ml_framework_return_20d
- ml_framework_return_5d
- ml_framework_rsi
- ml_framework_score
- ml_framework_score_raw
- ml_framework_skewness_20d
- ml_framework_skewness_50d
- ml_framework_volatility_10d
- ml_framework_volatility_20d
- ml_framework_volatility_50d
- ml_framework_volatility_5d
- ml_framework_volatility_annualized_10d
- ml_framework_volatility_annualized_20d
- ml_framework_volatility_annualized_50d
- ml_framework_volatility_annualized_5d
- ml_framework_volume_ma_20
- ml_framework_volume_ma_5
- ml_framework_volume_ma_50
- ml_framework_volume_ratio_20
- ml_framework_volume_ratio_5
- ml_framework_volume_ratio_50
- ml_framework_volume_weighted_price
- ml_framework_week_of_year

### multiasset (16)

- multiasset_beta_acwi_120
- multiasset_beta_iwm_120
- multiasset_beta_qqq_120
- multiasset_beta_spy_120
- multiasset_conf
- multiasset_confidence
- multiasset_corr_acwi_120
- multiasset_corr_iwm_120
- multiasset_corr_qqq_120
- multiasset_corr_spy_120
- multiasset_equity_factor_growth_value
- multiasset_equity_factor_market
- multiasset_equity_factor_size
- multiasset_score
- multiasset_score_raw
- multiasset_spread_spy_20

### news_nlp_hf (3)

- news_nlp_hf_conf
- news_nlp_hf_score
- news_nlp_hf_score_raw

### news_sentiment_hf (3)

- news_sentiment_hf_conf
- news_sentiment_hf_score
- news_sentiment_hf_score_raw

### online_learning (35)

- online_learning_asymmetric_calibration_ratio
- online_learning_conf
- online_learning_direction_accuracy
- online_learning_downside_calibration_error
- online_learning_drift_events
- online_learning_drift_flag
- online_learning_incremental_updates
- online_learning_mae
- online_learning_mape
- online_learning_model_confidence
- online_learning_model_updated_flag
- online_learning_partial_retrain_flag
- online_learning_partial_retrains
- online_learning_prediction_skewness
- online_learning_quantile_drift_avg
- online_learning_quantile_drift_q50
- online_learning_quantile_drift_std
- online_learning_quantile_spread_current
- online_learning_quantile_spread_shock
- online_learning_recalibration_needed
- online_learning_recalibration_strength
- online_learning_regime_trend_negative
- online_learning_regime_trend_neutral
- online_learning_regime_trend_positive
- online_learning_regime_vol_high
- online_learning_regime_vol_low
- online_learning_regime_vol_normal
- online_learning_rmse
- online_learning_samples
- online_learning_score
- online_learning_score_raw
- online_learning_trust_score
- online_learning_uncertainty_compression_alert
- online_learning_uncertainty_compression_ratio
- online_learning_upside_calibration_error

### options (23)

- options_atm_iv
- options_bid_ask_spread_pct
- options_call_ask_avg
- options_call_bid_avg
- options_call_iv_avg
- options_call_last_avg
- options_call_oi
- options_call_volume
- options_expiries_available
- options_has_data
- options_iv_spread
- options_nearest_expiry_days
- options_otm_call_pct
- options_otm_put_pct
- options_put_ask_avg
- options_put_bid_avg
- options_put_call_oi_ratio
- options_put_call_volume_ratio
- options_put_iv_avg
- options_put_last_avg
- options_put_oi
- options_put_volume
- options_strikes_available

### quantile_forecast (22)

- quantile_forecast_hf_conf
- quantile_forecast_hf_score
- quantile_forecast_q05
- quantile_forecast_q10
- quantile_forecast_q25
- quantile_forecast_q50
- quantile_forecast_q75
- quantile_forecast_q90
- quantile_forecast_q95
- quantile_forecast_q99
- quantile_forecast_q_high_75
- quantile_forecast_q_high_95
- quantile_forecast_q_low_25
- quantile_forecast_q_low_5
- quantile_forecast_q_median_50
- quantile_forecast_q_skewness_proxy
- quantile_forecast_q_spread_95_5
- quantile_forecast_q_tilt_direction
- quantile_forecast_q_vol_forecast
- quantile_forecast_skew
- quantile_forecast_uncertainty
- quantile_forecast_width

### regime (13)

- regime_bear_probability
- regime_bull_probability
- regime_change_flag
- regime_conf
- regime_confidence
- regime_duration
- regime_label
- regime_neutral_probability
- regime_score
- regime_score_raw
- regime_trend_ratio
- regime_trend_ratio_zscore
- regime_volatility_20d

### short_interest (22)

- short_interest_borrow_rate
- short_interest_borrow_rate_zscore_3y
- short_interest_change_1m
- short_interest_change_3m
- short_interest_conf
- short_interest_confidence
- short_interest_days_to_cover
- short_interest_float_short_pct
- short_interest_has_data
- short_interest_momentum
- short_interest_pct_zscore_3y
- short_interest_percent
- short_interest_ratio
- short_interest_score
- short_interest_score_raw
- short_interest_shares_on_loan_pct
- short_interest_short_to_oi_ratio
- short_interest_short_vs_institutional
- short_interest_squeeze_probability
- short_interest_squeeze_risk_flag
- short_interest_squeeze_risk_score
- short_interest_zscore_1y

### subsidiary (12)

- subsidiary_complexity_score
- subsidiary_conf
- subsidiary_confidence
- subsidiary_opex_qoq_growth
- subsidiary_opex_spending
- subsidiary_opex_yoy_growth
- subsidiary_rd_intensity
- subsidiary_rd_qoq_growth
- subsidiary_rd_spending
- subsidiary_rd_yoy_growth
- subsidiary_score
- subsidiary_score_raw

### tech_micro_hf (3)

- tech_micro_hf_conf
- tech_micro_hf_score
- tech_micro_hf_score_raw

### tft_features (20)

- tft_features_conf
- tft_features_confidence
- tft_features_month_phase
- tft_features_score
- tft_features_score_raw
- tft_features_seasonality_importance
- tft_features_stability_long
- tft_features_stability_score
- tft_features_stability_short
- tft_features_trend_consistency
- tft_features_trend_importance
- tft_features_trend_long_strength
- tft_features_trend_short_strength
- tft_features_trend_signal_to_noise
- tft_features_vol_long
- tft_features_vol_regime_zscore
- tft_features_vol_short
- tft_features_vol_trend
- tft_features_volatility_importance
- tft_features_weekday_effect

### vol_deriv_hf (3)

- vol_deriv_hf_conf
- vol_deriv_hf_score
- vol_deriv_hf_score_raw

## Union summary (all scanned artifacts)

- <base>: 6
- <unknown>: 12
- calibration: 3
- earnings_transcript_hf: 4
- forecast_hf: 2
- fundamental_val_hf: 2
- macro_regime_hf: 2
- news_nlp_hf: 2
- online_learning: 8
- options: 26
- quantile_forecast: 3
- short_interest: 22
- tech_micro_hf: 2
- vol_deriv_hf: 2

## cache/features/AAPL_h63_merged.parquet

### <base> (3)

- date
- has_data
- is_etf

### <unknown> (12)

- hf_confidence
- hf_embed_1
- hf_embed_2
- hf_embed_3
- hf_embed_4
- hf_embed_5
- hf_embed_6
- hf_embed_7
- hf_embed_8
- hf_macro_score
- hf_regime
- hf_volatility

### earnings_transcript_hf (4)

- earnings_transcript_hf_conf
- earnings_transcript_hf_has_data
- earnings_transcript_hf_score
- earnings_transcript_hf_score_raw

### options (26)

- options_atm_iv
- options_bid_ask_spread_pct
- options_call_ask_avg
- options_call_bid_avg
- options_call_iv_avg
- options_call_last_avg
- options_call_oi
- options_call_volume
- options_conf
- options_expiries_available
- options_has_data
- options_iv_spread
- options_nearest_expiry_days
- options_otm_call_pct
- options_otm_put_pct
- options_put_ask_avg
- options_put_bid_avg
- options_put_call_oi_ratio
- options_put_call_volume_ratio
- options_put_iv_avg
- options_put_last_avg
- options_put_oi
- options_put_volume
- options_score
- options_score_raw
- options_strikes_available

### short_interest (22)

- short_interest_borrow_rate
- short_interest_borrow_rate_zscore_3y
- short_interest_change_1m
- short_interest_change_3m
- short_interest_conf
- short_interest_confidence
- short_interest_days_to_cover
- short_interest_float_short_pct
- short_interest_has_data
- short_interest_momentum
- short_interest_pct_zscore_3y
- short_interest_percent
- short_interest_ratio
- short_interest_score
- short_interest_score_raw
- short_interest_shares_on_loan_pct
- short_interest_short_to_oi_ratio
- short_interest_short_vs_institutional
- short_interest_squeeze_probability
- short_interest_squeeze_risk_flag
- short_interest_squeeze_risk_score
- short_interest_zscore_1y

## cache/features/AAPL_h63_feast_phase2_base_v1.parquet

### <base> (3)

- event_timestamp
- split
- symbol

### calibration (3)

- calibration_q05_coverage
- calibration_q05_error
- calibration_q10_coverage

### online_learning (8)

- online_learning_asymmetric_calibration_ratio
- online_learning_direction_accuracy
- online_learning_mae
- online_learning_mape
- online_learning_quantile_spread_current
- online_learning_rmse
- online_learning_trust_score
- online_learning_uncertainty_compression_ratio

### quantile_forecast (3)

- quantile_forecast_q05
- quantile_forecast_q50
- quantile_forecast_q95

## cache/features/AAPL_h63_feast_phase2_full_v1.parquet

### <base> (3)

- event_timestamp
- split
- symbol

### calibration (3)

- calibration_q05_coverage
- calibration_q05_error
- calibration_q10_coverage

### forecast_hf (2)

- forecast_hf_conf
- forecast_hf_score

### fundamental_val_hf (2)

- fundamental_val_hf_conf
- fundamental_val_hf_score

### macro_regime_hf (2)

- macro_regime_hf_conf
- macro_regime_hf_score

### news_nlp_hf (2)

- news_nlp_hf_conf
- news_nlp_hf_score

### online_learning (8)

- online_learning_asymmetric_calibration_ratio
- online_learning_direction_accuracy
- online_learning_mae
- online_learning_mape
- online_learning_quantile_spread_current
- online_learning_rmse
- online_learning_trust_score
- online_learning_uncertainty_compression_ratio

### quantile_forecast (3)

- quantile_forecast_q05
- quantile_forecast_q50
- quantile_forecast_q95

### tech_micro_hf (2)

- tech_micro_hf_conf
- tech_micro_hf_score

### vol_deriv_hf (2)

- vol_deriv_hf_conf
- vol_deriv_hf_score
