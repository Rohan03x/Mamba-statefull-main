import os

import numpy as np
import pandas as pd
import streamlit as st

from dcf_lab.app import tft_forecast_app
from dcf_lab.core_features import compute_drivers, live_multiples
from dcf_lab.data_provider import get_provider
from dcf_lab.exporters import export_workbook
from dcf_lab.forecast_model import ForecastInputs, build_forecast
from dcf_lab.mapping import normalize_financials
from dcf_lab.ml import estimate_drivers_ml
from dcf_lab.ml_advanced import build_driver_paths
from dcf_lab.ml_infer import infer_paths_sklearn
from dcf_lab.ml_timeseries import build_paths_arima
from dcf_lab.news import rank_news
from dcf_lab.settings import DEFAULT_MARKET_RISK_PREMIUM, DEFAULT_RISK_FREE_RATE
from dcf_lab.valuation import discounted_cash_flow, monte_carlo, sensitivity_grid

st.set_page_config(
    page_title="DCF Valuation Suite",
    page_icon="💹",
    layout="wide")

# Page selection
PAGES = {
    "DCF Valuation": "main",
    "TFT Forecasting": "tft"
}
page = st.sidebar.radio("Navigation", list(PAGES.keys()))


def fmt_money(x, units: str = "Auto", money_decimals: int = 2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        v = float(x)
        if units == "Auto":
            if abs(v) >= 1e9:
                return f"{v/1e9:.{money_decimals}f}B"
            if abs(v) >= 1e6:
                return f"{v/1e6:.{money_decimals}f}M"
            if abs(v) >= 1e3:
                return f"{v/1e3:.{money_decimals}f}K"
            return f"{v:.{money_decimals}f}"
        scale_map = {
            "Billions (B)": (
                1e9, "B"), "Millions (M)": (
                1e6, "M"), "Thousands (K)": (
                1e3, "K"), "Raw": (
                    1.0, "")}
        scale, suf = scale_map.get(units, (1.0, ""))
        return f"{v/scale:.{money_decimals}f}{suf}"
    except Exception:
        return "—"


def fmt_ratio(x, decimals: int = 2):
    try:
        if x is None:
            return "—"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"


def scale_df(df: pd.DataFrame, units: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    scale_map = {
        "Billions (B)": 1e9,
        "Millions (M)": 1e6,
        "Thousands (K)": 1e3,
        "Raw": 1.0,
        "Auto": None}
    scale = scale_map.get(units, None)
    if scale in (None, 1.0):
        return df
    out = df.copy()
    for c in out.columns:
        if c != "year" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c] / scale
    return out


st.markdown(
    """
    <style>
    .small-note { color:#888; font-size:0.86rem; }
    .pill { display:inline-block; padding:2px 8px; border-radius:12px; background:#222; color:#ccc; font-size:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([0.8, 0.2])
with header_left:
    st.title("DCF Suite — API-First")
with header_right:
    st.write("")
    st.caption("Provider-driven valuation")

with st.sidebar:
    st.header("Provider")
    provider_options = [
        "auto",
        "yfinance",
        "finnhub",
        "fmp",
        "ciq_capiq",
        "ciq_gds",
        "ciq_excel",
        "ciq_api",
        "alpha",
        "ciq_snowflake",
    ]
    default_provider = os.getenv("PROVIDER", "yfinance")
    provider_name = st.selectbox(
        "Data provider", options=provider_options,
        index=provider_options.index(default_provider)
        if default_provider in provider_options else provider_options.index(
            "yfinance"),)
    st.markdown(
        f"<span class='pill'>Provider: {provider_name}</span>",
        unsafe_allow_html=True)
    st.header("Settings")
    ticker = st.text_input(
        "Company ticker",
        "AAPL",
        help="Public ticker (e.g., AAPL, MSFT, NVDA)").upper().strip()
    years = st.slider(
        "Forecast years",
        5,
        10,
        7,
        1,
        help="Explicit forecast horizon. Longer horizons usually increase terminal value weight.")
    terminal = st.selectbox(
        "Terminal value method", ["gordon", "exit_multiple"],
        index=0,
        help="Gordon growth (perpetuity) or Exit Multiple (EV/EBITDA at final year).")
    g = st.slider(
        "Terminal growth (g)", -0.01, 0.05, 0.025, 0.001,
        help="Perpetual growth beyond horizon; 0–3% typical for mature markets.")
    multiple = st.slider(
        "Exit multiple (EV/EBITDA)",
        4.0,
        20.0,
        10.0,
        0.5,
        help="Used only with Exit Multiple terminal method.")
    rf = st.slider(
        "Risk‑free rate",
        0.0,
        0.10,
        float(DEFAULT_RISK_FREE_RATE),
        0.001,
        help="10Y government yield proxy. Affects cost of equity and WACC.")
    mrp = st.slider(
        "Market risk premium",
        0.03,
        0.10,
        float(DEFAULT_MARKET_RISK_PREMIUM),
        0.001,
        help="Equity risk premium used by CAPM for cost of equity.")
    do_mc = st.checkbox(
        "Monte Carlo (1,000)", value=False,
        help="Run 1,000 simulations over growth, margins and WACC to get a valuation distribution.")
    st.subheader("Display")
    units = st.selectbox("Number units",
                         ["Auto",
                          "Billions (B)",
                          "Millions (M)",
                          "Thousands (K)",
                          "Raw"],
                         index=0,
                         help="Scale values across KPIs and tables.")
    money_decimals = st.slider(
        "Money decimals",
        0,
        3,
        2,
        1,
        help="Decimal places for currency values.")
    ratio_decimals = st.slider(
        "Ratio decimals",
        0,
        3,
        2,
        1,
        help="Decimal places for ratios and multiples.")
    st.subheader("Machine Learning")
    ml_assist = st.checkbox(
        "ML assist (trend‑based drivers)",
        value=False,
        key="ml_assist",
        help="Learns growth & ratios from recent history to refine drivers.")
    ml_paths = st.checkbox(
        "ML paths (advanced: trends)",
        value=False,
        key="ml_paths",
        help="Per‑year paths for growth/margins/CapEx/NWC using trends.")
    ml_sklearn = st.checkbox(
        "ML paths (sklearn)", value=False, key="ml_sklearn",
        help="Per‑year paths using ElasticNet/RandomForest trained on company history.")
    ts_arima = st.checkbox(
        "TS paths (ARIMA)", value=False, key="ts_arima",
        help="Time‑series (ARIMA) revenue level forecast converted to per‑year growth path.")

run = st.button("Run", type="primary")

# Route to appropriate page based on selection
if page == "TFT Forecasting":
    tft_forecast_app()
elif page == "DCF Valuation" and run:
    prov = get_provider(provider_name)
    try:
        with st.spinner("Fetching data…"):
            prof = prov.get_profile(ticker) or {}
            market = prov.get_market(ticker) or {}
            fin = prov.get_financials(ticker, years=5) or {}
        inc_df, bal_df, cf_df = normalize_financials(fin)
        # Use last N years for ML learning
        hist_years = 10
        inc_hist = inc_df.sort_values('date').tail(
            hist_years) if 'date' in inc_df.columns else inc_df.tail(hist_years)
        bal_hist = bal_df.sort_values('date').tail(
            hist_years) if 'date' in bal_df.columns else bal_df.tail(hist_years)
        cf_hist = cf_df.sort_values('date').tail(
            hist_years) if 'date' in cf_df.columns else cf_df.tail(hist_years)
        drivers = compute_drivers(inc_hist, bal_hist, cf_hist)
        # Apply ML toggles configured in sidebar (persist across reruns)
        if ml_assist:
            drivers = estimate_drivers_ml(inc_hist, bal_hist, cf_hist, drivers)
        if ml_paths:
            drivers.update(
                build_driver_paths(
                    inc_hist,
                    bal_hist,
                    cf_hist,
                    years,
                    drivers["rev_col"]))
        if ml_sklearn:
            drivers.update(
                infer_paths_sklearn(
                    inc_hist,
                    bal_hist,
                    cf_hist,
                    years,
                    drivers["rev_col"]))
        if ts_arima:
            drivers.update(
                build_paths_arima(
                    inc_hist,
                    bal_hist,
                    years,
                    drivers["rev_col"]))
        st.session_state["ml_enabled"] = bool(
            ml_assist or ml_paths or ml_sklearn or ts_arima)

        inputs = ForecastInputs(
            years=years,
            terminal_method=terminal,
            terminal_growth=float(g),
            exit_multiple=float(multiple),
            rf_rate=float(rf),
            market_risk_premium=float(mrp),
            beta=market.get("beta"),
            shares_outstanding=market.get("shares"),
        )

        with st.spinner("Building forecast & valuation…"):
            is_df, bs_df, cf_df, starting = build_forecast(
                inc_df, bal_df, cf_df, drivers, inputs)
            dcf = discounted_cash_flow(
                cf_df,
                is_df,
                drivers,
                inputs,
                market.get("marketcap"),
                starting)
            sens = sensitivity_grid(
                cf_df,
                is_df,
                drivers,
                inputs,
                market.get("marketcap"),
                starting)
            news_raw = prov.get_news(ticker, days=7)
            news = rank_news(news_raw)
            multiples = live_multiples(market, inc_df, bal_df)
            # ML snapshot
            is_ml = bs_ml = cf_ml = None
            if st.session_state.get("ml_enabled") and (
                drivers.get("rev_growth_path") or drivers.get(
                    "gross_margin_path") or drivers.get("ebit_margin_path")):
                try:
                    is_ml, bs_ml, cf_ml, _ = build_forecast(
                        inc_df, bal_df, cf_df, drivers, inputs)
                except Exception:
                    is_ml = bs_ml = cf_ml = None
        mc_eq = mc_ps = None
        if do_mc:
            mc_eq, mc_ps = monte_carlo(
                inc_df, bal_df, cf_df, drivers, inputs,
                market.get("marketcap"),
                N=1000)
    except Exception as e:
        st.error(str(e))
    else:
        tabs = st.tabs(["Overview",
                        "Forecast & DCF",
                        "ML Forecast",
                        "Price Forecast",
                        "News",
                        "Export"])
        with tabs[0]:
            st.subheader(f"{ticker} — {prof.get('name', '').strip()}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Price",
                fmt_money(
                    market.get('price'),
                    units="Raw",
                    money_decimals=money_decimals))
            c2.metric(
                "Market Cap",
                fmt_money(
                    market.get('marketcap'),
                    units=units,
                    money_decimals=money_decimals))
            c3.metric(
                "EV/EBITDA",
                fmt_ratio(
                    multiples.get('ev_ebitda'),
                    ratio_decimals))
            c4.metric("P/E", fmt_ratio(multiples.get('pe'), ratio_decimals))
            st.markdown("**DCF Summary**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "WACC", f"{dcf.get('wacc') * 100: .2f} %"
                if dcf.get('wacc') is not None else "—")
            k2.metric(
                "Enterprise Value", fmt_money(
                    dcf.get('enterprise_value')))
            k3.metric("Equity Value", fmt_money(dcf.get('equity_value')))
            k4.metric("Per Share", fmt_money(dcf.get('per_share')))
            with st.expander("Assumptions (Drivers)"):
                st.write({k: round(v, 4) if isinstance(v, (int, float))
                         else v for k, v in drivers.items() if k != 'rev_col'})

        with tabs[1]:
            st.write("Income Statement (Forecast)")
            st.dataframe(
                scale_df(
                    is_df,
                    units).round(money_decimals),
                use_container_width=True)
            st.write("Balance Sheet (Forecast)")
            st.dataframe(
                scale_df(
                    bs_df,
                    units).round(money_decimals),
                use_container_width=True)
            st.write("Cash Flow (Forecast)")
            st.dataframe(
                scale_df(
                    cf_df,
                    units).round(money_decimals),
                use_container_width=True)
            st.write("Sensitivity (WACC ±1%, g ±0.5%)")
            try:
                st.dataframe(sens, use_container_width=True)
            except Exception:
                st.dataframe(sens)
            try:
                tv_share = float(dcf.get('pv_terminal', 0)) / \
                    max(1e-9, float(dcf.get('enterprise_value', 0)))
                if tv_share > 0.8:
                    st.warning(
                        f"Terminal value share is {
                            tv_share:.0%} (>80%). Review assumptions.")
            except Exception:
                pass
            try:
                last = bs_df.iloc[-1]
                assets = float(last.get('assets_model', 0))
                liab_eq = float(last.get('liab_plus_equity_model', 0))
                if assets and liab_eq:
                    drift = abs(assets - liab_eq) / max(1e-9, assets)
                    if drift > 0.02:
                        st.error(
                            f"Balance check drift {
                                drift:.1%} (Assets vs Debt+Equity)")
            except Exception:
                pass
            if do_mc and mc_eq is not None:
                st.write("Monte Carlo — Equity Value Distribution (preview)")
                st.bar_chart(pd.Series(mc_eq), use_container_width=True)

        with tabs[2]:
            st.subheader("ML Forecast (vs Traditional)")
            if is_ml is None:
                st.info("Enable an ML option in the sidebar to see ML forecast.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Base — Income Statement")
                    st.dataframe(
                        scale_df(
                            is_df,
                            units).round(money_decimals),
                        use_container_width=True)
                with c2:
                    st.write("ML — Income Statement")
                    st.dataframe(
                        scale_df(
                            is_ml,
                            units).round(money_decimals),
                        use_container_width=True)
                try:
                    comp = pd.DataFrame(
                        {'Base': is_df.set_index('year')['revenue'],
                         'ML': is_ml.set_index('year')['revenue']})
                    st.line_chart(comp, use_container_width=True)
                except Exception:
                    pass
            with st.expander("Insights"):
                try:
                    hist_rev = pd.to_numeric(
                        inc_df.get('totalRevenue'), errors='coerce').dropna()
                    cagr_hist = None
                    if len(hist_rev) >= 2:
                        cagr_hist = (hist_rev.iloc[-1] /
                                     max(1e-6, hist_rev.iloc[0])) ** (1 /
                                                                      max(1, len(hist_rev) - 1)) - 1
                    cagr_fwd = None
                    if 'revenue' in is_df.columns and len(
                            is_df['revenue']) >= 2:
                        cagr_fwd = (is_df['revenue'].iloc[-1] /
                                    max(1e-6, is_df['revenue'].iloc[0])) ** (1 /
                                                                             max(1, len(is_df['revenue']) - 1)) - 1
                    msgs = []
                    if cagr_fwd is not None:
                        if cagr_hist is not None:
                            tilt = 'accelerating' if cagr_fwd > cagr_hist else 'decelerating'
                            msgs.append(
                                f"Revenue CAGR ~ {
                                    cagr_fwd *
                                    100:.1f}% ({tilt} vs history {
                                    cagr_hist *
                                    100:.1f}%).")
                        else:
                            msgs.append(
                                f"Revenue CAGR ~ {
                                    cagr_fwd *
                                    100:.1f}% over forecast horizon.")
                    if 'ebit' in is_df.columns and 'revenue' in is_df.columns:
                        em_base = (is_df['ebit']/is_df['revenue']).mean()
                        msgs.append(
                            f"EBIT margin averages ~ {
                                em_base*100:.1f}% across the forecast.")
                    if 'capex' in cf_df.columns and 'revenue' in is_df.columns and len(
                            is_df):
                        capex_int = (cf_df['capex'].abs().mean(
                        )/max(1e-6, is_df['revenue'].mean())) if len(cf_df) else None
                        if capex_int is not None:
                            msgs.append(
                                f"CapEx intensity ~ {
                                    capex_int *
                                    100:.1f}% of revenue (asset growth posture).")
                    try:
                        tv_share = float(dcf.get('pv_terminal', 0)) / \
                            max(1e-9, float(dcf.get('enterprise_value', 0)))
                        msgs.append(
                            f"Terminal value share ~ {
                                tv_share*100:.0f}% of EV.")
                    except Exception:
                        pass
                    try:
                        sents = [n.get('sentiment') for n in news
                                 if n.get('sentiment') is not None]
                        if sents:
                            avg_s = sum(sents)/len(sents)
                            msgs.append(
                                f"News sentiment (7d) {
                                    avg_s:+.2f} — {
                                    'positive tilt' if avg_s > 0 else 'negative tilt' if avg_s < 0 else 'neutral'}.")
                    except Exception:
                        pass
                    if msgs:
                        for m in msgs:
                            st.write("• ", m)
                    else:
                        st.write("No insights available.")
                except Exception:
                    st.write("No insights available.")

        # New: Price Forecast tab using Yahoo Finance
        with tabs[3]:
            st.subheader("Price Forecast (Yahoo Finance)")
            try:
                from dcf_lab.price_forecast import (
                    _ensure_cols,
                    arima_forecast_close,
                    fetch_prices,
                    long_term_monthly,
                    rf_short_term_forecast,
                )

                # Try importing advanced AI models
                try:
                    from dcf_lab.ai_price_forecast import ai_price_forecast
                    AI_MODELS_AVAILABLE = True
                    print("Advanced AI forecasting models available")
                except ImportError:
                    AI_MODELS_AVAILABLE = False
                    print("Advanced AI forecasting models not available")

                forecast_tabs = st.tabs(
                    ["Traditional ML", "Advanced AI Models",
                     "Technical Indicators"])

                with st.spinner(f"Fetching price data for {ticker}..."):
                    try:
                        pf = fetch_prices(ticker, start="2015-01-01")

                        # Check if we have data with the required columns
                        if not pf.empty and 'adj_close' in pf.columns and 'date' in pf.columns:
                            st.success(
                                f"Retrieved {
                                    len(pf)} price points for {ticker}")
                        else:
                            st.warning(
                                "Retrieved data is missing required columns.")
                            pf = _ensure_cols(pf)  # Try to fix column issues

                        # Check if we have enough data for forecasting
                        if not pf.empty and len(
                                pf) > 30:  # Need enough data points for forecasting
                            pf.attrs["ticker"] = ticker

                            # Traditional ML tab
                            with forecast_tabs[0]:
                                st.subheader("Traditional ML Forecasts")

                                with st.spinner("Generating short-term forecasts..."):
                                    rf_pred, top_imp, stats = rf_short_term_forecast(
                                        pf, horizon=10)
                                    ar_pred = arima_forecast_close(
                                        pf, horizon=10)

                                    if not rf_pred.empty and not ar_pred.empty:
                                        base_hist = pf.set_index(
                                            'date')['adj_close'].tail(180)
                                        chart = pd.concat(
                                            [base_hist.rename('Actual'),
                                             rf_pred.rename('RF'),
                                             ar_pred.rename('ARIMA')],
                                            axis=1)
                                        st.line_chart(
                                            chart, use_container_width=True)
                                    else:
                                        st.warning(
                                            "Could not generate short-term forecasts with the available data.")

                                if top_imp:
                                    st.write("Top ML features (importance)")
                                    st.dataframe(pd.DataFrame(
                                        {"feature": list(top_imp.keys()), "importance": list(top_imp.values())}))

                                with st.spinner("Generating long-term forecast..."):
                                    st.write("Long‑term (monthly, 12m)")
                                    lt = long_term_monthly(pf, months=12)
                                    if not lt.empty:
                                        st.line_chart(
                                            lt.rename('Monthly ARIMA'),
                                            use_container_width=True)
                                    else:
                                        st.info(
                                            "Insufficient historical data for long-term monthly forecasting.")

                            # Advanced AI Models tab
                            with forecast_tabs[1]:
                                st.subheader("Advanced AI Forecasts")
                                if AI_MODELS_AVAILABLE:
                                    model_type = st.selectbox(
                                        "Select AI Model Type",
                                        ["transformer", "lstm"],
                                        index=0,
                                        help="Transformer: Better at capturing long-range dependencies. LSTM: Good for sequential data patterns."
                                    )

                                    use_advanced_ai = st.checkbox(
                                        "Use Advanced AI Model (Takes Longer)",
                                        value=False)

                                    if use_advanced_ai:
                                        with st.spinner(f"Training {model_type.upper()} model (this may take a while)..."):
                                            ai_pred, ai_info = ai_price_forecast(
                                                pf, ticker, horizon=10, model_type=model_type)

                                            if not ai_pred.empty and "error" not in ai_info:
                                                st.success(
                                                    f"{model_type.upper()} model trained successfully")

                                                # Show forecast
                                                base_hist = pf.set_index(
                                                    'date')['adj_close'].tail(
                                                    180)
                                                chart = pd.concat([base_hist.rename('Actual'), ai_pred.rename(
                                                    f'AI ({model_type.upper()})')], axis=1)
                                                st.line_chart(
                                                    chart,
                                                    use_container_width=True)

                                                # Show model info
                                                st.write("Model Information")
                                                st.json({
                                                    "Model Type": ai_info.get("model_type", "Unknown"),
                                                    "Training Samples": ai_info.get("training_samples", 0),
                                                    "Features Used": len(ai_info.get("features", [])),
                                                    "Training Loss": round(ai_info.get("final_loss", 0), 6)
                                                })
                                            else:
                                                st.error(
                                                    f"Could not generate AI forecast: {
                                                        ai_info.get(
                                                            'error', 'Unknown error')}")
                                    else:
                                        st.info(
                                            "Enable advanced AI forecasting above to use transformer or LSTM models.")
                                        st.write(
                                            "Note: Training AI models may take a few minutes depending on the amount of data.")
                                else:
                                    st.warning(
                                        "Advanced AI models are not available. Install the required packages first.")
                                    st.code(
                                        "pip install transformers torch pandas numpy matplotlib",
                                        language="bash")

                            # Technical Indicators tab
                            with forecast_tabs[2]:
                                st.subheader("Technical Indicators")

                                # Add technical indicators
                                from dcf_lab.price_forecast import add_tech_indicators
                                tech_df = add_tech_indicators(pf)

                                # Select which indicators to display
                                indicators = st.multiselect(
                                    "Select indicators to display", options=[
                                        "ma20", "ma50", "rsi14", "macd", "vol_20"], default=[
                                        "ma20", "ma50"])

                                if indicators:
                                    # Create dataframe with price and selected
                                    # indicators
                                    plot_df = tech_df.set_index(
                                        "date")[["adj_close"] + indicators]

                                    # Plot
                                    st.line_chart(
                                        plot_df, use_container_width=True)

                                    # Show statistics
                                    st.write(
                                        "Current Technical Indicator Values")
                                    latest = plot_df.iloc[-1].to_dict()
                                    stats_df = pd.DataFrame({
                                        "Indicator": list(latest.keys()),
                                        "Value": [round(v, 4) if isinstance(v, float) else v for v in latest.values()]
                                    })
                                    st.dataframe(stats_df)
                                else:
                                    st.info(
                                        "Select indicators to display from the dropdown above.")
                        else:
                            st.error(
                                f"Insufficient price data available for {ticker}. Please check the ticker symbol or try again later.")
                            st.info(
                                f"Retrieved {
                                    len(pf)} data points, but at least 30 are needed for forecasting.")

                    except Exception as inner_ex:
                        st.error(
                            f"Error processing price data: {
                                str(inner_ex)}")
                        st.info(
                            "Data retrieval worked but there was an error processing the data. Try another ticker.")

            except Exception as ex:
                st.warning(f"Price forecast unavailable: {str(ex)}")
                st.info(
                    "This could be due to temporary data access limitations from Yahoo Finance or an invalid ticker symbol.")

        with tabs[4]:
            st.write("Latest News / Key Developments")
            try:
                with st.spinner(f"Fetching news for {ticker}..."):
                    # Try to fetch news directly if it wasn't fetched earlier
                    if not news:
                        news_raw = prov.get_news(ticker, days=7)
                        news = rank_news(news_raw)

                    if news:
                        df_news = pd.DataFrame(news)[
                            ["date", "headline", "source", "url", "sentiment"]]
                        st.dataframe(df_news)
                    else:
                        st.info("No recent news items found for this ticker.")
                        st.info(
                            "Try a more well-known company or check if the ticker symbol is correct.")
            except Exception as ex:
                st.warning(f"Unable to retrieve news: {str(ex)}")
                st.info(
                    "News retrieval may be limited due to API restrictions or connectivity issues.")

        with tabs[5]:
            st.write("Export results to Excel")
            if st.button("Export results.xlsx"):
                inputs_dict = {
                    "ticker": ticker,
                    "years": years,
                    "terminal_method": terminal,
                    "terminal_growth": g,
                    "exit_multiple": multiple,
                    "rf_rate": rf,
                    "mrp": mrp,
                }
                export_workbook(
                    "results.xlsx",
                    ticker,
                    inputs_dict,
                    is_df,
                    bs_df,
                    cf_df,
                    dcf,
                    sensitivity=sens,
                    mc_samples=mc_eq,
                    news_items=news,
                    data_sources=[{"provider": provider_name}],
                )
                st.success("Saved results.xlsx")
