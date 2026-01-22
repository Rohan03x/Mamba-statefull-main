import pandas as pd
# yfinance removed: use universal fetcher (Tiingo/cache) only
from typing import Dict, Optional
import os

try:
    from dcf_lab.providers.fred_provider import FREDProvider
except Exception:
    FREDProvider = None  # type: ignore
try:
    from src.dcf_lab.utils.timealign import to_nyse_close_index, lag_if_post_close
except Exception:
    def to_nyse_close_index(df):
        return df
    def lag_if_post_close(s, days=1):  # noqa: ARG001 - compatibility signature
        return s


DEFAULT_PROXIES = {
    'NVDA': ['^SOX'],
    'AAPL': ['DXY'],
    'MSFT': ['^SOX'],
}


def _download(symbols, start=None, end=None) -> pd.DataFrame:
    eodhd_only = os.environ.get("STAGE_B_EODHD_ONLY", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    if eodhd_only:
        return pd.DataFrame()
    # Use universal fetcher (Tiingo-only, no yfinance fallback)
    try:
        from universal_data_fetcher import get_universal_fetcher
        fetcher = get_universal_fetcher()
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        out = pd.DataFrame()
        for sym in symbols:
            df = None
            # Try Tiingo first
            try:
                # Use start/end if provided, otherwise use period='5y'
                if start and end:
                    df = fetcher.download([sym], start=start, end=end, interval='1d')
                else:
                    df = fetcher.download(sym, period='5y')
            except Exception:
                pass  # Fall through to yfinance
            
            # Fallback to yfinance if Tiingo failed
            if df is None or df.empty:
                try:
                    import yfinance as yf
                    if start and end:
                        df = yf.download(sym, start=start, end=end, progress=False)
                    else:
                        df = yf.download(sym, period='5y', progress=False)
                except Exception:
                    pass  # Skip this symbol
            
            # Process the result (whether from Tiingo or yfinance)
            if df is not None and not df.empty:
                # Ensure timezone-naive
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                # Use Close column
                if 'Close' in df.columns:
                    out[sym] = df['Close']
                elif 'close' in df.columns:
                    out[sym] = df['close']
        
        if not out.empty:
            # Ensure timezone-naive on output
            if hasattr(out.index, 'tz') and out.index.tz is not None:
                out.index = out.index.tz_localize(None)
            return out
    except Exception:
        pass
    
    # No fallback - return empty if Tiingo fails
    return pd.DataFrame()


def _fred_series_for_proxy(proxy: str) -> Optional[str]:
    # Map logical proxies to FRED series when possible
    if proxy.upper() == 'DXY':
        return 'DTWEXBGS'  # Broad Dollar Index
    return None


def _load_proxy_series(proxy: str, start: Optional[str], end: Optional[str], lag_days: int = 1) -> pd.DataFrame:
    # Check if this is a FRED-only symbol first to avoid unnecessary API calls
    fred_id = _fred_series_for_proxy(proxy)
    if FREDProvider is not None and fred_id is not None:
        try:
            fred = FREDProvider()
            df = fred.get_series_data(fred_id, start_date=start, end_date=end, frequency='d')
            if not df.empty and fred_id in df.columns:
                out = df.rename(columns={fred_id: proxy})
                out = to_nyse_close_index(out)
                # Ensure timezone-naive to match Tiingo data
                if hasattr(out.index, 'tz') and out.index.tz is not None:
                    out.index = out.index.tz_localize(None)
                # lag FRED to reflect post-close publication timing
                out[proxy] = lag_if_post_close(out[proxy], lag_days)
                return out
        except Exception as e:
            print(f"⚠️ FRED failed for {proxy} ({fred_id}): {e}")
    
    # Enhanced fallback mapping for problematic tickers
    fallback_mappings = {
        'DXY': ['UUP', 'DX-Y.NYB'],  # Dollar ETF and futures alternatives
        'T10YIE': ['^TNX'],  # 10Y Treasury yield as fallback
    }
    
    # For symbols with FRED mapping, skip direct ticker lookup and go straight to fallback
    # This avoids unnecessary API errors for symbols that don't exist as tickers
    if fred_id is not None:
        print(f"🔄 FRED lookup failed for {proxy}, trying fallback symbols")
    else:
        # Try original proxy first (only if not a FRED symbol)
        try:
            result = _download([proxy], start=start, end=end)
            if not result.empty:
                # Ensure timezone-naive
                if hasattr(result.index, 'tz') and result.index.tz is not None:
                    result.index = result.index.tz_localize(None)
                return result
        except Exception as e:
            print(f"⚠️ Primary universal fetcher lookup failed for {proxy}: {e}")
    
    # Try fallback symbols
    if proxy in fallback_mappings:
        for fallback in fallback_mappings[proxy]:
            try:
                print(f"🔄 Trying fallback {fallback} for {proxy}")
                result = _download([fallback], start=start, end=end)
                if not result.empty:
                    # Rename to original proxy name
                    result.columns = [proxy if col == fallback else col for col in result.columns]
                    # Ensure timezone-naive
                    if hasattr(result.index, 'tz') and result.index.tz is not None:
                        result.index = result.index.tz_localize(None)
                    return result
            except Exception as e:
                print(f"⚠️ Fallback {fallback} failed: {e}")
                continue
    
    # Return empty DataFrame if all attempts fail
    print(f"❌ All data sources failed for {proxy}")
    return pd.DataFrame()


def fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None, mapping: Optional[Dict[str, list]] = None, macro_lag_days: int = 1) -> pd.DataFrame:
    """
    HEDGE-FUND GRADE cross-asset analysis: macro risk-on/off + shock propagation (20 features).
    
    Philosophy: Level tells you where you are, CHANGE tells you when it breaks.
    
    Categories:
    
    A. ETF-Based Correlations (3):
       - spy_corr_20d: Rolling correlation with S&P 500 ETF
       - qqq_corr_20d: Rolling correlation with Nasdaq 100 ETF
       - sector_etf_corr_20d: Rolling correlation with sector ETF (XLK, XLY, XLF, etc.)
    
    B. Beta & Factor Exposure (5 - CRITICAL for 2021-2022 regime breaks):
       - beta_20d: Rolling beta to SPY (systematic risk exposure)
       - beta_change_rate: Rate of beta change (factor rotation detector) - WINSORIZED
       - beta_volatility_20d: Volatility of beta estimates (stability measure)
       - beta_volatility_change: Change in beta stability (regime break detector) - NEW
       - beta_sign_flip_flag: Regime shift indicator (1 if beta crossed zero)
    
    C. Volatility Relationships (3):
       - vix_corr_20d: Correlation with VIX (fear index sensitivity)
       - vix_spread_indicator: VIX / Realized Vol ratio (implied vs realized)
       - realized_vol_vs_spy_corr: Volatility regime co-movement with market
    
    D. Lead/Lag Analysis (2 - timing modifiers, not primary signals):
       - spy_leads_stock_5d: Market leads stock (beta lag detector, 1-day lag)
       - stock_leads_spy_5d: Stock leads market (sector leadership, 1-day lag)
    
    E. Rate Sensitivity (4 - WITH CHANGE FEATURES):
       - tnx_corr_20d: Correlation with 10Y Treasury yields
       - irx_corr_20d: Correlation with 3M T-bills
       - tnx_corr_change_5d: 5-day change in 10Y rate correlation - NEW
       - irx_corr_change_5d: 5-day change in 3M rate correlation - NEW
    
    F. Credit Spread Dynamics (2 - z-scored/clipped):
       - credit_spread_level: HYG/LQD ratio deviation from 1-year baseline (crisis indicator)
       - asset_corr_hyg_60: Correlation with high-yield credit (junk bond exposure)
    
    G. FX Risk-On/Off (1):
       - asset_corr_uup_60: Correlation with US Dollar Index (defensive vs risk-on)
    
    H. Composite Risk Factor (2 - GOVERNANCE: conditioning signal only):
       - risk_onoff_factor: Fixed-weight linear combination (macro thermostat, NOT trading signal)
         Formula: 0.5*corr_SPY + 0.3*corr_QQQ - 0.2*corr_UUP - 0.2*corr_VIX
         RULES: (1) weights fixed, (2) never learned per symbol, (3) never dominates Stage-A
       - cross_asset_coupling_change: 5-day change in risk factor (regime transition detector) - NEW
    
    Requirements:
    - Minimum 60 days of data for full feature set
    - 20/60-day rolling windows for features
    - EODHD data source (SPY, QQQ, VIX.INDX, TNX.INDX, IRX.INDX, HYG, LQD, UUP, sector ETFs)
    
    Hedge-fund design:
    - All change features for early regime detection
    - Beta block addresses 2021-2022 failure mode
    - Winsorization on volatile features
    - No over-fitting (lag horizons capped at 1-3 days)
    - Families stay independent (correlation, cboe_term, cross_asset serve different roles)
    """
    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider
        import numpy as np
        
        eodhd = get_eodhd_provider()
        if not eodhd or not eodhd.api_key:
            return pd.DataFrame()
        
        # Fetch stock data
        stock_df = eodhd.get_eod_prices(symbol, start, end)
        if stock_df is None or stock_df.empty or 'Close' not in stock_df.columns:
            return pd.DataFrame()
        
        stock_close = stock_df['Close']
        stock_returns = stock_close.pct_change()
        
        # Fetch market indices
        spy_df = eodhd.get_eod_prices('SPY', start, end)
        qqq_df = eodhd.get_eod_prices('QQQ', start, end)
        
        # Fetch volatility indices
        vix_df = eodhd.get_eod_prices('VIX.INDX', start, end)
        
        # Fetch rate indices
        tnx_df = eodhd.get_eod_prices('TNX.INDX', start, end)  # 10Y yields
        irx_df = eodhd.get_eod_prices('IRX.INDX', start, end)  # 3M T-bills
        
        # Determine sector ETF (simplified mapping)
        sector_etf_map = {
            'AAPL': 'XLK',  # Technology
            'MSFT': 'XLK',
            'GOOGL': 'XLK',
            'NVDA': 'XLK',
            'TSLA': 'XLY',  # Consumer Discretionary
            'AMZN': 'XLY',
            'JPM': 'XLF',   # Financials
            'BAC': 'XLF',
            'JNJ': 'XLV',   # Healthcare
            'PFE': 'XLV',
        }
        sector_etf = sector_etf_map.get(symbol, 'SPY')  # Default to SPY
        sector_df = eodhd.get_eod_prices(sector_etf, start, end)
        
        # Build output dataframe
        out = pd.DataFrame(index=stock_df.index)
        
        # ==== A. ETF-Based Correlations ====
        if spy_df is not None and not spy_df.empty and 'Close' in spy_df.columns:
            spy_returns = spy_df['Close'].pct_change()
            spy_returns = spy_returns.reindex(stock_returns.index)
            out['spy_corr_20d'] = stock_returns.rolling(20).corr(spy_returns)
        
        if qqq_df is not None and not qqq_df.empty and 'Close' in qqq_df.columns:
            qqq_returns = qqq_df['Close'].pct_change()
            qqq_returns = qqq_returns.reindex(stock_returns.index)
            out['qqq_corr_20d'] = stock_returns.rolling(20).corr(qqq_returns)
        
        if sector_df is not None and not sector_df.empty and 'Close' in sector_df.columns:
            sector_returns = sector_df['Close'].pct_change()
            sector_returns = sector_returns.reindex(stock_returns.index)
            out['sector_etf_corr_20d'] = stock_returns.rolling(20).corr(sector_returns)
        
        # ==== B. Beta & Factor Exposure ====
        if spy_df is not None and not spy_df.empty and 'Close' in spy_df.columns:
            spy_returns = spy_df['Close'].pct_change().reindex(stock_returns.index)
            
            # Rolling beta (20-day window)
            def rolling_beta(y, x, window=20):
                """Calculate rolling beta: cov(y,x) / var(x)"""
                covariance = y.rolling(window).cov(x)
                variance = x.rolling(window).var()
                return covariance / (variance + 1e-9)
            
            beta_20d = rolling_beta(stock_returns, spy_returns, 20)
            out['beta_20d'] = beta_20d
            
            # Beta change rate (how fast beta is changing) - ABSOLUTE MAGNITUDE for regime aggregation
            # CRITICAL: RoleAwareContext treats large values as "stress"; signed values confuse the aggregator
            # Store abs(zscore) so that any rapid beta change → higher stress
            beta_change_raw = beta_20d.pct_change(5)
            beta_change_mean = beta_change_raw.rolling(252, min_periods=60).mean()
            beta_change_std = beta_change_raw.rolling(252, min_periods=60).std()
            beta_change_z = (beta_change_raw - beta_change_mean) / (beta_change_std + 1e-9)
            out['beta_change_rate'] = np.abs(beta_change_z).clip(0, 3)
            
            # Beta volatility (stability of beta estimates)
            # Calculate volatility of beta over 20-day window
            # Need at least 20 valid beta values for meaningful rolling std
            if len(beta_20d.dropna()) >= 20:
                beta_vol = beta_20d.rolling(20, min_periods=10).std()
                out['beta_volatility_20d'] = beta_vol
                
                # Beta volatility change (regime break detector) - ABSOLUTE MAGNITUDE
                # Rapid changes in beta stability → regime break, regardless of direction
                out['beta_volatility_change'] = np.abs(beta_vol.pct_change(5)).clip(0, 5)
            else:
                out['beta_volatility_20d'] = pd.Series(index=stock_returns.index, dtype=float)
                out['beta_volatility_change'] = pd.Series(index=stock_returns.index, dtype=float)
            
            # Beta sign flip flag (regime shift detector)
            beta_sign = np.sign(beta_20d)
            beta_sign_shift = beta_sign.shift(1)
            out['beta_sign_flip_flag'] = ((beta_sign != beta_sign_shift) & 
                                          (beta_sign.notna()) & 
                                          (beta_sign_shift.notna())).astype(int)
        
        # ==== C. Cross-Asset Volatility Relationships ====
        # Calculate realized volatility
        realized_vol = stock_returns.rolling(20).std() * np.sqrt(252)
        
        if vix_df is not None and not vix_df.empty and 'Close' in vix_df.columns:
            vix_close = vix_df['Close'].reindex(stock_returns.index)
            
            # VIX correlation
            out['vix_corr_20d'] = stock_returns.rolling(20).corr(vix_close.pct_change())
            
            # VIX spread indicator (implied vs realized vol relationship)
            out['vix_spread_indicator'] = vix_close / (realized_vol + 1e-9)
        
        # Realized vol correlation (with SPY's realized vol)
        # Both series are already 20-day rolling stds, so correlation captures regime co-movement
        if spy_df is not None and not spy_df.empty:
            spy_rv = spy_df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            spy_rv = spy_rv.reindex(stock_returns.index)
            # Calculate if we have at least 20 overlapping non-null data points
            if len(realized_vol.dropna()) >= 20 and len(spy_rv.dropna()) >= 20:
                out['realized_vol_vs_spy_corr'] = realized_vol.rolling(20, min_periods=10).corr(spy_rv)
            else:
                out['realized_vol_vs_spy_corr'] = pd.Series(index=stock_returns.index, dtype=float)
        
        # ==== D. Cross-Asset Lead/Lag ====
        if spy_df is not None and not spy_df.empty and 'Close' in spy_df.columns:
            spy_returns = spy_df['Close'].pct_change().reindex(stock_returns.index)
            
            # SPY leads stock: corr(SPY(t-1), STOCK(t))
            spy_lagged = spy_returns.shift(1)
            out['spy_leads_stock_5d'] = stock_returns.rolling(20).corr(spy_lagged)
            
            # Stock leads SPY: corr(STOCK(t-1), SPY(t))
            stock_lagged = stock_returns.shift(1)
            out['stock_leads_spy_5d'] = spy_returns.rolling(20).corr(stock_lagged)
        
        # ==== E. Rate Sensitivity (with change features) ====
        if tnx_df is not None and not tnx_df.empty and 'Close' in tnx_df.columns:
            tnx_close = tnx_df['Close'].reindex(stock_returns.index)
            tnx_corr = stock_returns.rolling(20).corr(tnx_close.pct_change())
            out['tnx_corr_20d'] = tnx_corr
            # Change in rate correlation (regime transition detector) - ABSOLUTE MAGNITUDE
            # Any shift in rate sensitivity → regime break signal
            out['tnx_corr_change_5d'] = np.abs(tnx_corr.diff(5)).clip(0, 1)
        
        if irx_df is not None and not irx_df.empty and 'Close' in irx_df.columns:
            irx_close = irx_df['Close'].reindex(stock_returns.index)
            irx_corr = stock_returns.rolling(20).corr(irx_close.pct_change())
            out['irx_corr_20d'] = irx_corr
            # Change in rate correlation (regime transition detector) - ABSOLUTE MAGNITUDE
            out['irx_corr_change_5d'] = np.abs(irx_corr.diff(5)).clip(0, 1)
        
        # ==== F. Credit Spread Dynamics ====
        # Fetch credit ETFs: HYG (high yield) and LQD (investment grade)
        hyg_df = eodhd.get_eod_prices('HYG', start, end)
        lqd_df = eodhd.get_eod_prices('LQD', start, end)
        
        if hyg_df is not None and not hyg_df.empty and 'Close' in hyg_df.columns:
            hyg_close = hyg_df['Close'].reindex(stock_returns.index)
            hyg_returns = hyg_close.pct_change()
            
            # Asset correlation with high-yield credit (60-day window for credit regime)
            out['asset_corr_hyg_60'] = stock_returns.rolling(60, min_periods=30).corr(hyg_returns)
            
            # Credit spread level: HYG/LQD price ratio minus 1-year baseline
            if lqd_df is not None and not lqd_df.empty and 'Close' in lqd_df.columns:
                lqd_close = lqd_df['Close'].reindex(stock_returns.index)
                
                # Price ratio (HYG/LQD) - higher when credit spreads tighten (risk-on)
                credit_ratio = hyg_close / (lqd_close + 1e-9)
                
                # Baseline = 252-day (1 year) rolling mean
                credit_baseline = credit_ratio.rolling(252, min_periods=126).mean()
                
                # Spread level = current ratio minus baseline (normalized)
                out['credit_spread_level'] = (credit_ratio - credit_baseline) / (credit_baseline + 1e-9)
        
        # ==== G. FX Risk-On/Off ====
        # Fetch US Dollar Index ETF (UUP)
        uup_df = eodhd.get_eod_prices('UUP', start, end)
        
        if uup_df is not None and not uup_df.empty and 'Close' in uup_df.columns:
            uup_close = uup_df['Close'].reindex(stock_returns.index)
            uup_returns = uup_close.pct_change()
            
            # Asset correlation with USD (60-day window)
            # Negative correlation = risk-on (stock up when dollar down)
            # Positive correlation = defensive (stock up when dollar up)
            out['asset_corr_uup_60'] = stock_returns.rolling(60, min_periods=30).corr(uup_returns)
        
        # ==== H. Composite Risk-On/Off Factor (GOVERNANCE: conditioning signal only) ====
        # CRITICAL: This is a macro THERMOSTAT, not a trading signal
        # Rules:
        #   1. Weights are FIXED (never learned per symbol)
        #   2. Used for conditioning only (low weight in Stage-A)
        #   3. Slow-moving (20-60d windows)
        #   4. Never dominates primary signals
        # 
        # Linear combination: 0.5*corr_SPY + 0.3*corr_QQQ - 0.2*corr_UUP - 0.2*corr_VIX
        # Higher values = stronger risk-on behavior
        # Lower values = defensive/risk-off behavior
        
        risk_components = {}
        if 'spy_corr_20d' in out.columns:
            risk_components['spy'] = out['spy_corr_20d'].fillna(0) * 0.5
        if 'qqq_corr_20d' in out.columns:
            risk_components['qqq'] = out['qqq_corr_20d'].fillna(0) * 0.3
        if 'asset_corr_uup_60' in out.columns:
            risk_components['uup'] = -out['asset_corr_uup_60'].fillna(0) * 0.2
        if 'vix_corr_20d' in out.columns:
            risk_components['vix'] = -out['vix_corr_20d'].fillna(0) * 0.2
        
        # Only create risk factor if we have at least 2 components
        if len(risk_components) >= 2:
            risk_factor = sum(risk_components.values())
            out['risk_onoff_factor'] = risk_factor
            
            # Change in cross-asset coupling (regime transition detector) - ABSOLUTE MAGNITUDE
            # Level tells you where you are, CHANGE tells you when it breaks
            out['cross_asset_coupling_change'] = np.abs(risk_factor.diff(5)).clip(0, 1)
            
            # CRITICAL: risk_offness for portfolio overlays
            # risk_onoff_factor is directional (high = risk-on); don't feed raw into stress aggregator
            # risk_offness = one-sided stress signal: higher when market is risk-off
            # Formula: clip((0.5 - risk_onoff_factor) / scale, 0, 1)
            # When risk_onoff_factor < 0.5 (risk-off) → positive stress
            # When risk_onoff_factor >= 0.5 (risk-on) → zero stress (not "negative stress")
            scale = 0.5  # sensitivity parameter
            out['risk_offness'] = ((0.5 - risk_factor) / scale).clip(0, 1)
        
        # Forward-fill to handle rolling window NaNs, but preserve data integrity
        # Only fill gaps AFTER we have at least one valid value per column
        # This ensures we don't backfill unreliable early period data
        for col in out.columns:
            # Find first valid index for this column
            first_valid_idx = out[col].first_valid_index()
            if first_valid_idx is not None:
                # Only forward-fill from first valid value onward
                out.loc[first_valid_idx:, col] = out.loc[first_valid_idx:, col].ffill()
        
        # Drop rows where ALL columns are still NaN (early period before any valid data)
        out = out.dropna(how='all')
        
        # Add metadata
        out.attrs['provenance'] = {
            'source': 'eodhd_comprehensive_cross_asset_enhanced',
            'hedge_fund_grade': True,
            'original_count': 18,
            'current_count': len(out.columns),
            'change_features_added': ['beta_volatility_change', 'tnx_corr_change_5d', 'irx_corr_change_5d', 'cross_asset_coupling_change'],
        }
        out.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'eodhd',
            'proxy': False,
            'beta_change_winsorized': True,
            'composite_governance': 'fixed_weights_conditioning_only',
        }
        out.attrs['feature_counts'] = {'generated': len(out.columns), 'expected': 20}
        
        return out
        
    except Exception as e:
        print(f"❌ Cross-asset features failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
