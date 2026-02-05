import pandas as pd
import numpy as np
from typing import Optional
import os
try:
    from dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
except Exception:
    try:
        from src.dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
    except Exception:
        def to_nyse_close_index(df):
            return df


def fetch(start: Optional[str] = None, end: Optional[str] = None, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    HEDGE-FUND GRADE VIX term structure features.
    
    Features (16 total - normalized, z-scored, shock-aware):
    
    TERM SLOPES (3) - z-scored + winsorized:
    1. vxst_vix_term_slope: VIX9D - VIX (9-day vs 30-day spread)
    2. vix_vxv_term_slope: VIX - VIX3M (30-day vs 3-month spread)
    3. vix_vxmt_term_slope: VIX - VIX6M (30-day vs 6-month spread)
    
    ADVANCED FEATURES (4) - normalized:
    4. normalized_term_slope: (VIX6M - VIX) / (VIX6M + VIX) - range bounded
    5. vix_term_curvature: (VIX6M - VIX) - (VIX - VIX9D) - curve shape
    6. front_back_spread: VIX9D - VIX3M - pure shock measurement
    7. panic_premium: VIX9D / VIX - short-term panic ratio
    
    CONTINUOUS REGIME INDICATORS (5):
    8. vix_roll_yield: (VIX3M - VIX) / VIX - futures roll cost/benefit
    9. vix_ratio_term: VIX3M / VIX - term structure ratio
    10. vix_contango_strength: (VIX3M - VIX) / VIX - regime indicator
    11. vol_risk_premium: IV - RV_20d (z-scored, capped at ±3σ)
    12. vol_risk_premium_pct: (IV - RV) / RV (z-scored, capped)
    
    CHANGE/SHOCK FEATURES (4) - regime breaks:
    13. vix_term_slope_change_1d: daily change in term slope
    14. vix_term_slope_change_5d: 5-day change in term slope
    15. vix_curvature_change: daily change in curvature
    16. panic_premium_change: daily change in panic premium
    
    Data sources:
    - VIX indices: EODHD (preferred) or yfinance fallback
    - Stock RV: EODHD (for vol risk premium features)
    
    REMOVED (binary flags hurt Mamba gradients):
    - contango_flag, backwardation_flag, curve_shape_flag
    """
    def _env_flag(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

    no_proxy = _env_flag("STAGE_B_NO_PROXY_SOURCES", "0")
    eodhd_only = _env_flag("STAGE_B_EODHD_ONLY", "0")

    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider
        
        # ═══════════════════════════════════════════════════════════════════
        # DATA SOURCE: EODHD (Primary)
        # Used by: Linear alpha combiner features (cboe_panic, cboe_slope, cboe_vrp)
        # Enforcement: Set STAGE_B_EODHD_ONLY=1 to disable yfinance fallback
        # ═══════════════════════════════════════════════════════════════════
        # Prefer EODHD symbols (when available). If strict modes are enabled, we
        # refuse to use yfinance.
        vix_candidates = {
            'VIX9D': ['VIX9D.INDX', '^VIX9D'],
            'VIX': ['VIX.INDX', '^VIX'],
            # CBOE 3M and 6M indices are often referred to as VXV/VXMT.
            'VIX3M': ['VXV.INDX', 'VIX3M.INDX', '^VIX3M'],
            'VIX6M': ['VXMT.INDX', 'VIX6M.INDX', '^VIX6M'],
        }
        
        data = {}
        used_source = None

        eodhd = get_eodhd_provider()
        if eodhd and getattr(eodhd, 'api_key', None):
            for name, candidates in vix_candidates.items():
                for ticker_symbol in candidates:
                    try:
                        df = eodhd.get_eod_prices(ticker_symbol, start, end)
                        if df is not None and not df.empty:
                            close_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
                            if close_col is None:
                                continue
                            series = df[close_col].copy()
                            series.index = pd.to_datetime(series.index).tz_localize(None)
                            data[name] = series
                            used_source = 'eodhd'
                            break
                    except Exception:
                        continue

        # Fallback to yfinance only when strict modes are off
        if (not data) and (not no_proxy) and (not eodhd_only):
            import yfinance as yf
            vix_indices_yf = {
                'VIX9D': '^VIX9D',
                'VIX': '^VIX',
                'VIX3M': '^VIX3M',
                'VIX6M': '^VIX6M',
            }
            for name, ticker_symbol in vix_indices_yf.items():
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    df = ticker.history(start=start, end=end) if start and end else ticker.history(period='5y')
                    if df is not None and not df.empty and 'Close' in df.columns:
                        df.index = pd.to_datetime(df.index.date)
                        data[name] = df['Close']
                        used_source = 'yfinance'
                except Exception:
                    continue
        
        if not data:
            if no_proxy or eodhd_only:
                raise RuntimeError(
                    "cboe_term: no VIX term structure data available via EODHD, and strict source policy forbids yfinance"
                )
            return pd.DataFrame()
        
        # Combine all series
        df = pd.DataFrame(data)
        df = df.dropna(how='all')
        
        if df.empty:
            return pd.DataFrame()
        
        out = pd.DataFrame(index=df.index)
        
        # ========== TERM SLOPES (z-scored + winsorized) ==========
        z_window = 252  # 1 year for z-scoring
        
        if 'VIX9D' in df.columns and 'VIX' in df.columns:
            slope = df['VIX9D'] - df['VIX']
            # Z-score
            slope_mean = slope.rolling(z_window, min_periods=60).mean()
            slope_std = slope.rolling(z_window, min_periods=60).std()
            slope_z = (slope - slope_mean) / (slope_std + 1e-9)
            # Winsorize at ±3σ (tails explode in crises)
            out['vxst_vix_term_slope'] = slope_z.clip(-3, 3)
        
        if 'VIX' in df.columns and 'VIX3M' in df.columns:
            slope = df['VIX'] - df['VIX3M']
            slope_mean = slope.rolling(z_window, min_periods=60).mean()
            slope_std = slope.rolling(z_window, min_periods=60).std()
            slope_z = (slope - slope_mean) / (slope_std + 1e-9)
            out['vix_vxv_term_slope'] = slope_z.clip(-3, 3)
        
        if 'VIX' in df.columns and 'VIX6M' in df.columns:
            slope = df['VIX'] - df['VIX6M']
            slope_mean = slope.rolling(z_window, min_periods=60).mean()
            slope_std = slope.rolling(z_window, min_periods=60).std()
            slope_z = (slope - slope_mean) / (slope_std + 1e-9)
            out['vix_vxmt_term_slope'] = slope_z.clip(-3, 3)
        
        # ========== ADVANCED FEATURES (normalized) ==========
        if 'VIX' in df.columns and 'VIX6M' in df.columns:
            # Normalized term slope (already range-bounded by construction)
            out['normalized_term_slope'] = (df['VIX6M'] - df['VIX']) / (df['VIX6M'] + df['VIX'] + 1e-9)
            
            # Term curvature (second derivative of term structure)
            if 'VIX9D' in df.columns:
                raw_curvature = (df['VIX6M'] - df['VIX']) - (df['VIX'] - df['VIX9D'])
                # Normalize by typical range
                curv_mean = raw_curvature.rolling(z_window, min_periods=60).mean()
                curv_std = raw_curvature.rolling(z_window, min_periods=60).std()
                out['vix_term_curvature'] = (raw_curvature - curv_mean) / (curv_std + 1e-9)
        
        # Shock measures (normalized)
        if 'VIX9D' in df.columns and 'VIX3M' in df.columns:
            spread = df['VIX9D'] - df['VIX3M']
            spread_mean = spread.rolling(z_window, min_periods=60).mean()
            spread_std = spread.rolling(z_window, min_periods=60).std()
            out['front_back_spread'] = (spread - spread_mean) / (spread_std + 1e-9)
        
        if 'VIX9D' in df.columns and 'VIX' in df.columns:
            premium = df['VIX9D'] / (df['VIX'] + 1e-9)
            # Z-score the ratio
            premium_mean = premium.rolling(z_window, min_periods=60).mean()
            premium_std = premium.rolling(z_window, min_periods=60).std()
            out['panic_premium'] = (premium - premium_mean) / (premium_std + 1e-9)
        
        # ========== CONTINUOUS REGIME INDICATORS ==========
        if 'VIX' in df.columns and 'VIX3M' in df.columns:
            # Roll yield (continuous, already normalized by VIX level)
            out['vix_roll_yield'] = (df['VIX3M'] - df['VIX']) / (df['VIX'] + 1e-9)
            
            # Term structure ratio (continuous)
            out['vix_ratio_term'] = df['VIX3M'] / (df['VIX'] + 1e-9)
            
            # Contango strength (continuous, same as roll_yield but explicit)
            out['vix_contango_strength'] = (df['VIX3M'] - df['VIX']) / (df['VIX'] + 1e-9)
        
        # ========== STOCK-SPECIFIC VOL RISK PREMIUM (z-scored, capped) ==========
        # Only calculate if symbol is provided
        if symbol and 'VIX' in df.columns:
            try:
                eodhd = get_eodhd_provider()
                if eodhd and eodhd.api_key:
                    # Get stock prices to calculate realized volatility
                    stock_df = eodhd.get_eod_prices(symbol, start, end)
                    if stock_df is not None and not stock_df.empty and 'Close' in stock_df.columns:
                        returns = stock_df['Close'].pct_change()
                        # 20-day realized volatility (annualized, consistent window)
                        rv_20d = returns.rolling(20).std() * np.sqrt(252) * 100  # Convert to percentage
                        rv_20d = rv_20d.reindex(df.index)
                        
                        # Vol risk premium (IV - RV, where IV = VIX for market)
                        # NOTE: For individual stocks, this is approximate (VIX is SPX IV)
                        raw_vrp = df['VIX'] - rv_20d
                        
                        # Z-score cross-time (essential for regime stationarity)
                        vrp_mean = raw_vrp.rolling(z_window, min_periods=60).mean()
                        vrp_std = raw_vrp.rolling(z_window, min_periods=60).std()
                        vrp_z = (raw_vrp - vrp_mean) / (vrp_std + 1e-9)
                        
                        # Cap at ±3σ (unbounded extremes are dangerous)
                        out['vol_risk_premium'] = vrp_z.clip(-3, 3)
                        
                        # Vol risk premium percentage (also z-scored and capped)
                        raw_vrp_pct = (df['VIX'] - rv_20d) / (rv_20d + 1e-9)
                        vrp_pct_mean = raw_vrp_pct.rolling(z_window, min_periods=60).mean()
                        vrp_pct_std = raw_vrp_pct.rolling(z_window, min_periods=60).std()
                        vrp_pct_z = (raw_vrp_pct - vrp_pct_mean) / (vrp_pct_std + 1e-9)
                        out['vol_risk_premium_pct'] = vrp_pct_z.clip(-3, 3)
            except Exception:
                pass  # Skip vol risk premium if stock data unavailable
        
        # ========== CHANGE/SHOCK FEATURES (regime breaks) ==========
        # These capture WHEN term structure shifts, not just WHERE it is
        
        # Term slope changes (use the main VIX-VIX3M slope as reference)
        if 'vix_vxv_term_slope' in out.columns:
            out['vix_term_slope_change_1d'] = out['vix_vxv_term_slope'].diff(1)
            out['vix_term_slope_change_5d'] = out['vix_vxv_term_slope'].diff(5)
        
        # Curvature change (regime transitions show up here first)
        if 'vix_term_curvature' in out.columns:
            out['vix_curvature_change'] = out['vix_term_curvature'].diff(1)
        
        # Panic premium change (stress acceleration)
        if 'panic_premium' in out.columns:
            out['panic_premium_change'] = out['panic_premium'].diff(1)

        
        # Forward-fill to handle gaps
        for col in out.columns:
            first_valid_idx = out[col].first_valid_index()
            if first_valid_idx is not None:
                out.loc[first_valid_idx:, col] = out.loc[first_valid_idx:, col].ffill()
        
        # Drop rows where ALL columns are NaN
        out = out.dropna(how='all')
        
        # Add metadata
        out.attrs['provenance'] = {'source': f"{used_source or 'unknown'}_vix_term_structure"}
        out.attrs['telemetry'] = {'status': 'ok', 'source': used_source or 'unknown', 'proxy': False}
        out.attrs['feature_counts'] = {'generated': len(out.columns)}
        
        return out
        
    except Exception as e:
        print(f"❌ CBOE term structure failed: {e}")
        import traceback
        traceback.print_exc()
        if no_proxy or eodhd_only:
            raise
        return pd.DataFrame()
