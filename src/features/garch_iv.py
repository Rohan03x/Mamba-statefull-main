import pandas as pd
import numpy as np
from typing import Optional

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
except Exception:
    try:
        from src.dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
    except Exception:
        def to_nyse_close_index(df):
            return df


def _realized_vol(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate realized volatility (annualized)."""
    r = series.pct_change()
    return r.rolling(window).std() * np.sqrt(252)


def _garch_proxy(series: pd.Series, short: int, long: int) -> pd.Series:
    """Proxy for GARCH using realized volatility difference."""
    s = _realized_vol(series, short)
    l = _realized_vol(series, long)
    return s - l


def _skew_proxy(series: pd.Series, window: int = 60) -> pd.Series:
    """Downside volatility minus upside volatility."""
    r = series.pct_change()
    pos = r.clip(lower=0.0)
    neg = (-r.clip(upper=0.0))
    vol_pos = pos.rolling(window).std()
    vol_neg = neg.rolling(window).std()
    return (vol_neg - vol_pos) * np.sqrt(252)


def _fit_garch_rolling(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Fit GARCH(1,1) model on rolling windows and extract features.
    
    Returns DataFrame with:
    - garch_1d: 1-day ahead forecast
    - garch_5d: 5-day ahead forecast
    - garch_20d: 20-day ahead forecast
    - garch_persistence: alpha + beta (near 1 = persistent regime)
    - garch_long_run_variance: omega / (1 - alpha - beta)
    - garch_log_likelihood: model fit quality (low = regime shift)
    - garch_standardized_residual: returns / conditional_vol
    """
    if not ARCH_AVAILABLE:
        # Return empty features if arch not available
        return pd.DataFrame({
            'garch_1d': np.nan,
            'garch_5d': np.nan,
            'garch_20d': np.nan,
            'garch_persistence': np.nan,
            'garch_long_run_variance': np.nan,
            'garch_log_likelihood': np.nan,
            'garch_standardized_residual': np.nan,
        }, index=returns.index)
    
    results = {
        'garch_1d': [],
        'garch_5d': [],
        'garch_20d': [],
        'garch_persistence': [],
        'garch_long_run_variance': [],
        'garch_log_likelihood': [],
        'garch_standardized_residual': [],
        'index': []
    }
    
    # Convert returns to percentage for GARCH model stability
    returns_pct = returns * 100
    
    for i in range(window, len(returns)):
        window_returns = returns_pct.iloc[i-window:i]
        
        try:
            # Fit GARCH(1,1) model
            model = arch_model(window_returns, vol='Garch', p=1, q=1, rescale=False)
            fitted = model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            omega = fitted.params.get('omega', np.nan)
            alpha = fitted.params.get('alpha[1]', np.nan)
            beta = fitted.params.get('beta[1]', np.nan)
            
            # 1-day ahead forecast
            forecast = fitted.forecast(horizon=20)
            var_1d = forecast.variance.iloc[-1, 0]
            var_5d = forecast.variance.iloc[-1, 4] if len(forecast.variance.columns) > 4 else np.nan
            var_20d = forecast.variance.iloc[-1, 19] if len(forecast.variance.columns) > 19 else np.nan
            
            # Convert variance to annualized volatility
            garch_1d = np.sqrt(var_1d * 252) / 100  # Convert back from percentage
            garch_5d = np.sqrt(var_5d * 252) / 100 if not np.isnan(var_5d) else np.nan
            garch_20d = np.sqrt(var_20d * 252) / 100 if not np.isnan(var_20d) else np.nan
            
            # Persistence (alpha + beta)
            persistence = alpha + beta if not (np.isnan(alpha) or np.isnan(beta)) else np.nan
            
            # Long-run variance (steady state)
            if persistence < 1.0 and not np.isnan(omega):
                long_run_var = omega / (1 - persistence)
                long_run_vol = np.sqrt(long_run_var * 252) / 100
            else:
                long_run_vol = np.nan
            
            # Log-likelihood (model fit quality)
            log_likelihood = fitted.loglikelihood
            
            # Standardized residuals (last value)
            std_resid = fitted.std_resid.iloc[-1] if hasattr(fitted, 'std_resid') else np.nan
            
            results['garch_1d'].append(garch_1d)
            results['garch_5d'].append(garch_5d)
            results['garch_20d'].append(garch_20d)
            results['garch_persistence'].append(persistence)
            results['garch_long_run_variance'].append(long_run_vol)
            results['garch_log_likelihood'].append(log_likelihood)
            results['garch_standardized_residual'].append(std_resid)
            results['index'].append(returns.index[i])
            
        except Exception:
            # If GARCH fitting fails, use NaN
            results['garch_1d'].append(np.nan)
            results['garch_5d'].append(np.nan)
            results['garch_20d'].append(np.nan)
            results['garch_persistence'].append(np.nan)
            results['garch_long_run_variance'].append(np.nan)
            results['garch_log_likelihood'].append(np.nan)
            results['garch_standardized_residual'].append(np.nan)
            results['index'].append(returns.index[i])
    
    df = pd.DataFrame(results)
    df.set_index('index', inplace=True)
    
    # Reindex to match original series
    df = df.reindex(returns.index)
    
    return df


def fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Build comprehensive GARCH volatility features from daily prices.

    Features (15 total):
    
    LEGACY FEATURES (kept for backward compatibility):
      - GARCH30_MINUS_GARCH180: RV spread (30d vs 180d)
      - SKEW_PROXY_DOWNSIDE_MINUS_UPSIDE: Downside vol - upside vol
    
    CORE GARCH FORECASTS (from fitted GARCH(1,1) model):
      - garch_1d: 1-day ahead volatility forecast
      - garch_5d: 5-day ahead volatility forecast
      - garch_20d: 20-day ahead volatility forecast
    
    DERIVED METRICS:
      - garch_ratio_1d_20d: Short-term vs medium-term forecast ratio
      - garch_spike_flag: 1 if garch_1d > 1.5 * garch_20d (regime shock)
      - garch_zscore: Z-score of garch_1d vs 60-day distribution
      - garch_residual_vol: Std dev of standardized residuals (20d)
    
    HIGH-ALPHA FEATURES:
      - garch_vol_of_vol: Std dev of garch_1d over 20 days (regime instability)
      - garch_persistence: alpha + beta (near 1 = persistent regime)
      - garch_long_run_variance: Steady-state variance (slow regime anchor)
      - garch_short_long_ratio: garch_1d / long_run_variance
      - garch_vol_norm_20d: garch_20d / realized_vol_20d
      - garch_vol_momentum: Slope of garch_1d over 10 days
      - garch_shock_indicator: 1 if standardized residual > 2 std
      - garch_model_likelihood: Log-likelihood (low = regime shift)
    """
    try:
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.utcnow().normalize()
        start_dt = pd.to_datetime(start) if start else (end_dt - pd.Timedelta(days=365 * 5))

        # Ensure we have enough history for rolling windows / GARCH estimation
        history_start = start_dt - pd.Timedelta(days=365 * 6)

        data: Optional[pd.DataFrame] = None
        source_name = "unknown"

        # Prefer EODHD daily prices when configured
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider

            eodhd = get_eodhd_provider()
            if getattr(eodhd, "api_key", None):
                data = eodhd.get_eod_prices(
                    symbol,
                    start_date=history_start.strftime("%Y-%m-%d"),
                    end_date=end_dt.strftime("%Y-%m-%d"),
                )
                source_name = "eodhd_daily_close"
        except Exception:
            data = None

        # Fallback to universal data fetcher shim (cache/Tiingo/yfinance)
        if data is None or data.empty:
            from src.data.universal_data_fetcher import download

            data = download(
                symbol,
                start=history_start.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
            )
            source_name = "universal_fetcher_close"

        if data is None or data.empty:
            return pd.DataFrame()

        # Normalize column names
        cols_lower = {c: str(c).lower() for c in data.columns}
        data = data.rename(columns=cols_lower)

        if "close" not in data.columns:
            return pd.DataFrame()

        close = pd.to_numeric(data["close"], errors="coerce")
        close.index = pd.to_datetime(close.index)
        close = close.sort_index()

        # Ensure timezone-naive
        if hasattr(close.index, "tz") and close.index.tz is not None:
            close.index = close.index.tz_localize(None)

        close = close.dropna()
        if close.empty or len(close) < 252:  # Need at least 1 year for GARCH
            return pd.DataFrame()
        
        # Calculate returns
        returns = close.pct_change().dropna()
        
        # Initialize output DataFrame
        out = pd.DataFrame(index=close.index)
        
        # ============================================================
        # LEGACY FEATURES (backward compatibility)
        # ============================================================
        out['GARCH30_MINUS_GARCH180'] = _garch_proxy(close, 30, 180)
        out['SKEW_PROXY_DOWNSIDE_MINUS_UPSIDE'] = _skew_proxy(close, 60)
        
        # ============================================================
        # CORE GARCH MODEL FEATURES
        # ============================================================
        if ARCH_AVAILABLE and len(returns) >= 252:
            # Fit GARCH(1,1) on rolling windows
            garch_features = _fit_garch_rolling(returns, window=252)
            
            # Merge GARCH features
            for col in garch_features.columns:
                out[col] = garch_features[col]
            
            # ============================================================
            # DERIVED METRICS
            # ============================================================
            
            # Ratio of short-term to medium-term forecast
            out['garch_ratio_1d_20d'] = out['garch_1d'] / (out['garch_20d'] + 1e-9)
            
            # Spike flag: 1 if 1-day forecast >> 20-day forecast (regime shock)
            out['garch_spike_flag'] = (out['garch_1d'] > 1.5 * out['garch_20d']).astype(int)
            
            # Z-score of 1-day forecast vs recent distribution
            garch_1d_mean = out['garch_1d'].rolling(60).mean()
            garch_1d_std = out['garch_1d'].rolling(60).std()
            out['garch_zscore'] = (out['garch_1d'] - garch_1d_mean) / (garch_1d_std + 1e-9)
            
            # Residual volatility (uncertainty in model)
            out['garch_residual_vol'] = out['garch_standardized_residual'].rolling(20).std()
            
            # ============================================================
            # HIGH-ALPHA FEATURES
            # ============================================================
            
            # 8. Vol of vol: std dev of GARCH forecast (regime instability)
            out['garch_vol_of_vol'] = out['garch_1d'].rolling(20).std()
            
            # 9. Persistence already calculated in _fit_garch_rolling
            # (no additional calculation needed - already in 'garch_persistence')
            
            # 10. Long-run variance already calculated in _fit_garch_rolling
            # (no additional calculation needed - already in 'garch_long_run_variance')
            
            # 11. Short/long ratio: current vs steady state
            out['garch_short_long_ratio'] = out['garch_1d'] / (out['garch_long_run_variance'] + 1e-9)
            
            # 12. GARCH vs realized vol normalization
            realized_vol_20d = _realized_vol(close, 20)
            out['garch_vol_norm_20d'] = out['garch_20d'] / (realized_vol_20d + 1e-9)
            
            # 13. Vol momentum: slope of GARCH forecast over 10 days
            # Use simple linear regression slope
            def rolling_slope(series, window=10):
                slopes = []
                for i in range(len(series)):
                    if i < window - 1:
                        slopes.append(np.nan)
                    else:
                        y = series.iloc[i-window+1:i+1].values
                        x = np.arange(window)
                        if len(y) == window and not np.all(np.isnan(y)):
                            slope = np.polyfit(x, y, 1)[0]
                            slopes.append(slope)
                        else:
                            slopes.append(np.nan)
                return pd.Series(slopes, index=series.index)
            
            out['garch_vol_momentum'] = rolling_slope(out['garch_1d'], window=10)
            
            # 14. Shock indicator: standardized residual > 2 std
            out['garch_shock_indicator'] = (
                np.abs(out['garch_standardized_residual']) > 2.0
            ).astype(int)
            
            # 15. Model likelihood already calculated in _fit_garch_rolling
            # (no additional calculation needed - already in 'garch_log_likelihood')
            
        else:
            # If ARCH not available, create placeholder columns
            placeholder_cols = [
                'garch_1d', 'garch_5d', 'garch_20d', 'garch_persistence',
                'garch_long_run_variance', 'garch_log_likelihood', 
                'garch_standardized_residual', 'garch_ratio_1d_20d',
                'garch_spike_flag', 'garch_zscore', 'garch_residual_vol',
                'garch_vol_of_vol', 'garch_short_long_ratio', 
                'garch_vol_norm_20d', 'garch_vol_momentum',
                'garch_shock_indicator'
            ]
            for col in placeholder_cols:
                out[col] = np.nan
        
        # ============================================================
        # FILTER TO REQUESTED DATE RANGE
        # ============================================================
        if start:
            out = out[out.index >= pd.to_datetime(start)]
        if end:
            out = out[out.index <= pd.to_datetime(end)]
        
        out = out.dropna(how='all')
        
        # Ensure timezone-naive before to_nyse_close_index
        if hasattr(out.index, 'tz') and out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        
        out = to_nyse_close_index(out)
        
        # Remove timezone from result
        if hasattr(out.index, 'tz') and out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        
        out.attrs['provenance'] = {
            'source': source_name,
            'garch_model': 'GARCH(1,1)' if ARCH_AVAILABLE else 'proxy_only',
            'total_features': len(out.columns)
        }
        
        return out
        
    except Exception as e:
        import traceback
        print(f"Error in garch_iv.fetch for {symbol}: {e}")
        traceback.print_exc()
        return pd.DataFrame()
