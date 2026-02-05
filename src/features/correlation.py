import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import os

# Use universal fetcher (Tiingo/cache) only. yfinance fallback removed.

try:
    from src.dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
except Exception:  # pragma: no cover - fallback to identity
    def to_nyse_close_index(df: pd.DataFrame) -> pd.DataFrame:
        return df


# Default benchmark universe for correlation analysis
DEFAULT_BENCHMARKS: List[str] = [
    'SPY',     # S&P 500 ETF
    'QQQ',     # Nasdaq 100 ETF
    'UUP',     # DXY proxy (Dollar Index ETF)
    'VXX',     # VIX short-term futures ETF (Tiingo-compatible VIX proxy)
]

# Simple sector/industry mapping to include a sector ETF for added specificity
SECTOR_ETF_MAP: Dict[str, str] = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'SMH', 'AVGO': 'SOXX', 'AMD': 'SOXX',
    'GOOGL': 'XLC', 'GOOG': 'XLC', 'META': 'XLC', 'AMZN': 'XLY', 'TSLA': 'XLY',
    'JPM': 'XLF', 'GS': 'XLF', 'BAC': 'XLF', 'XOM': 'XLE', 'CVX': 'XLE',
}


YF_COL_ADJ = 'Adj Close'
YF_COL_CLOSE = 'Close'


def _pick_px(df: pd.DataFrame, sym: str) -> Optional[pd.Series]:
    """Pick price series for a symbol from yfinance download result, favoring Adj Close."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            cols = df[sym].columns if sym in df.columns.levels[0] else []
            col = YF_COL_ADJ if (YF_COL_ADJ in cols) else (YF_COL_CLOSE if (YF_COL_CLOSE in cols) else None)
            if col:
                return df[sym][col]
        else:
            col = YF_COL_ADJ if (YF_COL_ADJ in df.columns) else (YF_COL_CLOSE if (YF_COL_CLOSE in df.columns) else None)
            if col:
                return df[col]
        # Fallback: search tuple columns
        for col in df.columns:
            if isinstance(col, tuple) and col[0] == sym and col[1] in (YF_COL_ADJ, YF_COL_CLOSE):
                return df[col]
    except Exception:
        return None
    return None


def _download(symbols: List[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Download price data using EODHD provider (preferred) with universal fetcher fallback."""
    out = pd.DataFrame()
    
    # 🔥 TRY EODHD FIRST (best data quality for OHLCV)
    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider
        eodhd = get_eodhd_provider()
        
        if eodhd and eodhd.api_key:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            for sym in symbols:
                try:
                    # Use EODHD for full OHLCV data (returns DataFrame with OHLCV columns)
                    df = eodhd.get_eod_prices(sym, start_date=start, end_date=end, period='d')
                    if df is not None and not df.empty:
                        # EODHD returns proper datetime index already
                        # Use Close column (EODHD provides 'Close' capitalized after column mapping)
                        if 'Close' in df.columns:
                            out[sym] = df['Close']
                        elif 'Adj Close' in df.columns:
                            out[sym] = df['Adj Close']
                except Exception:
                    continue
            
            # If we got data from EODHD, return it
            if not out.empty:
                return out.dropna(how='all')
    except Exception:
        pass
    
    eodhd_only = os.environ.get("STAGE_B_EODHD_ONLY", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

    # FALLBACK: Universal fetcher (Tiingo)
    if eodhd_only:
        return out.dropna(how='all')

    try:
        from universal_data_fetcher import get_universal_fetcher
        fetcher = get_universal_fetcher()
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        for sym in symbols:
            if sym in out.columns:  # Skip if already fetched from EODHD
                continue
            try:
                # Use download method with period parameter
                df = fetcher.download(sym, period='5y')
                if df is not None and not df.empty:
                    # 🔥 FIX: Universal fetcher returns Date as a column, not index
                    if 'Date' in df.columns:
                        df = df.set_index('Date')
                        df.index = pd.to_datetime(df.index)
                    elif not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    # Filter by date range if specified
                    if start:
                        df = df[df.index >= pd.Timestamp(start)]
                    if end:
                        df = df[df.index <= pd.Timestamp(end)]
                    
                    # Use Close column
                    if 'Close' in df.columns:
                        out[sym] = df['Close']
                    elif 'close' in df.columns:
                        out[sym] = df['close']
            except Exception:
                continue
    except Exception:
        pass
    
    return out.dropna(how='all')


def fetch(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    windows: List[int] = [20, 60],  # 🔥 HEDGE-FUND: Only 2 windows (fast + medium)
    add_break_flags: bool = False,  # 🔥 DEPRECATED: Binary flags removed (use z-scores)
    low_break: float = 0.2,
    high_break: float = 0.8,
    flat_gate_n: int = 0,
    flat_eps: float = 1e-6,
    add_advanced_features: bool = True,  # 🔥 Alpha engines enabled by default
    lag_periods: List[int] = [1, 5],  # 🔥 HEDGE-FUND: Only 1d + 5d lags (shock + weekly)
    trend_lookback: int = 5,  # 🔥 Correlation momentum lookback
    vol_window: int = 20,  # 🔥 Correlation volatility window
) -> pd.DataFrame:
    """
    HEDGE-FUND GRADE correlation features (COMPRESSED from 118 → ~27-30 features).
    
    🔥 KEY CHANGES FROM V1:
    - Binary break flags REMOVED (replaced with continuous z-scores)
    - Windows reduced: 3 → 2 (20d fast, 60d medium; 126d dropped)
    - Lag periods: [1,2,5,10] → [1,5] (shock + weekly only)
    - Selective benchmarks: Only essential correlations computed
    
    PHILOSOPHY:
    Correlation features are META-SIGNALS that condition risk, timing, and confidence.
    They must be smooth, continuous, low-dimensional, and early-warning focused.
    
    FEATURES (~27-30 total):
    
    1. TRADITIONAL ROLLING CORRELATIONS (5 features) - Essential beta signals
       - CORR_20_SPY: Fast S&P 500 beta
       - CORR_60_SPY: Medium S&P 500 beta
       - CORR_20_QQQ: Tech/growth exposure
       - CORR_20_VXX: Volatility/fear exposure (regime-critical)
       - CORR_20_Sector: Sector beta (e.g., XLK for AAPL)
    
    2. DECOUPLING Z-SCORE (1 feature) - Replaces 30 binary flags
       - CORR_DECOUPLING_Z: Z-score of CORR_20_SPY vs long-run mean
         (< -1.5 = strong decoupling, > +1.5 = strong fusion)
    
    3. LAGGED CROSS-CORRELATION (4 features) - Shock transmission
       - LAG_CORR_1_SPY: 1-day SPY shock propagation
       - LAG_CORR_5_SPY: 5-day SPY spillover (weekly cycle)
       - LAG_CORR_1_VIX: 1-day VIX shock (fear contagion)
       - LAG_CORR_2_VIX: 2-day VIX spillover
    
    4. CORRELATION MOMENTUM/TREND (3 features) - Regime detector
       - CORR_20_SPY_TREND: SPY correlation slope (fusion vs divergence)
       - CORR_20_QQQ_TREND: QQQ correlation slope
       - CORR_20_VXX_TREND: VIX correlation slope
    
    5. CORRELATION VOLATILITY (2 features) - Regime stability
       - CORR_20_SPY_VOL: SPY correlation volatility
       - CORR_20_VXX_VOL: VIX correlation volatility
    
    6. CORRELATION SPREAD (2 features) - Transition detector
       - CORR_SPREAD_20_60_SPY: Short-term vs medium-term SPY correlation
       - CORR_SPREAD_20_60_QQQ: Short-term vs medium-term QQQ correlation
    
    7. CROSS-FEATURE CORRELATIONS (3 features) - Microstructure
       - CORR_RETURN_VOL_20: Return vs Volume (accumulation/distribution)
       - CORR_RETURN_RANGE_10: Return vs Range (trend/compression)
       - CORR_VOL_VOLATILITY_20: Volume vs Volatility (panic/stealth)
    
    8. VIX LAG CORRELATIONS (4 features) - Volatility spillover
       - CORR_20_VIX_lag1, CORR_60_VIX_lag1: 1-day VIX lag
       - CORR_20_VIX_lag2, CORR_60_VIX_lag2: 2-day VIX lag
    
    9. AUTOCORRELATION (4 features) - Internal dynamics
       - ACF_RET_1: 1-day return autocorrelation (momentum vs reversion)
       - ACF_RET_5: 5-day return autocorrelation (weekly pattern)
       - ACF_ABSRET_1: Volatility clustering
       - ACF_VOL_1: Volatility persistence
    
    Data sources:
      - EODHD (preferred) for OHLCV data
      - Universal fetcher (Tiingo) for benchmark indices
    
    Returns an empty DataFrame on failure (resilient by design).
    """
    benchmarks: List[str] = list(DEFAULT_BENCHMARKS)
    sector = SECTOR_ETF_MAP.get(symbol.upper())
    if sector:
        benchmarks.append(sector)

    uniq_bench = sorted(set(benchmarks + [symbol]))
    px = _download(uniq_bench, start, end)
    if px.empty or symbol not in px.columns:
        return pd.DataFrame()

    # Align to NYSE close and compute daily returns
    px = to_nyse_close_index(px.sort_index())
    rets = px.pct_change().dropna(how='all')
    if rets.empty:
        return pd.DataFrame()

    sym_ret = rets[symbol].rename('ret')
    out = pd.DataFrame(index=rets.index)

    # ========================================================================
    # 1. TRADITIONAL ROLLING CORRELATIONS (5 essential features)
    # ========================================================================
    # HEDGE-FUND: Only compute essential correlations, not exhaustive grid
    
    essential_correlations = [
        ('SPY', 20),      # Fast S&P 500 beta
        ('SPY', 60),      # Medium S&P 500 beta
        ('QQQ', 20),      # Tech/growth exposure (fast)
        ('QQQ', 60),      # Tech/growth exposure (medium) - REQUIRED for CORR_SPREAD_20_60_QQQ
        ('VXX', 20),      # Volatility/fear (regime-critical)
    ]
    
    # Add sector correlation if applicable. Always emit a schema-stable sector column
    # even when the sector ETF is unknown/unavailable.
    sector_available = bool(sector) and (sector in rets.columns)
    if sector_available:
        essential_correlations.append((sector, 20))
    
    for b, w in essential_correlations:
        if b not in rets.columns:
            continue
        b_ret = rets[b]
        corr = sym_ret.rolling(w, min_periods=max(5, w // 3)).corr(b_ret)
        name = f"CORR_{w}_{'SECTOR' if (sector_available and b == sector) else b.replace('^','')}"
        out[name] = corr

    # Ensure the sector correlation column is always present for downstream schema alignment.
    if 'CORR_20_SECTOR' not in out.columns:
        out['CORR_20_SECTOR'] = 0.0
    
    # ========================================================================
    # 2. DECOUPLING Z-SCORE (1 feature) - Replaces binary break flags
    # ========================================================================
    # HEDGE-FUND: Continuous signal, not binary (better for ML gradients)
    if 'CORR_20_SPY' in out.columns:
        corr_20_spy = out['CORR_20_SPY']
        # Z-score vs 252-day (1-year) rolling mean
        corr_mean = corr_20_spy.rolling(252, min_periods=60).mean()
        corr_std = corr_20_spy.rolling(252, min_periods=60).std()
        out['CORR_DECOUPLING_Z'] = (corr_20_spy - corr_mean) / (corr_std + 1e-9)
        # Negative z-score = decoupling, Positive = fusion

    # ========================================================================
    # 3. ALPHA ENGINE: LAGGED CROSS-CORRELATION (4 features)
    # ========================================================================
    # HEDGE-FUND: Only 1d + 5d lags for SPY/VIX (shock + weekly spillover)
    if add_advanced_features:
        lag_correlations = [
            ('SPY', 1),   # 1-day SPY shock propagation
            ('SPY', 5),   # 5-day SPY weekly spillover
            ('VXX', 1),   # 1-day VIX shock (fear contagion)
            ('VXX', 2),   # 2-day VIX spillover
        ]
        
        for b, lag in lag_correlations:
            if b not in rets.columns:
                continue
            b_ret = rets[b]
            b_name = 'SECTOR' if (sector_available and b == sector) else b.replace('^','')
            
            # Correlation between stock_return_t and benchmark_return_t-lag
            b_ret_lagged = b_ret.shift(lag)
            lag_corr = sym_ret.rolling(20, min_periods=10).corr(b_ret_lagged)
            # HEDGE-FUND: Winsorize to [-0.95, 0.95] (protect Mamba state during crashes)
            out[f"LAG_CORR_{lag}_{b_name}"] = lag_corr.clip(-0.95, 0.95)

    # ========================================================================
    # 4. ALPHA ENGINE: CORRELATION MOMENTUM/TREND (3 features)
    # ========================================================================
    # HEDGE-FUND: Only for essential benchmarks (SPY, QQQ, VXX)
    if add_advanced_features:
        trend_correlations = ['CORR_20_SPY', 'CORR_20_QQQ', 'CORR_20_VXX']
        
        # Linear regression slope calculator
        x = np.arange(trend_lookback)
        x_mean = x.mean()
        x_var = x.var()
        
        def calc_slope(window):
            if len(window) < trend_lookback:
                return np.nan
            y = window.values
            y_mean = y.mean()
            cov = ((x - x_mean) * (y - y_mean)).sum()
            return cov / x_var if x_var > 0 else 0.0
        
        for cname in trend_correlations:
            if cname in out.columns:
                out[f"{cname}_TREND"] = out[cname].rolling(trend_lookback).apply(calc_slope, raw=False)

    # ========================================================================
    # 5. ALPHA ENGINE: CORRELATION VOLATILITY (2 features)
    # ========================================================================
    # HEDGE-FUND: Only SPY and VXX volatility (regime stability indicators)
    if add_advanced_features:
        vol_correlations = ['CORR_20_SPY', 'CORR_20_VXX']
        
        for cname in vol_correlations:
            if cname in out.columns:
                # Rolling standard deviation of the correlation itself
                out[f"{cname}_VOL"] = out[cname].rolling(vol_window, min_periods=10).std()

    # ========================================================================
    # 6. ALPHA ENGINE: CORRELATION SPREAD (2 features)
    # ========================================================================
    # HEDGE-FUND: Only SPY and QQQ spreads (20d - 60d)
    if add_advanced_features:
        spread_pairs = [
            ('CORR_20_SPY', 'CORR_60_SPY', 'SPY'),
            ('CORR_20_QQQ', 'CORR_60_QQQ', 'QQQ'),
        ]
        
        for short_name, long_name, bench in spread_pairs:
            if short_name in out.columns and long_name in out.columns:
                out[f"CORR_SPREAD_20_60_{bench}"] = out[short_name] - out[long_name]

    # ========================================================================
    # 7. ALPHA ENGINE: CROSS-FEATURE CORRELATIONS (3 features) - KEEP ALL
    # ========================================================================
    if add_advanced_features:
        # Get OHLCV data from EODHD for volume, range calculations
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            eodhd = get_eodhd_provider()
            
            if eodhd and eodhd.api_key:
                # Get full OHLCV data (returns DataFrame with Open, High, Low, Close, Volume)
                ohlcv = eodhd.get_eod_prices(symbol, start_date=start, end_date=end, period='d')
                
                if ohlcv is not None and not ohlcv.empty:
                    # EODHD already returns proper datetime index
                    # Normalize timezone for alignment
                    if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
                        ohlcv.index = ohlcv.index.tz_localize(None)
                    
                    # Ensure rets index is also timezone-naive for proper alignment
                    rets_idx = rets.index
                    if hasattr(rets_idx, 'tz') and rets_idx.tz is not None:
                        rets_idx = rets_idx.tz_localize(None)
                        rets.index = rets_idx
                        sym_ret.index = rets_idx
                        # IMPORTANT: Also remove timezone from out.index for proper loc assignment
                        out.index = rets_idx
                    
                    # Volume-based correlations (EODHD uses 'Volume' capitalized)
                    if 'Volume' in ohlcv.columns:
                        volume = ohlcv['Volume']
                        # Align indices
                        common_idx = rets_idx.intersection(volume.index)
                        if len(common_idx) > 20:
                            vol_aligned = volume.reindex(common_idx)
                            ret_aligned = sym_ret.reindex(common_idx)
                            
                            # Correlation between returns and volume
                            corr_vol = ret_aligned.rolling(20, min_periods=10).corr(vol_aligned)
                            out.loc[common_idx, 'CORR_RETURN_VOL_20'] = corr_vol
                    
                    # Range-based correlations (EODHD uses capitalized column names)
                    if 'High' in ohlcv.columns and 'Low' in ohlcv.columns and 'Close' in ohlcv.columns:
                        # Daily range (high - low) / close
                        daily_range = (ohlcv['High'] - ohlcv['Low']) / ohlcv['Close']
                        
                        # Align indices
                        common_idx = rets_idx.intersection(daily_range.index)
                        if len(common_idx) > 10:
                            range_aligned = daily_range.reindex(common_idx)
                            ret_aligned = sym_ret.reindex(common_idx)
                            
                            corr_range = ret_aligned.rolling(10, min_periods=5).corr(range_aligned)
                            out.loc[common_idx, 'CORR_RETURN_RANGE_10'] = corr_range
                    
                    # Volatility cross-correlations
                    if 'Volume' in ohlcv.columns:
                        # Volatility (rolling std of returns)
                        volatility = sym_ret.rolling(20, min_periods=10).std()
                        
                        # Align volume
                        common_idx = volatility.index.intersection(ohlcv.index)
                        if len(common_idx) > 20:
                            vol_data = ohlcv['Volume'].reindex(common_idx)
                            vol_metric = volatility.reindex(common_idx)
                            
                            corr_vol_volatility = vol_data.rolling(20, min_periods=10).corr(vol_metric)
                            out.loc[common_idx, 'CORR_VOL_VOLATILITY_20'] = corr_vol_volatility
        except Exception as e:
            # Cross-feature correlations are optional - don't fail if unavailable
            pass

    # ========================================================================
    # 8. ALPHA ENGINE: VIX LAG CORRELATIONS (4 features) - KEEP ALL
    # ========================================================================
    if add_advanced_features:
        # VXX is already in benchmarks as VIX proxy.
        #
        # IMPORTANT (schema stability): always emit the same VIX-lag column set
        # even when VXX data is unavailable. Downstream merges frequently drop
        # all-NaN columns; using a neutral constant preserves column alignment.
        vxx_available = 'VXX' in rets.columns
        if vxx_available:
            vxx_ret = rets['VXX']
            for lag in [1, 2]:
                vxx_lagged = vxx_ret.shift(lag)
                for w in [20, 60]:
                    vix_lag_corr = sym_ret.rolling(w, min_periods=max(5, w//3)).corr(vxx_lagged)
                    # HEDGE-FUND: Winsorize to [-0.95, 0.95] (VIX spikes during crashes)
                    out[f"CORR_{w}_VIX_lag{lag}"] = vix_lag_corr.clip(-0.95, 0.95)
        else:
            for lag in [1, 2]:
                for w in [20, 60]:
                    out[f"CORR_{w}_VIX_lag{lag}"] = 0.0

    # ========================================================================
    # 9. ALPHA ENGINE: AUTO-CORRELATION (4 features) - Trim redundancy
    # ========================================================================
    # HEDGE-FUND: Keep ACF_RET_1, ACF_RET_5, ACF_ABSRET_1, ACF_VOL_1
    # DROP: ACF_RET_2, ACF_ABSRET_2 (redundant)
    if add_advanced_features:
        # Returns autocorrelation (trend/mean reversion)
        for lag in [1, 5]:  # Only 1d and 5d
            sym_ret_lagged = sym_ret.shift(lag)
            acf_ret = sym_ret.rolling(20, min_periods=10).corr(sym_ret_lagged)
            # HEDGE-FUND: Winsorize to [-0.95, 0.95]
            out[f"ACF_RET_{lag}"] = acf_ret.clip(-0.95, 0.95)
        
        # Absolute returns autocorrelation (volatility clustering)
        abs_ret = sym_ret.abs()
        abs_ret_lagged = abs_ret.shift(1)  # Only 1-day lag
        acf_absret = abs_ret.rolling(20, min_periods=10).corr(abs_ret_lagged)
        # HEDGE-FUND: Winsorize to [-0.95, 0.95]
        out[f"ACF_ABSRET_1"] = acf_absret.clip(-0.95, 0.95)
        
        # Volatility autocorrelation (volatility persistence)
        volatility = sym_ret.rolling(10, min_periods=5).std()
        vol_lagged = volatility.shift(1)
        acf_vol = volatility.rolling(20, min_periods=10).corr(vol_lagged)
        # HEDGE-FUND: Winsorize to [-0.95, 0.95]
        out[f"ACF_VOL_1"] = acf_vol.clip(-0.95, 0.95)

    out = out.dropna(how='all')

    # Attach provenance and telemetry
    out.attrs['provenance'] = {
        'benchmarks': ['SPY', 'QQQ', 'VXX', sector] if sector else ['SPY', 'QQQ', 'VXX'],
        'returns': 'pct_change(Adj Close/Close)',
        'hedge_fund_grade': True,
        'compressed_from': 118,
        'alpha_engines': {
            'traditional_correlations': 5,
            'decoupling_z_score': 1,
            'lagged_cross_corr': 4,
            'corr_momentum': 3,
            'corr_volatility': 2,
            'corr_spread': 2,
            'cross_feature_corr': 3,
            'vix_lag_corr': 4,
            'autocorrelation': 4,
        }
    }
    # HEDGE-FUND GOVERNANCE: Feature role tagging (for debugging, audits, regime analysis)
    out.attrs['feature_roles'] = {
        'causal_lagged': [  # Time-shifted features (t-1, t-2, etc.) - causal inference
            'LAG_CORR_1_SPY', 'LAG_CORR_5_SPY', 'LAG_CORR_1_VXX', 'LAG_CORR_2_VXX',
            'CORR_20_VIX_lag1', 'CORR_60_VIX_lag1', 'CORR_20_VIX_lag2', 'CORR_60_VIX_lag2',
            'ACF_RET_1', 'ACF_RET_5', 'ACF_ABSRET_1', 'ACF_VOL_1'
        ],
        'contemporaneous': [  # Same-period correlations (t vs t) - regime detection
            'CORR_20_SPY', 'CORR_60_SPY', 'CORR_20_QQQ', 'CORR_20_VXX', 'CORR_20_SECTOR',
            'CORR_20_SPY_TREND', 'CORR_20_QQQ_TREND', 'CORR_20_VXX_TREND',
            'CORR_20_SPY_VOL', 'CORR_20_VXX_VOL',
            'CORR_SPREAD_20_60_SPY', 'CORR_SPREAD_20_60_QQQ',
            'CORR_RETURN_VOL_20', 'CORR_RETURN_RANGE_10', 'CORR_VOL_VOLATILITY_20'
        ],
        'structural': [  # Meta-regime indicators (z-scores, deviations) - structural breaks
            'CORR_DECOUPLING_Z'
        ]
    }
    out.attrs['telemetry'] = {
        'windows': windows,
        'binary_flags_removed': True,
        'continuous_z_scores': True,
        'lagged_correlations_winsorized': True,  # HEDGE-FUND: Clipped to [-0.95, 0.95]
        'winsorization_bounds': [-0.95, 0.95],
        'advanced_features': {
            'enabled': add_advanced_features,
            'lag_periods': lag_periods if add_advanced_features else None,
            'trend_lookback': trend_lookback if add_advanced_features else None,
            'vol_window': vol_window if add_advanced_features else None,
        },
        'vxx_available': bool('VXX' in rets.columns) if add_advanced_features else None,
        'rows': int(out.shape[0]),
        'cols': int(out.shape[1])
    }
    
    # Add governance columns - required for all families
    out['correlation_has_data'] = 1.0
    out['correlation_activity'] = 1.0
    out['correlation_days_since_update'] = 0.0
    
    return out


__all__ = ['fetch']
