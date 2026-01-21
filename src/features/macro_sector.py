import pandas as pd
# yfinance removed: rely on FRED/Tiingo/universal fetcher only
from typing import Optional
from pathlib import Path
try:
    from dcf_lab.utils.timealign import lag_if_post_close, to_nyse_close_index  # type: ignore
except Exception:
    try:
        from src.dcf_lab.utils.timealign import lag_if_post_close, to_nyse_close_index  # type: ignore
    except Exception:
        def lag_if_post_close(s, days=1):  # noqa: ARG001 - keep signature compatibility
            return s
        def to_nyse_close_index(df):
            return df

# Prefer free macro sources via FRED, fallback to Yahoo when needed
try:
    from src.dcf_lab.providers.fred_provider import FREDProvider
except ImportError:
    try:
        from dcf_lab.providers.fred_provider import FREDProvider
    except Exception:  # safe import guard for environments without package path
        FREDProvider = None  # type: ignore

# Universal data fetcher (Tiingo/cache instead of yfinance)
try:
    from src.dcf_lab.data.universal_data_fetcher import get_universal_fetcher
except ImportError:
    try:
        from dcf_lab.data.universal_data_fetcher import get_universal_fetcher
    except ImportError:
        get_universal_fetcher = None


def _macro_from_fred(start: Optional[str], end: Optional[str], lag_days: int = 1) -> pd.DataFrame:
    if FREDProvider is None:
        return pd.DataFrame()
    
    # Try cache-first approach
    cache_dir = Path("data/cache/fred")
    
    def _load_cached_series(series_id: str) -> Optional[pd.DataFrame]:
        cache_path = cache_dir / f"{series_id}.parquet"
        if cache_path.exists():
            try:
                data = pd.read_parquet(cache_path)
                # Filter by date range if needed
                if start or end:
                    if start:
                        start_ts = pd.Timestamp(start)
                        data = data[data.index >= start_ts]
                    if end:
                        end_ts = pd.Timestamp(end)
                        data = data[data.index <= end_ts]
                return data
            except Exception:
                # Silently fail - return None to trigger fresh fetch
                pass
        return None
    
    try:
        fred = FREDProvider()
        out = pd.DataFrame()
        
        # Load yield curve data (with cache fallback)
        yield_data = {}
        for series in ["DGS10", "DGS3MO", "T10YIE"]:
            cached = _load_cached_series(series)
            if cached is not None and not cached.empty:
                print(f"📂 Using cached {series}")
                yield_data[series] = cached[series] if series in cached.columns else cached.iloc[:, 0]
            else:
                try:
                    print(f"📡 Fetching {series} from FRED (cache miss)...")
                    fresh_data = fred.get_series_data(series, start_date=start, end_date=end, frequency='d')
                    if not fresh_data.empty and series in fresh_data.columns:
                        yield_data[series] = fresh_data[series]
                        # Cache for future use
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        fresh_data.to_parquet(cache_dir / f"{series}.parquet")
                except Exception as e:
                    print(f"⚠️ Failed to fetch {series}: {e}")
        
        # Calculate REAL_YIELD_SLOPE_PROXY if we have the data
        if all(k in yield_data for k in ["DGS10", "DGS3MO"]):
            real_slope = (yield_data["DGS10"] - yield_data["DGS3MO"]) / 100.0
            breakeven = (yield_data.get("T10YIE", 0) / 100.0) if "T10YIE" in yield_data else 0.0
            
            # Align indices
            combined_index = real_slope.index
            if isinstance(breakeven, pd.Series):
                combined_index = real_slope.index.intersection(breakeven.index)
                real_slope = real_slope.loc[combined_index]
                breakeven = breakeven.loc[combined_index]
            
            out['REAL_YIELD_SLOPE_PROXY'] = (real_slope - breakeven).dropna()
        
        # Load dollar index data (with cache fallback)
        dxy_cached = _load_cached_series("DTWEXBGS")
        if dxy_cached is not None and not dxy_cached.empty:
            print("📂 Using cached DTWEXBGS")
            dxy_col = "DTWEXBGS" if "DTWEXBGS" in dxy_cached.columns else dxy_cached.columns[0]
            out['USD_PRESSURE'] = dxy_cached[dxy_col].pct_change()
        else:
            try:
                print("📡 Fetching DTWEXBGS from FRED (cache miss)...")
                dxy_df = fred.get_series_data("DTWEXBGS", start_date=start, end_date=end, frequency='d')
                if not dxy_df.empty and 'DTWEXBGS' in dxy_df.columns:
                    out['USD_PRESSURE'] = dxy_df['DTWEXBGS'].pct_change()
                    # Cache for future use
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    dxy_df.to_parquet(cache_dir / "DTWEXBGS.parquet")
            except Exception as e:
                print(f"⚠️ Failed to fetch DTWEXBGS: {e}")

        # FRED series often publish after close: apply lag and align to NYSE date
        out = to_nyse_close_index(out)
        if 'REAL_YIELD_SLOPE_PROXY' in out.columns:
            out['REAL_YIELD_SLOPE_PROXY'] = lag_if_post_close(out['REAL_YIELD_SLOPE_PROXY'], lag_days)
        if 'USD_PRESSURE' in out.columns:
            out['USD_PRESSURE'] = lag_if_post_close(out['USD_PRESSURE'], lag_days)
        return out
        
    except Exception:
        # Silently fail for missing data - this is expected for some date ranges
        return pd.DataFrame()


def _macro_from_yahoo(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Fetch macro data via universal fetcher (Tiingo/cache) instead of direct yfinance."""
    try:
        # Use universal fetcher if available; do NOT fallback to yfinance
        if get_universal_fetcher is not None:
            fetcher = get_universal_fetcher()
            if fetcher is not None:
                # Fetch each symbol individually via universal fetcher
                symbols = ["^TNX", "^IRX", "DXY", "TIP", "IEF", "UUP"]
                data_dict = {}
                for sym in symbols:
                    try:
                        df = fetcher.download(sym, period='2y', interval='1d')
                        if df is not None and not df.empty:
                            # Normalize columns
                            df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
                            # Filter to date range
                            if start:
                                df = df[df.index >= pd.to_datetime(start)]
                            if end:
                                df = df[df.index <= pd.to_datetime(end)]
                            data_dict[sym] = df
                    except Exception:
                        continue
                
                # Build combined DataFrame similar to yfinance group_by='ticker'
                if data_dict:
                    def col(t):
                        if t in data_dict and 'close' in data_dict[t].columns:
                            return data_dict[t]['close']
                        return pd.Series(dtype=float)
                    
                    out = pd.DataFrame()
                    try:
                        tnx = col("^TNX")
                        irx = col("^IRX")
                        tip = col("TIP")
                        ief = col("IEF")
                        if not tip.empty and not ief.empty:
                            breakeven_proxy = (tip / ief).pct_change().rolling(20).mean()
                        else:
                            breakeven_proxy = pd.Series(0.0, index=tnx.index if not tnx.empty else irx.index)
                        
                        idx = tnx.index if not tnx.empty else irx.index
                        out = pd.DataFrame(index=idx)
                        if not tnx.empty and not irx.empty:
                            out['REAL_YIELD_SLOPE_PROXY'] = ((tnx - irx) / 100.0) - breakeven_proxy
                    except Exception:
                        pass
                    
                    # USD pressure: prefer DXY; fallback UUP ETF if DXY unavailable
                    for t in ("DXY", "UUP"):
                        try:
                            usd = col(t)
                            if not usd.empty:
                                if out.index.empty:
                                    out = pd.DataFrame(index=usd.index)
                                out['USD_PRESSURE'] = usd.pct_change()
                                break
                        except Exception:
                            continue
                    
                    # Align to NYSE date for consistency
                    return to_nyse_close_index(out)
        
        # If universal fetcher path did not return data, return empty (no yfinance fallback)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch(start: Optional[str] = None, end: Optional[str] = None, lag_days: int = 1) -> pd.DataFrame:
    """
    Macro–sector proxies with FRED-first strategy and Yahoo fallback:
      - REAL_YIELD_SLOPE_PROXY: (DGS10 - DGS3MO)/100 - T10YIE/100 (breakeven proxy)
      - USD_PRESSURE: pct_change of broad dollar index (DTWEXBGS); fallback to Yahoo DXY
    """
    fred_df = _macro_from_fred(start, end, lag_days=lag_days)
    # Determine if we still need any signals
    need_real = ('REAL_YIELD_SLOPE_PROXY' not in fred_df.columns) or fred_df['REAL_YIELD_SLOPE_PROXY'].dropna().empty
    need_usd = ('USD_PRESSURE' not in fred_df.columns) or fred_df['USD_PRESSURE'].dropna().empty
    if need_real or need_usd:
        yahoo_df = _macro_from_yahoo(start, end)
        if fred_df.empty and yahoo_df.empty:
            return pd.DataFrame()
        if fred_df.empty:
            out = yahoo_df.dropna(how='all')
            out.attrs['provenance'] = {'source': 'Yahoo'}
            out.attrs['telemetry'] = {'status': 'ok', 'source': 'Yahoo', 'proxy': False}
            return out
        if yahoo_df.empty:
            out = fred_df.dropna(how='all')
            out.attrs['provenance'] = {'source': 'FRED'}
            out.attrs['telemetry'] = {'status': 'ok', 'source': 'FRED', 'proxy': False}
            return out
        combined = fred_df.combine_first(yahoo_df)
        out = combined.dropna(how='all')
        out.attrs['provenance'] = {'source': 'FRED+Yahoo'}
        out.attrs['telemetry'] = {'status': 'ok', 'source': 'FRED+Yahoo', 'proxy': False}
        return out
    # All signals satisfied by FRED
    out = fred_df.dropna(how='all')
    out.attrs['provenance'] = {'source': 'FRED'}
    out.attrs['telemetry'] = {'status': 'ok', 'source': 'FRED', 'proxy': False}
    return out
