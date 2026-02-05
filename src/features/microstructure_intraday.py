import os
import time
import logging
import pandas as pd
import numpy as np
# yfinance removed: rely on Tiingo/AlphaVantage intraday providers only
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def _maybe_add_eodhd_quote_features(df: pd.DataFrame, symbol: str, end: Optional[str]) -> pd.DataFrame:
    """Optionally add top-of-book quote features (snapshot) onto the last row.

    This uses EODHD Live v2 delayed quote snapshot when enabled. Because the quote
    endpoint is *not historical*, we gate it to only run for near-real-time ranges
    (default: end within 2 days of today) to avoid accidental leakage during backtests.
    """

    try:
        if os.getenv("MICROSTRUCTURE_USE_EODHD_QUOTES", "0").strip() != "1":
            return df

        if df is None or df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        try:
            max_age_days = int(os.getenv("MICROSTRUCTURE_QUOTES_MAX_AGE_DAYS", "2"))
        except Exception:
            max_age_days = 2

        # Only allow quote snapshot when end date is recent.
        if end:
            end_dt = pd.to_datetime(end, errors="coerce")
            if pd.isna(end_dt):
                return df
            end_dt = end_dt.normalize()
            today = pd.Timestamp.utcnow().normalize()
            if abs(int((today - end_dt).days)) > max_age_days:
                return df

        from src.data_sources.eodhd_provider import get_eodhd_provider

        eodhd = get_eodhd_provider()
        if not getattr(eodhd, "api_key", None):
            return df

        qdf = eodhd.get_us_quote_delayed(symbol)
        if qdf is None or qdf.empty:
            return df

        # Normalize symbol key.
        sym_key = symbol if "." in symbol else f"{symbol}.US"
        if sym_key not in qdf.index and sym_key.upper() in qdf.index:
            sym_key = sym_key.upper()
        if sym_key not in qdf.index:
            # Fallback to first row if batch normalization differs.
            sym_key = str(qdf.index[0])

        row = qdf.loc[sym_key]

        # Create quote feature columns (zeros by default to keep schema stable).
        out = df.copy()
        cols = [
            "micro_quote_bid_price",
            "micro_quote_ask_price",
            "micro_quote_bid_size",
            "micro_quote_ask_size",
            "micro_quote_mid_price",
            "micro_quote_spread_abs",
            "micro_quote_spread_bps",
            "micro_quote_imbalance",
            "micro_quote_has_data",
        ]
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0

        last_idx = out.index.max()

        bid_p = float(row.get("bid_price", np.nan))
        ask_p = float(row.get("ask_price", np.nan))
        bid_s = float(row.get("bid_size", np.nan))
        ask_s = float(row.get("ask_size", np.nan))

        mid = (bid_p + ask_p) / 2.0 if np.isfinite(bid_p) and np.isfinite(ask_p) else np.nan
        spread = (ask_p - bid_p) if np.isfinite(bid_p) and np.isfinite(ask_p) else np.nan
        spread_bps = (1e4 * spread / mid) if np.isfinite(spread) and np.isfinite(mid) and mid != 0 else np.nan

        denom = bid_s + ask_s
        imbalance = (bid_s - ask_s) / denom if np.isfinite(denom) and denom != 0 else np.nan

        out.loc[last_idx, "micro_quote_bid_price"] = 0.0 if not np.isfinite(bid_p) else bid_p
        out.loc[last_idx, "micro_quote_ask_price"] = 0.0 if not np.isfinite(ask_p) else ask_p
        out.loc[last_idx, "micro_quote_bid_size"] = 0.0 if not np.isfinite(bid_s) else bid_s
        out.loc[last_idx, "micro_quote_ask_size"] = 0.0 if not np.isfinite(ask_s) else ask_s
        out.loc[last_idx, "micro_quote_mid_price"] = 0.0 if not np.isfinite(mid) else float(mid)
        out.loc[last_idx, "micro_quote_spread_abs"] = 0.0 if not np.isfinite(spread) else float(spread)
        out.loc[last_idx, "micro_quote_spread_bps"] = 0.0 if not np.isfinite(spread_bps) else float(spread_bps)
        out.loc[last_idx, "micro_quote_imbalance"] = 0.0 if not np.isfinite(imbalance) else float(imbalance)
        out.loc[last_idx, "micro_quote_has_data"] = 1.0

        try:
            out.attrs.setdefault("provenance", {})
            out.attrs["provenance"].update({"quote_source": "eodhd.us-quote-delayed", "quote_snapshot": True})
        except Exception:
            pass

        return out
    except Exception:
        return df

try:
    from src.dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
except Exception:
    try:
        from dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
    except Exception:
        def to_nyse_close_index(df):
            return df

try:
    from src.dcf_lab.providers.alpha_vantage_provider import AlphaVantageProvider
except ImportError:
    try:
        from dcf_lab.providers.alpha_vantage_provider import AlphaVantageProvider
    except Exception:
        AlphaVantageProvider = None  # type: ignore
        
try:
    from src.dcf_lab.utils import rate as rate_utils  # type: ignore
except ImportError:
    try:
        from dcf_lab.utils import rate as rate_utils  # type: ignore
    except Exception:
        rate_utils = None  # type: ignore


def _uptick_downtick_flags(prices: pd.Series) -> pd.Series:
    diff = prices.diff().fillna(0)
    flags = pd.Series(0, index=prices.index)
    flags[diff > 0] = 1
    flags[diff < 0] = -1
    return flags


def _build_daily_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    close = df['close'] if 'close' in df.columns else df['Close']
    vol = df['volume'] if 'volume' in df.columns else df['Volume']
    vol = vol.replace(0, np.nan)
    flags = _uptick_downtick_flags(close)
    tmp = pd.DataFrame({'ret1m': close.pct_change(), 'flag': flags, 'vol': vol})
    tmp['ofi'] = np.sign(tmp['ret1m']).fillna(0) * tmp['vol'].fillna(0)
    grouped = tmp.groupby(tmp.index.date)
    daily = pd.DataFrame()
    daily['OFI_PROXY'] = grouped['ofi'].sum()
    up_vol = tmp.loc[tmp['flag'] > 0, 'vol'].groupby(tmp.loc[tmp['flag'] > 0].index.date).sum()
    dn_vol = tmp.loc[tmp['flag'] < 0, 'vol'].groupby(tmp.loc[tmp['flag'] < 0].index.date).sum()
    tot_vol = grouped['vol'].sum()
    daily['UPDOWN_IMBALANCE'] = ((up_vol - dn_vol) / (tot_vol + 1e-9)).reindex(daily.index)
    intraday_vol = grouped['ret1m'].std()
    daily['MICRO_LIQ_SLOPE'] = (intraday_vol / (tot_vol ** 0.5)).reindex(daily.index)
    minutes = grouped.size()
    zero_minutes = tmp['vol'].fillna(0).eq(0).groupby(tmp.index.date).sum()
    daily['STALLED_PRINTS_RATIO'] = (zero_minutes / (minutes + 1e-9)).reindex(daily.index)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    # Align to NYSE close calendar (UTC midnight per NYSE date)
    return to_nyse_close_index(daily)


def _aliases(sym: str) -> List[str]:
    """Return common symbol aliases to improve provider hit-rate.
    Examples: GOOGL<->GOOG, dot/dash variants like BRK.B<->BRK-B
    """
    al: List[str] = [sym]
    su = sym.upper()
    if su == 'GOOGL':
        al.append('GOOG')
    elif su == 'GOOG':
        al.append('GOOGL')
    if '.' in sym:
        al.append(sym.replace('.', '-'))
    if '-' in sym:
        al.append(sym.replace('-', '.'))
    # dedupe preserving order
    seen = set()
    out: List[str] = []
    for s in al:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _apply_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if start:
        start_dt = pd.to_datetime(start)
        # Make timezone-aware if index is timezone-aware
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
        df = df[df.index >= start_dt]
    if end:
        end_dt = pd.to_datetime(end)
        # Make timezone-aware if index is timezone-aware
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            end_dt = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
        df = df[df.index <= end_dt]
    return df


def _attach_attrs(df: pd.DataFrame, provenance: Dict, telemetry: Optional[Dict] = None) -> None:
    try:
        df.attrs['provenance'] = provenance
        if telemetry:
            df.attrs['telemetry'] = telemetry
    except Exception:
        pass


def _within_days(start: Optional[str], end: Optional[str], max_days: int) -> bool:
    try:
        if not end:
            return False
        if not start:
            return False
        dt_s = pd.to_datetime(start)
        dt_e = pd.to_datetime(end)
        return (dt_e - dt_s).days <= max_days
    except Exception:
        return False


class _TempLoggerLevel:
    """Temporarily set a logger level and restore it after."""
    def __init__(self, logger_name: str, level: int):
        import logging
        self.logger_name = logger_name
        self.level = level
        self.prev = None
        self._logging = logging
    def __enter__(self):
        lg = self._logging.getLogger(self.logger_name)
        self.prev = lg.level
        lg.setLevel(self.level)
    def __exit__(self, exc_type, exc, tb):
        lg = self._logging.getLogger(self.logger_name)
        if self.prev is not None:
            lg.setLevel(self.prev)


def _ensure_cache_dir() -> str:
    # Use centralized cache path
    try:
        from src.cache_paths import SHARED_CACHE_ROOT
        root = str(SHARED_CACHE_ROOT / 'microstructure')
    except ImportError:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        root = os.path.join(root_dir, 'cache', 'shared', 'microstructure')
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    return root


def _cache_path(symbol: str, interval: str) -> str:
    cache_dir = _ensure_cache_dir()
    fname = f"{symbol}_{interval}.parquet"
    return os.path.join(cache_dir, fname)


def _load_cache(symbol: str, interval: str, ttl_seconds: int = 3600) -> Optional[pd.DataFrame]:
    try:
        path = _cache_path(symbol, interval)
        if not os.path.exists(path):
            return None
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) > ttl_seconds:
            return None
        df = pd.read_parquet(path)
        df.attrs['cache'] = {'hit': True, 'interval': interval}
        return df
    except Exception:
        return None


def _save_cache(symbol: str, interval: str, df: pd.DataFrame) -> None:
    try:
        path = _cache_path(symbol, interval)
        df.to_parquet(path, index=True)
    except Exception:
        pass


def _alpha_fetch_interval_with_cache(ap: Any, symbol: str, itv: str) -> Tuple[pd.DataFrame, int, int]:
    """Fetch a single interval with caching; returns (df, hit, miss)."""
    cached = _load_cache(symbol, itv, ttl_seconds=1800)
    if cached is not None and not cached.empty:
        return cached, 1, 0
    df = ap.get_intraday_data(symbol, interval=itv, outputsize="compact")
    if not df.empty:
        try:
            df.attrs['cache'] = {'hit': False, 'interval': itv}
        except Exception:
            pass
        _save_cache(symbol, itv, df)
        # polite backoff between requests to respect AV rate limits
        time.sleep(1.2)
        return df, 0, 1
    return pd.DataFrame(), 0, 0


def _alpha_fetch(symbol: str, intervals: List[str]) -> pd.DataFrame:
    if AlphaVantageProvider is None:
        return pd.DataFrame()
    ap = AlphaVantageProvider()
    frames: List[pd.DataFrame] = []
    hits = misses = 0
    for itv in intervals:
        try:
            df, h, m = _alpha_fetch_interval_with_cache(ap, symbol, itv)
            if not df.empty:
                frames.append(df)
            hits += h
            misses += m
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    try:
        tel = {"hits": hits, "misses": misses}
        if rate_utils is not None:
            tel.update(rate_utils.get_stats())
        out.attrs['telemetry'] = {"alpha_vantage": tel}
    except Exception:
        pass
    return out


def _try_alpha_intraday(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Alpha Vantage intraday via aliases; returns daily features with provenance."""
    # Skip AV intraday for long historical windows or when using demo/missing key
    skip_reason = None
    try:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        if os.getenv('ALPHA_VANTAGE_SKIP_INTRADAY', '0') == '1':
            skip_reason = 'env_skip'
        elif api_key.strip().lower() == 'demo' or api_key.strip() == '':
            # Avoid slow rate-limited demo key for intraday; daily proxy will be fine
            skip_reason = 'no_or_demo_key'
        elif not _within_days(start, end, max_days=30):
            # For long windows, intraday adds little and triggers rate limits
            skip_reason = 'window_gt_30d'
    except Exception:
        pass
    if skip_reason is not None:
        # return empty so caller falls back; attach hint in provenance via attrs later if possible
        return pd.DataFrame()

    av_df = pd.DataFrame()
    alias_used: Optional[str] = None
    for sym_try in _aliases(symbol):
        av_df = _alpha_fetch(sym_try, intervals=["1min", "5min", "15min"])
        if not av_df.empty:
            alias_used = sym_try if sym_try != symbol else None
            break
    daily = _build_daily_from_intraday(av_df)
    if daily.empty:
        return pd.DataFrame()
    daily = _apply_range(daily, start, end)
    cache_info = getattr(av_df, 'attrs', {}).get('cache', {}) if hasattr(av_df, 'attrs') else {}
    telemetry = getattr(av_df, 'attrs', {}).get('telemetry', {}) if hasattr(av_df, 'attrs') else {}
    prov = {"source": "alpha_vantage", **({"alias_used": alias_used} if alias_used else {})}
    _attach_attrs(daily, {**prov, **({"cache": cache_info} if cache_info else {})}, telemetry if telemetry else None)
    return daily


def _try_eodhd_intraday(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """EODHD intraday minute bars -> daily microstructure aggregates.

    Uses the EODHD intraday endpoint when available. This is only attempted for
    relatively short windows to avoid excessive API load.
    """
    try:
        if os.getenv("EODHD_SKIP_INTRADAY", "0") == "1":
            return pd.DataFrame()
        if not start or not end:
            return pd.DataFrame()
        try:
            max_days = int(os.getenv("EODHD_INTRADAY_MAX_DAYS", "30"))
        except Exception:
            max_days = 30
        if not _within_days(start, end, max_days=max_days):
            return pd.DataFrame()

        try:
            interval = os.getenv("EODHD_INTRADAY_INTERVAL", "5m")
            interval = str(interval).strip() or "5m"
        except Exception:
            interval = "5m"

        from src.data_sources.eodhd_provider import get_eodhd_provider

        provider = get_eodhd_provider()
        if not getattr(provider, "api_key", None):
            return pd.DataFrame()

        intraday = provider.get_intraday_prices(
            symbol,
            interval=interval,
            start_date=start,
            end_date=end,
        )
        if intraday is None or intraday.empty:
            return pd.DataFrame()

        daily = _build_daily_from_intraday(intraday)
        if daily.empty:
            return pd.DataFrame()
        daily = _apply_range(daily, start, end)

        _attach_attrs(
            daily,
            {"source": "eodhd", "method": "intraday", "interval": interval},
            {"status": "ok", "source": f"eodhd_intraday_{interval}", "proxy": False},
        )
        return _maybe_add_eodhd_quote_features(daily, symbol, end)
    except Exception:
        return pd.DataFrame()


def _try_yf_intraday(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    # Try Tiingo IEX intraday first (requires paid tier)
    try:
        from src.data.universal_data_fetcher import get_fetcher
        fetcher = get_fetcher()
        
        # Check if we have Tiingo provider with intraday capability
        if hasattr(fetcher, '_tiingo_provider') and fetcher._tiingo_provider:
            import requests
            import os
            from pathlib import Path
            
            # Load API token
            env_file = Path('.env')
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip() == 'TIINGO_API_TOKEN':
                            os.environ['TIINGO_API_TOKEN'] = value.strip()
                            break
            
            tiingo_token = os.getenv('TIINGO_API_TOKEN')
            if tiingo_token and start and end:
                # Use Tiingo IEX endpoint for intraday
                headers = {'Authorization': f'Token {tiingo_token}'}
                url = f'https://api.tiingo.com/iex/{symbol}/prices'
                params = {'startDate': start, 'resampleFreq': '1min'}
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        
                        # Build daily from intraday
                        out = _build_daily_from_intraday(df)
                        if not out.empty:
                            out = _apply_range(out, start, end)
                            _attach_attrs(out, {"source": "tiingo_iex_1m"})
                            return out
    except Exception:
        pass
    
    # Fallback: Only attempt Tiingo intraday or AlphaVantage; do not use yfinance
    if not _within_days(start, end, max_days=8):
        return pd.DataFrame()
    # No direct yfinance intraday fallback allowed; return empty so caller can decide
    return pd.DataFrame()


def _try_yf_daily_proxy(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Enhanced RAW MICROSTRUCTURE features from daily OHLCV data.
    
    These are pure mathematical features with HUGE alpha potential:
    - Range & Volatility Ratios (8 features)
    - Volume & Liquidity Proxies (6 features)
    - Order Flow Proxies (5 features)
    - Volatility & Price Impact (5 features)
    - Staleness Proxies (3 features)
    - Overnight Features (4 features)
    
    Total: ~31 raw microstructure features (NO ML, NO HF)
    
    Data Source: EODHD (primary) with fallback to UniversalDataFetcher.

    Note: These are engineered features derived from the chosen data provider's
    daily OHLCV. They are not treated as a "proxy" data source for Stage-B
    strict-source enforcement.
    """
    try:
        eodhd_only = os.getenv('STAGE_B_EODHD_ONLY', '0') == '1'

        # Try EODHD first (primary data source)
        df = None
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            eodhd = get_eodhd_provider()
            df = eodhd.get_eod_prices(symbol, start_date=start, end_date=end)
            if df is not None and not df.empty:
                # Convert EODHD column names to lowercase for consistency
                df.columns = df.columns.str.lower()
        except Exception as e_eodhd:
            logger.debug(f"EODHD fetch failed for {symbol}: {e_eodhd}")
        
        # Fallback to UniversalDataFetcher if EODHD fails
        if (df is None or df.empty) and not eodhd_only:
            from src.data.universal_data_fetcher import UniversalDataFetcher
            udf = UniversalDataFetcher()
            df = udf.get_stock_data(symbol, start_date=start, end_date=end)
            data_source = 'universal_data_fetcher'
        elif df is None or df.empty:
            # Strict EODHD-only mode: do not attempt any other provider.
            return pd.DataFrame()
        else:
            data_source = 'eodhd'
        
        if df.empty or 'close' not in df.columns:
            return pd.DataFrame()
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return pd.DataFrame()
        
        result = pd.DataFrame(index=df.index)
        
        # Helper: avoid division by zero
        close_safe = df['close'].replace(0, np.nan)
        open_safe = df['open'].replace(0, np.nan)
        volume_safe = df['volume'].replace(0, np.nan)
        
        # ============================================================
        # 1. RANGE & VOLATILITY RATIOS (8 features)
        # ============================================================
        
        # True range and ATR
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr_14 = true_range.rolling(14, min_periods=5).mean()
        
        result['micro_true_range'] = true_range
        result['micro_atr_ratio'] = atr_14 / close_safe
        result['micro_range_pct'] = high_low / close_safe
        
        # Body and wick analysis
        body = np.abs(df['close'] - df['open'])
        body_pct = body / close_safe
        result['micro_body_pct'] = body_pct
        
        max_oc = pd.concat([df['open'], df['close']], axis=1).max(axis=1)
        min_oc = pd.concat([df['open'], df['close']], axis=1).min(axis=1)
        wick_top = df['high'] - max_oc
        wick_bottom = min_oc - df['low']
        
        result['micro_wick_top'] = wick_top / close_safe
        result['micro_wick_bottom'] = wick_bottom / close_safe
        result['micro_shadow_ratio'] = (wick_top + wick_bottom) / (body + 1e-9)
        result['micro_range_scaled'] = high_low / (atr_14 + 1e-9)
        
        # ============================================================
        # 2. VOLUME & LIQUIDITY PROXIES (6 features)
        # ============================================================
        
        # Volume statistics
        volume_mean = df['volume'].rolling(20, min_periods=5).mean()
        volume_std = df['volume'].rolling(20, min_periods=5).std()
        
        result['micro_volume_zscore'] = (df['volume'] - volume_mean) / (volume_std + 1e-9)
        result['micro_volume_surge'] = df['volume'] / (volume_mean + 1e-9)
        result['micro_volume_liquidity'] = df['volume'] * high_low
        result['micro_turnover'] = df['volume'] * df['close']
        
        # Amihud illiquidity measure (scaled by 1e6 for readability)
        returns_1d = df['close'].pct_change()
        result['micro_amihud'] = (np.abs(returns_1d) / (df['volume'] + 1e-9)) * 1e6
        
        # High-Low vs Volume correlation (rolling 20-day)
        result['micro_hl_volume_corr'] = high_low.rolling(20, min_periods=5).corr(df['volume'])
        
        # ============================================================
        # 3. ORDER FLOW PROXIES (5 features)
        # ============================================================
        
        # Order flow imbalance proxy (signed volume)
        price_direction = np.sign(df['close'] - df['open'])
        result['micro_ofi_proxy'] = price_direction * df['volume']
        result['micro_signed_volume'] = df['volume'] * returns_1d
        
        # Buying/selling pressure proxy
        result['micro_pressure_proxy'] = (df['close'] - df['open']) / (high_low + 1e-9)
        
        # Demand/supply ratio (where close is within H-L range)
        close_above_low = df['close'] - df['low']
        high_above_close = df['high'] - df['close']
        result['micro_demand_supply_ratio'] = close_above_low / (high_above_close + 1e-9)
        
        # Liquidity imbalance (centered at 0.5)
        result['micro_liquidity_imbalance'] = result['micro_demand_supply_ratio'] - 0.5
        
        # ============================================================
        # 4. VOLATILITY & PRICE IMPACT MEASURES (5 features)
        # ============================================================
        
        # Price impact per unit volume (scaled by 1e9 for readability)
        result['micro_impact_ratio'] = (high_low / (df['volume'] + 1e-9)) * 1e9
        
        # Volatility impact (scaled by 1e9)
        vol_5d = returns_1d.rolling(5, min_periods=2).std()
        result['micro_impact_volatility'] = (vol_5d / (df['volume'] + 1e-9)) * 1e9
        
        # Spread proxy (impact ratio already scaled, multiply by additional factor)
        result['micro_spread_proxy'] = result['micro_impact_ratio'] * 100
        
        # Volatility of volatility
        rolling_vol = returns_1d.rolling(5, min_periods=2).std()
        result['micro_vol_of_vol'] = rolling_vol.rolling(5, min_periods=2).std()
        
        # Intraday volatility proxy
        result['micro_intraday_vol_proxy'] = high_low / (open_safe + 1e-9)
        
        # ============================================================
        # 5. STALENESS (INACTIVITY) PROXIES (3 features)
        # ============================================================
        
        # Zero range flag (high == low)
        result['micro_stale_tick'] = (df['high'] == df['low']).astype(float)
        result['micro_zero_range_flag'] = result['micro_stale_tick']
        
        # Low liquidity flag (volume < 30% of 20-day average)
        result['micro_low_liquidity_flag'] = (df['volume'] < volume_mean * 0.3).astype(float)
        
        # ============================================================
        # 6. OVERNIGHT FEATURES (4 features)
        # ============================================================
        
        # Overnight gap (open vs prior close)
        prior_close = df['close'].shift(1)
        overnight_gap = df['open'] - prior_close
        result['micro_overnight_gap'] = overnight_gap
        result['micro_overnight_vol'] = np.abs(overnight_gap) / (prior_close + 1e-9)
        
        # Intraday vs overnight volatility ratio
        intraday_range = high_low
        overnight_range = np.abs(overnight_gap)
        result['micro_intraday_vs_overnight_vol'] = intraday_range / (overnight_range + 1e-9)
        result['micro_gap_direction'] = np.sign(overnight_gap)
        
        # ============================================================
        # FINALIZE
        # ============================================================
        
        # Optionally add quote snapshot features (last row only; gated to near-real-time windows)
        result = _maybe_add_eodhd_quote_features(result, symbol, end)

        # Fill NaNs and infinities
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(0)
        
        # Add metadata
        result.attrs['telemetry'] = {
            'status': 'ok',
            'source': f'raw_microstructure_daily_{data_source}',
            'proxy': False,
            'feature_count': len(result.columns),
            'data_provider': data_source
        }
        result.attrs['provenance'] = {
            'source': data_source,
            'method': 'raw_microstructure_enhanced',
            'feature_categories': [
                'range_volatility_ratios',
                'volume_liquidity_proxies',
                'order_flow_proxies',
                'volatility_price_impact',
                'staleness_proxies',
                'overnight_features'
            ]
        }
        
        return result
        
    except Exception:
        return pd.DataFrame()


def fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Build microstructure-lite daily features from 1m bars via yfinance (best effort):
      - OFI_PROXY
      - UPDOWN_IMBALANCE
      - MICRO_LIQ_SLOPE (intraday vol per sqrt(volume))
      - STALLED_PRINTS_RATIO
    """
    # 0) Prefer EODHD intraday when available (short windows only)
    df = _try_eodhd_intraday(symbol, start, end)
    if not df.empty:
        return df

    # 1) Try Alpha Vantage intraday
    df = _try_alpha_intraday(symbol, start, end)
    if not df.empty:
        return _maybe_add_eodhd_quote_features(df, symbol, end)
    # 2) yfinance 1m intraday
    df = _try_yf_intraday(symbol, start, end)
    if not df.empty:
        return _maybe_add_eodhd_quote_features(df, symbol, end)
    # 3) yfinance daily proxy
    df = _try_yf_daily_proxy(symbol, start, end)
    if not df.empty:
        return df
    return pd.DataFrame()
