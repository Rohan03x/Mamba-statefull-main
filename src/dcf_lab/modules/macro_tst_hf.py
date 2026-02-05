"""
Hedge-Fund Grade Macro Features with HuggingFace TimeSeriesTransformer
======================================================================

CORE REGIME (10-12 features):
- Yields: TNX (10Y), IRX (3M), 2Y levels
- Yield changes: 1d, 5d, curve slope change
- VIX: level, 1d return, 5d return, spike flag
- Credit: spread level, change, z-score
- Commodities: oil change, gold change

ECONOMIC MOMENTUM (6-8 features):
- derived_inflation_accel: CPI acceleration (change of change)
- derived_gdp_growth_accel: GDP growth momentum
- derived_unemployment_change: Employment delta
- derived_real_rate_change: Real rate velocity
- derived_debt_to_gdp_change: Fiscal trajectory
- derived_trade_balance_change: Trade flow velocity

SHOCK-AWARE SIGNALS (8 features):
- vix_return_1d, vix_return_5d: Volatility shocks
- vix_spike_flag: Binary regime break indicator
- rates_2y_change_1d, rates_2y_change_5d: Rate shock velocity
- curve_slope_change: Curve steepening/flattening
- credit_spread_change: Credit stress velocity
- credit_spread_z: Credit stress percentile

INTERACTIONS (4-6 features, symbol-specific):
- real_rate_change × growth_beta: Rate sensitivity
- oil_change × energy_exposure: Commodity exposure
- vix_change × beta: Vol sensitivity
- curve_slope_change × bank_exposure: Financials proxy

HF TRANSFORMER (internal):
- Uses HF TimeSeriesTransformer internally to compute the final signal.
- Does NOT emit embedding or auxiliary hf_* columns into the feature panel.

Total Target: ~25-30 features (focus on transitions, not levels)
REMOVED: population, GDP levels, GNI, sector %, slow fundamentals
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..signal_bus import ModuleSignal
from ..utils import to_market_session

LOGGER = logging.getLogger(__name__)

# Import centralized cache paths
try:
    from src.cache_paths import MACRO_PANEL_CACHE_ROOT
    _DATA_CACHE = MACRO_PANEL_CACHE_ROOT
except ImportError:
    _DATA_CACHE = Path(__file__).resolve().parents[3] / "cache" / "shared" / "macro_panel"


_MACRO_TST_HF_DROP_PREFIXES: Tuple[str, ...] = (
    "hf_embed_",
)


_MACRO_TST_HF_DROP_COLUMNS: Tuple[str, ...] = (
    "hf_macro_score",
    "hf_confidence",
    "hf_regime",
    "hf_volatility",
)


def _strip_macro_tst_hf_placeholder_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    cols = list(frame.columns)
    drop_cols: List[str] = []
    for c in cols:
        c_str = str(c)
        if c_str in _MACRO_TST_HF_DROP_COLUMNS:
            drop_cols.append(c)
            continue
        if any(c_str.startswith(prefix) for prefix in _MACRO_TST_HF_DROP_PREFIXES):
            drop_cols.append(c)
    if drop_cols:
        return frame.drop(columns=drop_cols, errors="ignore")
    return frame


def _cache_file(symbol: str) -> Path:
    return _DATA_CACHE / f"macro_tst_hf_{symbol.upper()}.parquet"


def _fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch a single FRED series via the public CSV endpoint.

    This avoids adding optional dependencies (e.g., pandas-datareader) and provides a
    robust fallback when EODHD is unavailable.
    """
    series_id = str(series_id).strip()
    if not series_id:
        return pd.Series(dtype=float)

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start_date}&coed={end_date}"
    )
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        LOGGER.debug("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    date_col = None
    for candidate in ("DATE", "date", "observation_date", "observation_date "):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        return pd.Series(dtype=float)

    value_col = series_id if series_id in df.columns else None
    if value_col is None:
        # FRED usually uses the series id as the value column; fall back to last column.
        value_col = df.columns[-1]

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(values.values, index=dates)
    out = out.dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    return out

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
except Exception:  # pragma: no cover - optional dependency
    TimeSeriesTransformerConfig = None  # type: ignore
    TimeSeriesTransformerModel = None  # type: ignore


def _rolling_zscore(series: pd.Series, window: int, *, min_periods: int) -> pd.Series:
    shifted = series.shift(1)
    mean = shifted.rolling(window, min_periods=min_periods).mean()
    std = shifted.rolling(window, min_periods=min_periods).std()
    std = std.replace(0.0, np.nan)
    return (series - mean) / std


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class _PanelBundle:
    frame: pd.DataFrame
    features: List[str]
    coverage: pd.Series
    session_dates: pd.Series


class MacroTSTHF:
    """Predict next-horizon macro direction probability using HF TimeSeriesTransformer."""

    NAME = "macro_tst_hf"
    metadata = {
        "group": "hf",
        "group_cap": 0.5,
        "group_penalty": 1.0,
        "w_min": 0.0,
        "w_max": 0.4,
    }

    def __init__(
        self,
        model_id: Optional[str] = None,
        revision: Optional[str] = None,
        *,
        context_length: int = 180,  # Reduced to allow for lag overhead
        zscore_window: int = 63,
        min_periods: int = 21,
        min_windows: int = 24,  # Adjusted proportionally
        finetune_epochs: int = 10,
        lr: float = 1e-3,
        lags: Optional[Sequence[int]] = None,  # Default will be (1,2,3) - max_lag affects effective context
        d_model_scale: int = 4,
        encoder_layers: int = 2,
        dropout: float = 0.1,
        min_coverage: float = 0.25,
        confidence_scale: float = 0.85,
        use_gpu: bool = True,
        fallback_beta: float = 1.25,
    ) -> None:
        self.model_id = model_id or ""
        self.revision = revision
        self.context_length = int(max(16, context_length))
        self.zscore_window = int(max(10, zscore_window))
        self.min_periods = int(max(5, min_periods))
        self.min_windows = int(max(8, min_windows))
        self.finetune_epochs = int(max(1, finetune_epochs))
        self.lr = float(lr)
        self.lags = tuple(lags) if lags is not None else (1,)  # Minimal lag of 1 (TimeSeriesTransformer requires at least one lag)
        self.d_model_scale = int(max(1, d_model_scale))
        self.encoder_layers = int(max(1, encoder_layers))
        self.dropout = float(max(0.0, min(dropout, 0.5)))
        self.min_coverage = float(max(0.0, min(min_coverage, 1.0)))
        self.confidence_scale = float(max(0.1, min(confidence_scale, 2.0)))
        self.fallback_beta = float(max(0.1, fallback_beta))

        self.device = "cpu"
        if torch is not None and use_gpu and torch.cuda.is_available():  # type: ignore[attr-defined]
            self.device = "cuda"

        self._model: Optional[TimeSeriesTransformerModel] = None  # type: ignore[assignment]
        self._model_input_features: Optional[int] = None
        self._symbol_heads: Dict[str, nn.Module] = {}  # type: ignore[type-arg]
        self._warned_transformers = False
        self._warned_panel = False
        self._warned_eodhd = False

    # ------------------------------------------------------------------
    # Local cache helpers
    # ------------------------------------------------------------------
    def _load_cached_signal(self, symbol: str) -> Optional[pd.DataFrame]:
        path = _cache_file(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            LOGGER.debug("Failed to load macro cache %s: %s", path, exc)
            return None
        if df.empty:
            return None
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()

        # Cache hygiene: some upstream providers occasionally emit placeholder 0.0
        # values for real-interest-rate updates. If we persist those, multiple
        # downstream derived features go constant-zero. Treat that as an unhealthy
        # cache and force a rebuild.
        try:
            if "l3_real_interest_rate" in df.columns:
                tail = pd.to_numeric(df["l3_real_interest_rate"], errors="coerce").fillna(0.0).tail(252)
                if not tail.empty and float(tail.std()) < 1e-12 and float(tail.abs().max()) == 0.0:
                    LOGGER.warning(
                        "macro_tst_hf cache %s has degenerate l3_real_interest_rate; forcing rebuild",
                        path.name,
                    )
                    return None
        except Exception:
            pass

        before_cols = tuple(df.columns)
        df = _strip_macro_tst_hf_placeholder_columns(df)
        if tuple(df.columns) != before_cols:
            try:
                self._write_signal_cache(symbol, df)
            except Exception:
                pass
        return df

    def _write_signal_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        try:
            _DATA_CACHE.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        payload = _strip_macro_tst_hf_placeholder_columns(frame.copy().sort_index())
        payload = payload.reset_index()
        payload = payload.rename(columns={payload.columns[0]: "timestamp"})
        path = _cache_file(symbol)
        try:
            payload.to_parquet(path, index=False)
            LOGGER.info("🗄️  macro_tst_hf cache updated: %s", path)
        except Exception as exc:
            LOGGER.warning("Failed to write macro cache %s: %s", path, exc)

    # ------------------------------------------------------------------
    # LAYER 1: Essential Economic Indicators (MINIMAL - for transitions only)
    # ------------------------------------------------------------------
    def _fetch_eodhd_fundamental_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch MINIMAL fundamental macro - ONLY what's needed for transition features.
        
        HEDGE-FUND DISCIPLINE:
        REMOVED (slow, useless for daily trading):
        - population_total, population_growth
        - gdp_current_usd, consumer_price_index, gni_per_capita
        
        KEPT (for transitions only):
        - inflation_cpi_annual → for derived_inflation_accel
        - unemployment_rate → for derived_unemployment_change
        - gdp_growth_annual → for derived_gdp_growth_accel
        
        Returns: DataFrame with ONLY 3 essential features (daily, forward-filled from annual)
        """
        daily_range = pd.date_range(start=start_date, end=end_date, freq='D')

        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            from eodhd import APIClient
            
            eodhd_provider = get_eodhd_provider()
            
            if not eodhd_provider.api_key:
                if not self._warned_eodhd:
                    LOGGER.debug("EODHD API key not available for macro data")
                    self._warned_eodhd = True
                return pd.DataFrame()
            
            # Initialize official EODHD API client
            api = APIClient(eodhd_provider.api_key)
            
            # MINIMAL Layer 1: ONLY features needed for derived transitions
            layer1_indicators = {
                'inflation_consumer_prices_annual': 'inflation_cpi_annual',
                'unemployment_total_percent': 'unemployment_rate',
                'gdp_growth_annual': 'gdp_growth_annual',
            }
            
            macro_data = {}
            
            for indicator_code, feature_name in layer1_indicators.items():
                try:
                    response = api.get_macro_indicators_data(
                        country='USA',
                        indicator=indicator_code,
                    )
                    if not response:
                        continue
                    df = pd.DataFrame(response)
                    if df.empty or 'Date' not in df.columns or 'Value' not in df.columns:
                        continue
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date').sort_index()
                    macro_data[feature_name] = pd.to_numeric(df['Value'], errors='coerce')
                    LOGGER.debug("✅ Fetched %s: %d data points", feature_name, len(df))
                except Exception as e:
                    LOGGER.debug("Failed to fetch %s: %s", indicator_code, e)
                    continue
            
            if not macro_data:
                raise RuntimeError("EODHD fundamental macro unavailable")
            
            # Combine all indicators
            macro_df = pd.DataFrame(macro_data)
            
            macro_df = macro_df.reindex(daily_range, method='ffill')
            
            LOGGER.debug(f"✅ Layer 1 Minimal: {len(macro_df)} daily rows, {len(macro_df.columns)} essential indicators (hedge-fund grade)")
            return macro_df

        except Exception as e:
            # Fallback to FRED (minimal set only)
            LOGGER.debug("Layer 1 EODHD fundamental macro fetch failed; falling back to FRED: %s", e)

            cpi = _fetch_fred_series("CPIAUCSL", start_date, end_date)
            unrate = _fetch_fred_series("UNRATE", start_date, end_date)
            gdp = _fetch_fred_series("GDP", start_date, end_date)

            macro_df = pd.DataFrame(index=daily_range)
            if not cpi.empty:
                cpi_d = cpi.reindex(daily_range).ffill()
                # Calculate annual inflation rate from CPI level
                macro_df["inflation_cpi_annual"] = cpi_d.pct_change(365).mul(100.0)
            if not unrate.empty:
                macro_df["unemployment_rate"] = unrate.reindex(daily_range).ffill()
            if not gdp.empty:
                gdp_d = gdp.reindex(daily_range).ffill()
                # Calculate annual growth rate from GDP level
                macro_df["gdp_growth_annual"] = gdp_d.pct_change(365).mul(100.0)

            macro_df = macro_df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
            macro_df = macro_df.dropna(axis=1, how="all")
            return macro_df
    
    # ------------------------------------------------------------------
    # LAYER 2: Market Macro (MINIMAL - for transitions only)
    # ------------------------------------------------------------------
    def _fetch_eodhd_market_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch market macro indicators - ONLY what's needed for transition features.
        
        HEDGE-FUND DISCIPLINE:
        REMOVED (slow, useless for daily trading):
        - exports_pct_gdp, imports_pct_gdp, capital_formation_pct_gdp
        
        KEPT (for transitions only):
        - net_trade_balance → for derived_trade_balance_change
        - govt_debt_pct_gdp → for derived_debt_to_gdp_change
        
        Returns: DataFrame with ONLY 2 essential features (daily, forward-filled from annual)
        """
        daily_range = pd.date_range(start=start_date, end=end_date, freq='D')

        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            from eodhd import APIClient
            
            eodhd_provider = get_eodhd_provider()
            
            if not eodhd_provider.api_key:
                return pd.DataFrame()
            
            # Initialize official EODHD API client
            api = APIClient(eodhd_provider.api_key)
            
            # MINIMAL Layer 2: ONLY features needed for derived transitions
            layer2_indicators = {
                'net_trades_goods_services': 'net_trade_balance',
                'debt_percent_gdp': 'govt_debt_pct_gdp',
            }
            
            macro_data = {}
            
            for indicator_code, feature_name in layer2_indicators.items():
                try:
                    response = api.get_macro_indicators_data(
                        country='USA',
                        indicator=indicator_code,
                    )
                    if not response:
                        continue
                    df = pd.DataFrame(response)
                    if df.empty or 'Date' not in df.columns or 'Value' not in df.columns:
                        continue
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date').sort_index()
                    macro_data[feature_name] = pd.to_numeric(df['Value'], errors='coerce')
                    LOGGER.debug("✅ Fetched %s: %d data points", feature_name, len(df))
                except Exception as e:
                    LOGGER.debug("Failed to fetch %s: %s", indicator_code, e)
                    continue
            
            if not macro_data:
                raise RuntimeError("EODHD market macro unavailable")
            
            # Combine and forward-fill to daily
            macro_df = pd.DataFrame(macro_data)
            macro_df = macro_df.reindex(daily_range, method='ffill')
            
            LOGGER.debug(f"✅ Layer 2 Minimal: {len(macro_df)} daily rows, {len(macro_df.columns)} essential indicators (hedge-fund grade)")
            return macro_df

        except Exception as e:
            LOGGER.debug("Layer 2 EODHD market macro fetch failed; falling back to FRED: %s", e)

            netexp = _fetch_fred_series("NETEXP", start_date, end_date)
            debt_pct = _fetch_fred_series("GFDEGDQ188S", start_date, end_date)

            macro_df = pd.DataFrame(index=daily_range)
            if not netexp.empty:
                macro_df["net_trade_balance"] = netexp.reindex(daily_range).ffill()
            if not debt_pct.empty:
                macro_df["govt_debt_pct_gdp"] = debt_pct.reindex(daily_range).ffill()

            macro_df = macro_df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
            macro_df = macro_df.dropna(axis=1, how="all")
            
            LOGGER.debug(f"✅ Layer 2 FRED fallback: {len(macro_df.columns)} essential indicators")
            return macro_df
    
    # ------------------------------------------------------------------
    # LAYER 3: Core Regime + Shock-Aware Signals (HEDGE-FUND GRADE)
    # ------------------------------------------------------------------
    def _fetch_eodhd_structural_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch SHOCK-AWARE structural macro (levels + changes for regime breaks).
        
        HEDGE-FUND DISCIPLINE:
        REMOVED (slow sector composition):
        - industry_pct_gdp, services_pct_gdp, agriculture_pct_gdp
        
        KEPT (shock-aware daily signals):
        - tnx_10y, tnx_2y, irx_3mo: Rate levels (for curve + changes)
        - vix: Volatility level (for shocks + spikes)
        - gold, oil: Commodity prices (for momentum)
        - hyg_credit, lqd_ig: Credit spreads (for stress)
        - real_interest_rate: Real rate level (for velocity)
        
        Returns: DataFrame with 9 daily market signals (NOT slow fundamentals)
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            from eodhd import APIClient

            try:
                import yfinance as yf  # type: ignore
            except Exception:
                yf = None  # type: ignore

            eodhd_provider = get_eodhd_provider()

            # Always create a daily index so we can fill/fuse multiple sources.
            daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
            macro_df = pd.DataFrame(index=daily_range)

            api = None
            if getattr(eodhd_provider, 'api_key', None):
                try:
                    api = APIClient(eodhd_provider.api_key)
                except Exception:
                    api = None
            
            # 1. Real interest rate (annual, for velocity calculation)
            if api is not None:
                try:
                    response = api.get_macro_indicators_data(
                        country='USA',
                        indicator='real_interest_rate',
                    )
                    if response:
                        df = pd.DataFrame(response)
                        if not df.empty and 'Date' in df.columns and 'Value' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.set_index('Date').sort_index()
                            annual_real_rate = pd.to_numeric(df['Value'], errors='coerce')
                            macro_df['real_interest_rate'] = annual_real_rate.reindex(daily_range, method='ffill')
                            LOGGER.debug("✅ Fetched real_interest_rate: %d data points", len(df))
                except Exception as e:
                    LOGGER.debug("Failed to fetch real_interest_rate: %s", e)

            # FRED fallbacks for key structural macro series (works without EODHD & yfinance).
            fred_map = {
                "tnx_10y": "DGS10",
                "tnx_2y": "DGS2",  # 2Y for rate shock analysis
                "irx_3mo": "DGS3MO",
                "vix": "VIXCLS",
                "gold": "GOLDAMGBD228NLBM",
                "oil": "DCOILWTICO",
                # Credit spreads (basis points)
                "hyg_credit": "BAMLH0A0HYM2",
                "lqd_ig": "BAMLC0A0CM",
                # Real rates (daily)
                "real_interest_rate": "REAINTRATREARAT10Y",
            }
            
            # 2. Daily index/ETF proxies (shock-aware signals)
            daily_indices = {
                'TNX.INDX': 'tnx_10y',
                'FVX.INDX': 'tnx_2y',  # 5Y as proxy for 2Y if not available
                'IRX.INDX': 'irx_3mo',
                'VIX.INDX': 'vix',
                'GLD.US': 'gold',
                'USO.US': 'oil',
                'HYG.US': 'hyg_credit',
                'LQD.US': 'lqd_ig',
            }

            yf_map = {
                'TNX.INDX': '^TNX',
                'FVX.INDX': '^FVX',
                'IRX.INDX': '^IRX',
                'VIX.INDX': '^VIX',
                'GLD.US': 'GLD',
                'USO.US': 'USO',
                'HYG.US': 'HYG',
                'LQD.US': 'LQD',
            }
            
            for symbol, feature_name in daily_indices.items():
                try:
                    series = None
                    data = None
                    try:
                        if getattr(eodhd_provider, 'api_key', None):
                            data = eodhd_provider.get_eod_prices(symbol, start_date, end_date)
                    except Exception:
                        data = None

                    if data is not None and not data.empty:
                        close_col = 'Close' if 'Close' in data.columns else ('close' if 'close' in data.columns else None)
                        if close_col is not None:
                            series = pd.to_numeric(data[close_col], errors='coerce')

                    # Fallback to yfinance for structural proxies when EODHD is unavailable.
                    if (series is None or series.dropna().empty) and yf is not None:
                        yf_symbol = yf_map.get(symbol, symbol.replace('.US', ''))
                        hist = yf.Ticker(yf_symbol).history(start=start_date, end=end_date)
                        if hist is not None and not hist.empty:
                            if 'Close' in hist.columns:
                                series = pd.to_numeric(hist['Close'], errors='coerce')
                            elif 'close' in hist.columns:
                                series = pd.to_numeric(hist['close'], errors='coerce')

                    # Fallback to FRED (preferred over yfinance in restricted environments).
                    if series is None or series.dropna().empty:
                        fred_id = fred_map.get(feature_name)
                        if fred_id:
                            fred_series = _fetch_fred_series(fred_id, start_date, end_date)
                            if not fred_series.empty:
                                series = fred_series

                    if series is not None and series.dropna().empty is False:
                        series.index = pd.to_datetime(series.index).tz_localize(None)
                        macro_df[feature_name] = series.reindex(daily_range).ffill()
                        LOGGER.debug("✅ Structural proxy %s (%s): %d points", feature_name, symbol, int(series.notna().sum()))
                
                except Exception as e:
                    LOGGER.debug(f"Failed to fetch {symbol}: {e}")
                    continue
            
            # Also attempt annual-ish structural fallbacks if EODHD macro endpoints are blocked.
            # Real interest rate: always attempt FRED and prefer it when the EODHD
            # endpoint emits placeholder zeros.
            real_rate_fred = _fetch_fred_series(fred_map["real_interest_rate"], start_date, end_date)
            if not real_rate_fred.empty:
                real_rate_daily = real_rate_fred.reindex(daily_range).ffill()
                if "real_interest_rate" not in macro_df.columns or macro_df["real_interest_rate"].dropna().empty:
                    macro_df["real_interest_rate"] = real_rate_daily
                else:
                    existing = pd.to_numeric(macro_df["real_interest_rate"], errors="coerce")
                    blended = existing.copy()
                    # Prefer FRED where existing is missing or appears to be a placeholder 0.0.
                    replace_mask = blended.isna() | ((blended == 0.0) & (real_rate_daily != 0.0))
                    try:
                        blended.loc[replace_mask] = real_rate_daily.loc[replace_mask]
                    except Exception:
                        pass

                    # If the recent window is still degenerate but FRED is not,
                    # replace the entire series with FRED.
                    tail = blended.tail(366).fillna(0.0)
                    tail_fred = real_rate_daily.tail(366).fillna(0.0)
                    if not tail.empty and float(tail.std()) < 1e-12 and float(tail_fred.std()) > 1e-12:
                        blended = real_rate_daily
                    macro_df["real_interest_rate"] = blended

            macro_df = macro_df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
            macro_df = macro_df.dropna(axis=1, how="all")

            LOGGER.debug(f"✅ Layer 3 Structural: {len(macro_df)} daily rows, {len(macro_df.columns)} indicators")
            return macro_df
            
        except Exception as e:
            LOGGER.debug(f"Layer 3 structural macro fetch failed: {e}")
            return pd.DataFrame()
    
    def _calculate_market_macro(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Calculate market-based macro features from price data.
        
        Returns DataFrame with columns:
        - vol_63d: 63-day realized volatility
        - trend_126d: 126-day trend (MA/price)
        - trend_slope: 20-day slope of trend
        - realized_vol: Annualized volatility
        - zscore_momentum: Z-scored 21-day returns
        """
        try:
            # Fetch price data directly from EODHD
            from src.data_sources.eodhd_provider import get_eodhd_provider
            eodhd = get_eodhd_provider()
            price_df = eodhd.get_eod_prices(symbol, start_date, end_date)
            
            if price_df is None or price_df.empty:
                return pd.DataFrame()
            
            # Ensure we have close prices
            if 'Close' not in price_df.columns:
                return pd.DataFrame()
            
            close = price_df['Close'].astype(float)
            returns = close.pct_change()
            
            market_df = pd.DataFrame(index=price_df.index)
            
            # 63-day volatility regime
            market_df['vol_63d'] = returns.rolling(63, min_periods=30).std() * np.sqrt(252)
            
            # 126-day trend regime
            ma_126 = close.rolling(126, min_periods=60).mean()
            market_df['trend_126d'] = ma_126 / close
            
            # Trend slope (20-day)
            market_df['trend_slope'] = market_df['trend_126d'].rolling(20, min_periods=10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # Realized volatility (21-day)
            market_df['realized_vol'] = returns.rolling(21, min_periods=10).std() * np.sqrt(252)
            
            # Z-scored momentum (21-day returns)
            returns_21d = close.pct_change(21)
            market_df['zscore_momentum'] = _rolling_zscore(
                returns_21d, 
                window=63, 
                min_periods=21
            )
            
            LOGGER.debug(f"✅ Market macro: Calculated {len(market_df.columns)} features for {symbol}")
            return market_df
            
        except Exception as e:
            LOGGER.debug(f"Market macro calculation failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_derived_features(
        self, 
        layer1_fundamental: pd.DataFrame,
        layer2_market: pd.DataFrame, 
        layer3_structural: pd.DataFrame,
        symbol: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Calculate HEDGE-FUND GRADE derived features focused on TRANSITIONS not levels.
        
        ECONOMIC MOMENTUM (transitions only):
        - derived_inflation_accel: CPI rate-of-change (NOT level)
        - derived_gdp_growth_accel: GDP growth momentum
        - derived_unemployment_change: Employment velocity
        - derived_real_rate_change: Real rate velocity  
        - derived_debt_to_gdp_change: Fiscal trajectory
        - derived_trade_balance_change: Trade velocity
        
        SHOCK-AWARE SIGNALS:
        - vix_return_1d, vix_return_5d: Vol shocks
        - vix_spike_flag: Regime break (>2 std moves)
        - rates_2y_change_1d, rates_2y_change_5d: Rate shocks
        - curve_slope_change: Curve steepening velocity
        - credit_spread_change, credit_spread_z: Credit stress
        
        CORE REGIME (levels kept):
        - yield_curve_slope: 10Y-3M spread (level)
        - credit_spread: HY-IG spread (level)
        - oil_change, gold_change: Commodity momentum
        
        INTERACTIONS (symbol-specific):
        - real_rate_change × growth_beta
        - oil_change × energy_exposure  
        - vix_change × beta
        - curve_slope_change × bank_exposure
        """
        # Combine all indices
        all_indices = layer1_fundamental.index.union(layer2_market.index).union(layer3_structural.index)
        derived_df = pd.DataFrame(index=all_indices)
        
        # ========== CORE REGIME (levels kept for context) ==========
        
        # Yield curve slope (10Y - 3M, level)
        if 'tnx_10y' in layer3_structural.columns and 'irx_3mo' in layer3_structural.columns:
            derived_df['yield_curve_slope'] = (
                layer3_structural['tnx_10y'] - layer3_structural['irx_3mo']
            )
            # Curve slope CHANGE (velocity)
            derived_df['curve_slope_change'] = derived_df['yield_curve_slope'].diff(1)
        
        # Credit spread (HY - IG, level + change + z-score)
        if 'hyg_credit' in layer3_structural.columns and 'lqd_ig' in layer3_structural.columns:
            derived_df['credit_spread'] = (
                layer3_structural['hyg_credit'] - layer3_structural['lqd_ig']
            )
            # Credit spread CHANGE (stress velocity)
            derived_df['credit_spread_change'] = derived_df['credit_spread'].diff(1)
            # Credit spread Z-SCORE (percentile stress)
            spread_mean = derived_df['credit_spread'].rolling(252, min_periods=60).mean()
            spread_std = derived_df['credit_spread'].rolling(252, min_periods=60).std()
            derived_df['credit_spread_z'] = (derived_df['credit_spread'] - spread_mean) / (spread_std + 1e-8)
        
        # ========== SHOCK-AWARE SIGNALS ==========
        
        # VIX shocks (returns, not levels)
        if 'vix' in layer3_structural.columns:
            vix = layer3_structural['vix']
            derived_df['vix_return_1d'] = vix.pct_change(1)
            derived_df['vix_return_5d'] = vix.pct_change(5)
            
            # VIX SPIKE FLAG (>2 std move = regime break)
            vix_change = vix.diff(1)
            vix_std = vix_change.rolling(63, min_periods=20).std()
            derived_df['vix_spike_flag'] = (vix_change.abs() > 2 * vix_std).astype(float)
        
        # 2Y Rate shocks (if available, otherwise use 3M)
        rate_col = 'tnx_2y' if 'tnx_2y' in layer3_structural.columns else 'irx_3mo'
        if rate_col in layer3_structural.columns:
            rates = layer3_structural[rate_col]
            derived_df['rates_2y_change_1d'] = rates.diff(1)
            derived_df['rates_2y_change_5d'] = rates.diff(5)
        
        # ========== ECONOMIC MOMENTUM (transitions only) ==========
        
        # GDP growth ACCELERATION (not level, not even growth rate)
        if 'gdp_growth_annual' in layer1_fundamental.columns:
            gdp_growth = layer1_fundamental['gdp_growth_annual']
            derived_df['derived_gdp_growth_accel'] = gdp_growth.diff(63)  # Quarterly acceleration
        
        # Inflation ACCELERATION (rate-of-change of rate-of-change)
        if 'inflation_cpi_annual' in layer1_fundamental.columns:
            inflation = layer1_fundamental['inflation_cpi_annual']
            derived_df['derived_inflation_accel'] = inflation.diff(21)  # Monthly acceleration
        
        # Unemployment CHANGE (velocity, not level)
        if 'unemployment_rate' in layer1_fundamental.columns:
            unemp = layer1_fundamental['unemployment_rate']
            derived_df['derived_unemployment_change'] = unemp.diff(21)  # Monthly change
        
        # Real rate CHANGE (velocity)
        if 'real_interest_rate' in layer3_structural.columns:
            real_rate = layer3_structural['real_interest_rate']
            derived_df['derived_real_rate_change'] = real_rate.diff(5)  # Weekly velocity
        
        # Debt/GDP CHANGE (fiscal trajectory)
        if 'govt_debt_pct_gdp' in layer2_market.columns:
            debt = layer2_market['govt_debt_pct_gdp']
            derived_df['derived_debt_to_gdp_change'] = debt.diff(252)  # Annual change
        
        # Trade balance CHANGE (trade flow velocity)
        if 'net_trade_balance' in layer2_market.columns:
            trade = layer2_market['net_trade_balance']
            derived_df['derived_trade_balance_change'] = trade.diff(63)  # Quarterly change
        
        # ========== COMMODITY MOMENTUM ==========
        
        # Oil change (NOT level)
        if 'oil' in layer3_structural.columns:
            derived_df['oil_change'] = layer3_structural['oil'].pct_change(5)  # Weekly
        
        # Gold change (NOT level)
        if 'gold' in layer3_structural.columns:
            derived_df['gold_change'] = layer3_structural['gold'].pct_change(5)  # Weekly
        
        # ========== INTERACTIONS (symbol-specific) ==========
        # These are FEATURES, not weights - they enter the model directly
        
        # Define symbol betas/exposures (static for now, could be dynamic)
        sector_betas = {
            # Growth stocks (high rate sensitivity)
            'AAPL': {'growth_beta': 1.2, 'energy': 0.0, 'beta': 1.1, 'bank': 0.0},
            'MSFT': {'growth_beta': 1.15, 'energy': 0.0, 'beta': 1.05, 'bank': 0.0},
            'NVDA': {'growth_beta': 1.5, 'energy': 0.0, 'beta': 1.6, 'bank': 0.0},
            'META': {'growth_beta': 1.3, 'energy': 0.0, 'beta': 1.25, 'bank': 0.0},
            'GOOGL': {'growth_beta': 1.2, 'energy': 0.0, 'beta': 1.1, 'bank': 0.0},
            'AMZN': {'growth_beta': 1.25, 'energy': 0.0, 'beta': 1.15, 'bank': 0.0},
            'TSLA': {'growth_beta': 1.4, 'energy': 0.2, 'beta': 1.8, 'bank': 0.0},
            
            # Energy (high oil sensitivity)
            'XOM': {'growth_beta': 0.6, 'energy': 1.5, 'beta': 0.85, 'bank': 0.0},
            
            # Financials (high curve sensitivity)
            'JPM': {'growth_beta': 0.9, 'energy': 0.0, 'beta': 1.05, 'bank': 1.4},
            
            # Healthcare/Defensive
            'UNH': {'growth_beta': 0.8, 'energy': 0.0, 'beta': 0.75, 'bank': 0.0},
            
            # Cyclicals
            'AMD': {'growth_beta': 1.4, 'energy': 0.0, 'beta': 1.5, 'bank': 0.0},
            'COST': {'growth_beta': 0.9, 'energy': 0.1, 'beta': 0.85, 'bank': 0.0},
            
            # Market
            'SPY': {'growth_beta': 1.0, 'energy': 0.15, 'beta': 1.0, 'bank': 0.2},
        }
        
        exposures = sector_betas.get(symbol, {'growth_beta': 1.0, 'energy': 0.0, 'beta': 1.0, 'bank': 0.0})
        
        # Interaction 1: Real rate change × growth beta
        if 'derived_real_rate_change' in derived_df.columns:
            derived_df['interact_rate_growth'] = derived_df['derived_real_rate_change'] * exposures['growth_beta']
        
        # Interaction 2: Oil change × energy exposure
        if 'oil_change' in derived_df.columns:
            derived_df['interact_oil_energy'] = derived_df['oil_change'] * exposures['energy']
        
        # Interaction 3: VIX change × beta
        if 'vix_return_1d' in derived_df.columns:
            derived_df['interact_vix_beta'] = derived_df['vix_return_1d'] * exposures['beta']
        
        # Interaction 4: Curve slope change × bank exposure
        if 'curve_slope_change' in derived_df.columns:
            derived_df['interact_curve_bank'] = derived_df['curve_slope_change'] * exposures['bank']
        
        LOGGER.debug(f"✅ Hedge-fund derived features: {len(derived_df.columns)} features (transitions + shocks + interactions)")
        return derived_df
    
    def _calculate_structural_features(self, eodhd_macro: pd.DataFrame, market_macro: pd.DataFrame) -> pd.DataFrame:
        """
        DEPRECATED: Old structural features calculation (kept for compatibility).
        Now using _calculate_derived_features() instead.
        
        Returns DataFrame with columns:
        - yield_curve_slope: TNX - IRX
        - inflation_gap: TIP/TLT deviation from mean
        - usd_normalized: USD z-score
        - vol_regime_score: Volatility regime classification
        - trend_regime_score: Trend regime classification
        - macro_tightening_signal: Combined tightening indicator
        """
        # Combine indices
        all_indices = eodhd_macro.index.union(market_macro.index)
        structural_df = pd.DataFrame(index=all_indices)
        
        # Yield curve slope
        if 'tnx_10y' in eodhd_macro.columns and 'irx_3mo' in eodhd_macro.columns:
            structural_df['yield_curve_slope'] = (
                eodhd_macro['tnx_10y'] - eodhd_macro['irx_3mo']
            ) / 100.0  # Convert to decimal
        
        # Inflation gap (deviation from 63-day mean)
        if 'tip_inflation' in eodhd_macro.columns:
            tip_mean = eodhd_macro['tip_inflation'].rolling(63, min_periods=20).mean()
            structural_df['inflation_gap'] = eodhd_macro['tip_inflation'] - tip_mean
        
        # USD normalized (z-score over 126 days)
        if 'uup_usd' in eodhd_macro.columns:
            structural_df['usd_normalized'] = _rolling_zscore(
                eodhd_macro['uup_usd'],
                window=126,
                min_periods=40
            )
        
        # Volatility regime score (0-1, high vol = 1)
        if 'vol_63d' in market_macro.columns:
            vol_percentile = market_macro['vol_63d'].rolling(252, min_periods=60).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 1 and (x.max() - x.min()) > 0 else 0.5
            )
            structural_df['vol_regime_score'] = vol_percentile
        
        # Trend regime score (0-1, strong uptrend = 1)
        if 'trend_126d' in market_macro.columns:
            # Trend > 1 means price above MA (uptrend)
            trend_signal = market_macro['trend_126d'] - 1.0
            structural_df['trend_regime_score'] = trend_signal.clip(-0.5, 0.5) + 0.5  # Normalize to 0-1
        
        # Macro tightening signal (combines yield slope inversion + vol spike)
        if 'yield_curve_slope' in structural_df.columns and 'vol_regime_score' in structural_df.columns:
            # Inverted curve (negative slope) + high vol = tightening
            slope_signal = -structural_df['yield_curve_slope'].clip(-2, 0) / 2.0  # 0-1, inverted=1
            vol_signal = structural_df['vol_regime_score']
            structural_df['macro_tightening_signal'] = (slope_signal + vol_signal) / 2.0
        
        LOGGER.debug(f"✅ Structural: Calculated {len(structural_df.columns)} features")
        return structural_df

    # ------------------------------------------------------------------
    # Data ingestion helpers
    # ------------------------------------------------------------------
    def _load_panel(self, symbol: str) -> Optional[_PanelBundle]:
        if not _DATA_CACHE.exists():
            return None
        path = _DATA_CACHE / f"macro_features_{symbol.upper()}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - cache errors are non-fatal
            LOGGER.debug("Failed to read macro panel %s: %s", path, exc)
            return None

        if df.empty:
            return None

        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            timestamps = pd.to_datetime(df.index, utc=True, errors="coerce")
        timestamps = timestamps.dropna()
        if timestamps.empty:
            return None

        session_dates = to_market_session(timestamps, tz="America/New_York")
        df = df.assign(session_date=session_dates.values)
        df = df.dropna(subset=["session_date"]).copy()
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()

        feature_cols = [
            col
            for col in df.columns
            if col not in {"timestamp", "session_date"} and not str(col).lower().startswith("meta_")
        ]
        if not feature_cols:
            return None

        feature_frame = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        feature_frame = feature_frame.groupby(df["session_date"]).last().sort_index()
        coverage = feature_frame.notna().mean(axis=1).clip(0.0, 1.0)

        return _PanelBundle(frame=feature_frame, features=feature_cols, coverage=coverage, session_dates=feature_frame.index.to_series())

    def _zscore_panel(self, bundle: _PanelBundle) -> pd.DataFrame:
        zed = pd.DataFrame(index=bundle.frame.index)
        for col in bundle.features:
            series = bundle.frame[col].astype(float)
            zed[col] = _rolling_zscore(series, self.zscore_window, min_periods=self.min_periods)
        return zed

    # ------------------------------------------------------------------
    # Model utilities
    # ------------------------------------------------------------------
    def _ensure_model(self, num_features: int) -> Optional[TimeSeriesTransformerModel]:  # type: ignore[override]
        if torch is None or TimeSeriesTransformerConfig is None or TimeSeriesTransformerModel is None:
            if not self._warned_transformers:
                LOGGER.warning("Transformers/torch unavailable for MacroTSTHF; falling back to volatility proxy.")
                self._warned_transformers = True
            return None

        # Defensive: some environments (and some transformers loading paths) can
        # create models on the special 'meta' device as a placeholder (no real
        # storage). Moving such a model with `.to(...)` fails with:
        # "Cannot copy out of meta tensor; no data!".
        #
        # Force new tensors to materialize on CPU first; then move to CUDA.
        try:  # pragma: no cover
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
        except Exception:
            pass

        if self._model is not None and self._model_input_features == num_features:
            return self._model

        d_model = max(32, num_features * self.d_model_scale)
        lags_list = list(self.lags) if self.lags else []  # Empty list if no lags specified
        config = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=self.context_length,
            input_size=1,
            num_time_features=num_features,
            lags_sequence=lags_list,  # Can be empty list
            d_model=d_model,
            encoder_layers=self.encoder_layers,
            decoder_layers=0,
            dropout=self.dropout,
        )

        model: Optional[TimeSeriesTransformerModel] = None
        if self.model_id:
            try:
                # Avoid low-mem meta initialization paths where possible.
                try:
                    model = TimeSeriesTransformerModel.from_pretrained(
                        self.model_id,
                        config=config,
                        revision=self.revision,
                        low_cpu_mem_usage=False,
                        device_map=None,
                    )
                except TypeError:
                    model = TimeSeriesTransformerModel.from_pretrained(
                        self.model_id,
                        config=config,
                        revision=self.revision,
                    )
            except Exception as exc:  # pragma: no cover - remote load issues
                LOGGER.debug("Failed to load pretrained TST %s: %s", self.model_id, exc)
                model = None
        if model is None:
            model = TimeSeriesTransformerModel(config)

        def _has_meta_tensors(module: object) -> bool:
            try:
                return any(getattr(p, "is_meta", False) for p in module.parameters())  # type: ignore[attr-defined]
            except Exception:
                return False

        # If the model somehow contains meta tensors, materialize them.
        # We do NOT disable transformers: worst case we fall back to a randomly
        # initialized model built from `config` (which is still a valid TST).
        if _has_meta_tensors(model):
            LOGGER.warning(
                "macro_tst_hf: model has meta tensors; attempting to materialize weights on CPU before device move"
            )

            # 1) Best-effort: allocate empty storage on CPU, then init weights.
            try:  # pragma: no cover
                to_empty = getattr(model, "to_empty", None)
                if callable(to_empty):
                    try:
                        model = to_empty(device="cpu")
                    except TypeError:
                        model = to_empty(torch.device("cpu"))
            except Exception as exc:
                LOGGER.debug("macro_tst_hf: to_empty failed: %s", exc)

            try:  # pragma: no cover
                init_weights = getattr(model, "init_weights", None)
                if callable(init_weights):
                    init_weights()
            except Exception as exc:
                LOGGER.debug("macro_tst_hf: init_weights failed: %s", exc)

            # 2) If still meta, rebuild from config to force real storage.
            if _has_meta_tensors(model):
                LOGGER.warning(
                    "macro_tst_hf: weights still meta after materialization; rebuilding model from config (random init)"
                )
                try:  # pragma: no cover
                    if hasattr(torch, "set_default_device"):
                        torch.set_default_device("cpu")
                except Exception:
                    pass
                model = TimeSeriesTransformerModel(config)

                # Optional: ensure the rebuilt model has concrete weights.
                if _has_meta_tensors(model):
                    LOGGER.warning(
                        "macro_tst_hf: rebuilt model still meta; forcing to_empty+init_weights again"
                    )
                    try:  # pragma: no cover
                        to_empty = getattr(model, "to_empty", None)
                        if callable(to_empty):
                            try:
                                model = to_empty(device="cpu")
                            except TypeError:
                                model = to_empty(torch.device("cpu"))
                    except Exception:
                        pass
                    try:  # pragma: no cover
                        init_weights = getattr(model, "init_weights", None)
                        if callable(init_weights):
                            init_weights()
                    except Exception:
                        pass

        # Move to target device (GPU if available), falling back to CPU.
        try:
            model.to(self.device)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("macro_tst_hf: failed to move model to %s: %s; falling back to CPU", self.device, exc)
            self.device = "cpu"
            model.to("cpu")

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self._model = model
        self._model_input_features = num_features
        return self._model

    def _get_head(self, symbol: str, hidden_size: int) -> Optional[nn.Module]:
        """
        LAYER 5: Create signal heads for generating HF signals from embeddings.
        
        Creates 4 prediction heads:
        - score_head: Macro regime score (-1 to +1)
        - conf_head: Confidence (0 to 1)
        - regime_head: Economic regime (0 to 1)
        - vol_head: Volatility expectation (0 to 1)
        """
        if torch is None or nn is None:
            return None
        
        head = self._symbol_heads.get(symbol)
        if head is None:
            # Create multi-head network for Layer 5 signals
            head = nn.ModuleDict({
                'score': nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, 1),
                    nn.Tanh()  # Score: -1 to +1
                ),
                'conf': nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, 1),
                    nn.Sigmoid()  # Confidence: 0 to 1
                ),
                'regime': nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, 1),
                    nn.Sigmoid()  # Regime: 0 (recession) to 1 (expansion)
                ),
                'volatility': nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, 1),
                    nn.Sigmoid()  # Volatility: 0 (low) to 1 (high)
                )
            }).to(self.device)
            self._symbol_heads[symbol] = head
        else:
            head = head.to(self.device)
        return head

    def _hidden_for_window(self, model: TimeSeriesTransformerModel, window: np.ndarray) -> Optional[torch.Tensor]:
        """
        LAYER 4: Extract HF Transformer embedding from sequence.
        
        Input: window of shape (context_length, num_features)
               e.g., (180, 34) for 180 days × 34 macro features
        
        Process:
        1. Construct sequence X_t = [f_{t-179}, f_{t-178}, ..., f_t]
        2. Feed to TimeSeriesTransformer
        3. Extract last_hidden_state from encoder
        
        Output: hidden state vector of shape (d_model,) e.g., (128,)
        """
        if torch is None:
            return None
        
        # Sequence construction:
        # past_values: (batch_size, context_length, 1) - composite for univariate requirement
        # past_time_features: (batch_size, context_length, num_features) - all features as time covariates
        
        # Ensure numerical stability (HF models can propagate NaNs if inputs contain non-finite values)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        composite = window.mean(axis=1, dtype=np.float32)  # Aggregate for past_values
        composite = np.nan_to_num(composite, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # TimeSeriesTransformer expects (batch, time, input_size)
        past_values = torch.from_numpy(composite).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # All Layers 1-3 features as time-varying covariates
        time_feats = torch.from_numpy(window).unsqueeze(0).to(self.device)
        
        mask = torch.ones_like(past_values)
        
        with torch.no_grad():
            outputs = model(
                past_values=past_values,  # Shape: (1, context_length, 1)
                past_time_features=time_feats,  # Shape: (1, context_length, num_features)
                past_observed_mask=mask,  # Shape: (1, context_length, 1)
            )

        # Extract an embedding from the *encoder*.
        # `last_hidden_state` is the decoder output; when decoder_layers=0 it can be
        # degenerate/constant. `encoder_last_hidden_state` varies with the inputs.
        encoder_hidden = getattr(outputs, "encoder_last_hidden_state", None)
        if encoder_hidden is not None:
            hidden = encoder_hidden[:, -1, :]  # Shape: (1, d_model)
        else:
            hidden = outputs.last_hidden_state[:, -1, :]  # Shape: (1, d_model)

        # Guard against non-finite hidden states (rare but can happen with malformed inputs / device issues)
        if torch.isfinite(hidden).all() is False:
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)
        
        return hidden.squeeze(0)  # Shape: (d_model,)

    def _train_head(
        self,
        symbol: str,
        model: TimeSeriesTransformerModel,
        windows: Sequence[np.ndarray],
        targets: Sequence[float],
    ) -> Optional[nn.Module]:
        if torch is None or nn is None or not windows or len(targets) != len(windows):
            return None
        head = self._get_head(symbol, model.config.d_model)
        if head is None:
            return None
        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(self.finetune_epochs):
            total_loss = 0.0
            for window, target in zip(windows, targets):
                hidden = self._hidden_for_window(model, window)
                if hidden is None:
                    continue
                logit = head(hidden.unsqueeze(0)).squeeze(0)
                target_tensor = torch.tensor([target], dtype=torch.float32, device=self.device)
                loss = criterion(logit, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            if total_loss == 0.0:
                break
        head.eval()
        return head

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _prepare_windows(
        self,
        zpanel: pd.DataFrame,
        coverage: pd.Series,
    ) -> Tuple[List[np.ndarray], List[float]]:
        # Need at least context_length + max_lag + 2 to create one window + target
        max_lag = max(self.lags) if self.lags else 0
        min_required = self.context_length + max_lag + 2
        if len(zpanel) < min_required:
            return [], []
        values = zpanel.fillna(0.0).to_numpy(dtype=np.float32)
        coverage_arr = coverage.reindex(zpanel.index).fillna(0.0).to_numpy(dtype=np.float32)
        composite = values.mean(axis=1)
        windows: List[np.ndarray] = []
        targets: List[float] = []
        max_lag = max(self.lags) if self.lags else 0
        window_size = self.context_length + max_lag  # Need extra rows for lag computation
        for idx in range(window_size, len(values) - 1):
            window = values[idx - window_size : idx]  # Grab context_length + max_lag rows
            window_cov = coverage_arr[idx - window_size : idx]
            if window_cov.mean() < self.min_coverage:
                continue
            target = 1.0 if composite[idx] - composite[idx - 1] > 0 else 0.0
            windows.append(window)
            targets.append(target)
        return windows, targets

    def _fallback_probability(self, bundle: _PanelBundle) -> Tuple[float, float]:
        zpanel = self._zscore_panel(bundle)
        vix_col = None
        for col in zpanel.columns:
            if "vix" in col.lower():
                vix_col = col
                break
        if vix_col is None:
            return 0.5, 0.0
        z_vix = zpanel[vix_col].dropna()
        if z_vix.empty:
            return 0.5, 0.0
        latest = float(z_vix.iloc[-1])
        prob = _sigmoid(self.fallback_beta * -latest)
        coverage_last = float(bundle.coverage.reindex(zpanel.index).fillna(0.0).iloc[-1])
        conf = float(np.clip(abs(prob - 0.5) * 2.0 * coverage_last * self.confidence_scale, 0.0, 1.0))
        return prob, conf

    def emit_signal(self, *, symbol: str, horizon: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> ModuleSignal:
        """
        Generate unified macro super-family signal combining:
        1. EODHD fundamental macro (yields, USD, inflation)
        2. Market-based macro (volatility, trends, momentum)
        3. Structural features (curve slope, regime scores)
        4. HF Transformer embeddings (latent macro representation)
        
        Returns ~20 comprehensive macro features.
        """
        # Set date range with buffer for rolling calculations
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        request_start_ts = pd.Timestamp(start_date)
        request_end_ts = pd.Timestamp(end_date)

        cached_panel = self._load_cached_signal(symbol)
        df_signal_full: Optional[pd.DataFrame] = None
        if cached_panel is not None:
            cache_start = cached_panel.index.min()
            cache_end = cached_panel.index.max()
            if cache_start <= request_start_ts and cache_end >= request_end_ts:
                LOGGER.info(
                    "♻️ macro_tst_hf: reusing cached features for %s (%s → %s)",
                    symbol,
                    cache_start.date(),
                    cache_end.date(),
                )
                df_signal_full = cached_panel
            else:
                LOGGER.info(
                    "♻️ macro_tst_hf cache present but insufficient coverage (%s → %s vs requested %s → %s)",
                    cache_start.date(),
                    cache_end.date(),
                    request_start_ts.date(),
                    request_end_ts.date(),
                )
                cached_panel = None

        # Add 1-year buffer for rolling window calculations when rebuilding
        buffer_start = (request_start_ts - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        
        # ================================================================
        # UNIFIED 3-LAYER MACRO ARCHITECTURE
        # ================================================================
        
        if df_signal_full is None:
            # LAYER 1: Fundamental Economic Indicators (8 annual indicators)
            LOGGER.debug("📊 Layer 1: Fetching fundamental macro (GDP, inflation, employment, population)")
            layer1_fundamental = self._fetch_eodhd_fundamental_macro(buffer_start, end_date)

            # LAYER 2: Market Macro Indicators (5 annual indicators)
            LOGGER.debug("🌍 Layer 2: Fetching market macro (trade, debt, investment)")
            layer2_market = self._fetch_eodhd_market_macro(buffer_start, end_date)

            # LAYER 3: Structural Indicators (4 annual + 7 daily = 11 total)
            LOGGER.debug("🏗️ Layer 3: Fetching structural macro (rates, industry, services, indices)")
            layer3_structural = self._fetch_eodhd_structural_macro(buffer_start, end_date)

            # DERIVED: Calculate cross-layer features (22 hedge-fund grade features)
            LOGGER.debug("🔧 Calculating hedge-fund grade derived features (transitions, shocks, interactions)")
            derived_features = self._calculate_derived_features(
                layer1_fundamental,
                layer2_market,
                layer3_structural,
                symbol=symbol,
            )

            # ================================================================
            # COMBINE ALL LAYERS INTO UNIFIED PANEL
            # ================================================================
            combined_index = (
                layer1_fundamental.index
                .union(layer2_market.index)
                .union(layer3_structural.index)
                .union(derived_features.index)
            )

            macro_panel = pd.DataFrame(index=combined_index)

            # Add Layer 1 (Fundamental Economic)
            for col in layer1_fundamental.columns:
                macro_panel[f"l1_{col}"] = layer1_fundamental[col]

            # Add Layer 2 (Market Macro)
            for col in layer2_market.columns:
                macro_panel[f"l2_{col}"] = layer2_market[col]

            # Add Layer 3 (Structural)
            for col in layer3_structural.columns:
                macro_panel[f"l3_{col}"] = layer3_structural[col]

            # Add Derived features
            for col in derived_features.columns:
                macro_panel[f"derived_{col}"] = derived_features[col]

            # Forward-fill missing values (macro data is slower-updating)
            macro_panel = macro_panel.sort_index().ffill()

            LOGGER.info(
                "✅ Unified macro panel: %d rows × %d features\n"
                "   Layer 1 (Fundamental): %d features\n"
                "   Layer 2 (Market): %d features\n"
                "   Layer 3 (Structural): %d features\n"
                "   Derived: %d features",
                len(macro_panel),
                len(macro_panel.columns),
                len(layer1_fundamental.columns),
                len(layer2_market.columns),
                len(layer3_structural.columns),
                len(derived_features.columns),
            )

            if macro_panel.empty or len(macro_panel) < self.context_length:
                LOGGER.warning(
                    "Insufficient macro data for %s: %d rows < %d required",
                    symbol,
                    len(macro_panel),
                    self.context_length,
                )
                return ModuleSignal(
                    name=self.NAME,
                    horizon=int(horizon),
                    df=pd.DataFrame(columns=["score", "conf"]),
                    symbol=symbol,
                )

            # Step 4: Apply HF TRANSFORMER for embeddings (if available)
            zpanel = self._zscore_panel_from_df(macro_panel)
            model = self._ensure_model(zpanel.shape[1])

            if model is not None and torch is not None:
                # Use transformer to generate embeddings
                df_signal = self._build_hf_signal_with_embeddings(
                    macro_panel, zpanel, model, start_date, end_date, horizon, symbol
                )

                # If the transformer path produces a degenerate signal (constant / all-zero),
                # fall back to the deterministic non-transformer signal so we don't leak
                # useless features into TrackC.
                try:
                    if df_signal is None or df_signal.empty:
                        raise ValueError("empty transformer signal")

                    score = pd.to_numeric(df_signal.get("score"), errors="coerce") if "score" in df_signal.columns else None
                    conf = pd.to_numeric(df_signal.get("conf"), errors="coerce") if "conf" in df_signal.columns else None

                    degenerate = False
                    if score is None or conf is None:
                        degenerate = True
                    else:
                        score_filled = score.fillna(0.0)
                        conf_filled = conf.fillna(0.0)
                        degenerate = bool(
                            (float(score_filled.abs().sum()) == 0.0)
                            or (score_filled.nunique(dropna=False) <= 1)
                            or (conf_filled.nunique(dropna=False) <= 1)
                        )

                    if degenerate:
                        LOGGER.warning("HF transformer signal degenerate for %s; falling back.", symbol)
                        df_signal = self._build_signal_without_transformer(
                            macro_panel, start_date, end_date, horizon, symbol
                        )
                except Exception as exc:
                    LOGGER.debug("HF embedding validation failed (%s); falling back.", exc)
                    df_signal = self._build_signal_without_transformer(
                        macro_panel, start_date, end_date, horizon, symbol
                    )
            else:
                # Fallback: use raw features without transformer
                df_signal = self._build_signal_without_transformer(
                    macro_panel, start_date, end_date, horizon, symbol
                )

            df_signal_full = df_signal.copy()
            self._write_signal_cache(symbol, df_signal_full)

        df_signal = df_signal_full.copy()

        # Filter to requested date range
        if not df_signal.empty:
            df_signal = df_signal[
                (df_signal.index >= request_start_ts)
                & (df_signal.index <= request_end_ts)
            ]

        LOGGER.info(
            "✅ macro_tst_hf: Generated %d rows × %d features for %s",
            len(df_signal),
            len(df_signal.columns) if not df_signal.empty else 0,
            symbol,
        )

        return ModuleSignal(name=self.NAME, horizon=int(horizon), df=df_signal, symbol=symbol)
    
    def _zscore_panel_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling z-score normalization to DataFrame."""
        zpanel = pd.DataFrame(index=df.index)
        for col in df.columns:
            series = df[col].astype(float)
            zpanel[col] = _rolling_zscore(series, self.zscore_window, min_periods=self.min_periods)
        return zpanel
    
    def _build_hf_signal_with_embeddings(
        self,
        macro_panel: pd.DataFrame,
        zpanel: pd.DataFrame,
        model: TimeSeriesTransformerModel,
        start_date: str,
        end_date: str,
        horizon: int,
        symbol: str,
    ) -> pd.DataFrame:
        """Build macro_tst_hf signal using HF TimeSeriesTransformer.

        Notes:
        - Embeddings are computed internally but not emitted as columns.
        - Output includes macro panel features plus `score`/`conf`.
        """
        values = zpanel.fillna(0.0).to_numpy(dtype=np.float32)
        valid_dates = []
        all_features = []
        
        max_lag = max(self.lags) if self.lags else 0
        window_size = self.context_length + max_lag
        
        LOGGER.debug(
            f"🔮 Layer 4+5: Processing {len(values)} dates with {zpanel.shape[1]} features, "
            f"window_size={window_size}"
        )
        
        for idx in range(window_size, len(values)):
            date = zpanel.index[idx]
            window = values[idx - window_size : idx]  # Shape: (context_length, num_features)
            
            # ================================================================
            # LAYER 4: HF TRANSFORMER EMBEDDINGS
            # ================================================================
            # Feed 180-day sequence of 34 macro features → Get latent embedding
            hidden = self._hidden_for_window(model, window)
            if hidden is None:
                continue
            
            embedding = hidden.cpu().detach().numpy()  # Shape: (d_model,) e.g., (128,)
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Build feature dict for this date
            features = {}
            
            # ----------------------------------------------------------------
            # LAYER 1-3: Include all raw macro features (34 total)
            # ----------------------------------------------------------------
            if date in macro_panel.index:
                for col in macro_panel.columns:
                    features[col] = float(macro_panel.loc[date, col])
            else:
                # Forward-fill from previous date
                prev_idx = macro_panel.index.get_indexer([date], method='ffill')[0]
                if prev_idx >= 0:
                    for col in macro_panel.columns:
                        features[col] = float(macro_panel.iloc[prev_idx][col])
            
            embedding_dims = min(8, len(embedding))
            
            # ================================================================
            # LAYER 5: HF SEQUENCE SIGNALS (4 features)
            # ================================================================
            # Generate signals from embedding using neural heads (if available)
            # Otherwise use heuristic combinations of embedding + derived features
            
            # Get signal head if exists (trained neural network)
            head = self._symbol_heads.get(symbol)
            
            if head is not None and isinstance(head, nn.ModuleDict):
                # Use trained Layer 5 heads
                with torch.no_grad():
                    hidden_tensor = torch.from_numpy(embedding).unsqueeze(0).to(self.device)
                    
                    features['score'] = float(head['score'](hidden_tensor).squeeze().cpu().numpy())
                    features['conf'] = float(head['conf'](hidden_tensor).squeeze().cpu().numpy())
            else:
                # Fallback: Heuristic Layer 5 signals from embedding + derived features
                
                # 1. MACRO REGIME SCORE (-1 to +1)
                # Interpretation: +1 = bullish macro, -1 = bearish macro
                embedding_mean = float(np.mean(embedding[:embedding_dims]))
                features['score'] = float(np.tanh(embedding_mean))  # -1 to +1
                
                # 2. CONFIDENCE (0 to 1)
                # Based on embedding magnitude (how strong is the signal?)
                embedding_magnitude = float(np.linalg.norm(embedding[:embedding_dims]))
                features['conf'] = float(min(1.0, embedding_magnitude / 15.0))  # 0-1 scale
            
            # Final safety: never allow NaN/inf into output features.
            for key, value in list(features.items()):
                try:
                    if isinstance(value, (float, int, np.floating, np.integer)):
                        if not np.isfinite(value):
                            features[key] = 0.0
                except Exception:
                    continue

            valid_dates.append(date)
            all_features.append(features)
        
        if not all_features:
            LOGGER.warning("No valid features generated from HF transformer")
            return pd.DataFrame(columns=["score", "conf"])
        
        df = pd.DataFrame(all_features, index=pd.DatetimeIndex(valid_dates).normalize())
        
        # Ensure score/conf exist and are finite.
        df['score'] = pd.to_numeric(df.get('score', 0.0), errors='coerce').fillna(0.0)
        df['conf'] = pd.to_numeric(df.get('conf', 0.5), errors='coerce').fillna(0.5)
        
        LOGGER.info(
            f"✅ Layer 4+5 complete: {len(df)} rows with "
            f"{len([c for c in df.columns if c.startswith('l1_')])} Layer1 + "
            f"{len([c for c in df.columns if c.startswith('l2_')])} Layer2 + "
            f"{len([c for c in df.columns if c.startswith('l3_')])} Layer3 + "
            f"{len([c for c in df.columns if c.startswith('derived_')])} Derived + "
            f"signal score/conf"
        )
        
        return df
    
    def _build_signal_without_transformer(
        self,
        macro_panel: pd.DataFrame,
        start_date: str,
        end_date: str,
        horizon: int,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Fallback: Build signal using only Layers 1-3 (no transformer).
        
        Used when:
        - PyTorch not available
        - transformers library not available
        - GPU memory issues
        """
        LOGGER.warning("⚠️ HF Transformer not available, using fallback signal from raw features")
        
        df = macro_panel.copy()
        
        # ================================================================
        # FALLBACK LAYER 5: Simple signals from derived features
        # ================================================================
        
        # 1. MACRO SCORE: Combine yield curve + GDP + credit conditions
        score_components = []
        
        if 'derived_yield_curve_slope' in df.columns:
            # Positive slope = expansion (bullish)
            score_components.append(np.tanh(df['derived_yield_curve_slope'] / 2.0) * 0.4)
        
        if 'derived_gdp_momentum' in df.columns:
            # Positive GDP growth = bullish
            score_components.append(np.tanh(df['derived_gdp_momentum'] / 2.0) * 0.3)
        
        if 'derived_credit_spread' in df.columns:
            # Wider spreads = bearish
            score_components.append(-np.tanh(df['derived_credit_spread'] / 5.0) * 0.3)
        
        if score_components:
            df['score'] = pd.concat(score_components, axis=1).mean(axis=1)
        else:
            df['score'] = 0.0
        
        # 2. CONFIDENCE: Based on data completeness
        df['conf'] = (1.0 - df.isna().mean(axis=1)).clip(0.3, 0.8)
        
        LOGGER.info(f"✅ Fallback signals: {len(df)} rows with {len(df.columns)} features")
        
        return df

    def _build_signal(
        self,
        session_date: pd.Timestamp,
        probability: float,
        confidence: float,
        horizon: int,
        symbol: str,
    ) -> ModuleSignal:
        probability = float(np.clip(probability, 0.0, 1.0))
        score = float(2.0 * probability - 1.0)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        df = pd.DataFrame({"score": [score], "conf": [confidence]}, index=[pd.to_datetime(session_date).normalize()])
        return ModuleSignal(name=self.NAME, horizon=int(horizon), df=df, symbol=symbol)

    def _build_timeseries_signal(
        self,
        bundle: _PanelBundle,
        zpanel: pd.DataFrame,
        model: TimeSeriesTransformerModel,
        head: nn.Module,
        start_date: Optional[str],
        end_date: Optional[str],
        horizon: int,
        symbol: str,
    ) -> ModuleSignal:
        """Generate predictions for each date in range with BOTH z-score and transformer features."""
        # Filter dates if specified
        dates = bundle.session_dates
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]
        if end_date:
            dates = dates[dates <= pd.Timestamp(end_date)]
        
        if len(dates) == 0:
            return ModuleSignal(name=self.NAME, horizon=int(horizon), df=pd.DataFrame(columns=["score", "conf"]), symbol=symbol)
        
        scores = []
        confs = []
        embeddings_list = []  # Store transformer embeddings
        z_composites = []  # Store z-score composites
        valid_dates = []
        
        values = zpanel.fillna(0.0).to_numpy(dtype=np.float32)
        coverage_arr = bundle.coverage.reindex(zpanel.index).fillna(0.0).to_numpy(dtype=np.float32)
        
        head = head.to(self.device)
        head.eval()
        
        max_lag = max(self.lags) if self.lags else 0
        window_size = self.context_length + max_lag  # Need extra rows for lag computation
        
        for date in dates:
            # Find index in zpanel
            try:
                idx = zpanel.index.get_loc(date)
            except KeyError:
                continue
            
            # Need at least window_size history
            if idx < window_size:
                continue
            
            window = values[idx - window_size : idx]  # Grab context_length + max_lag rows
            window_cov = coverage_arr[idx - window_size : idx]
            
            if window_cov.mean() < self.min_coverage:
                continue
            
            # Get transformer hidden state (macro embedding)
            hidden = self._hidden_for_window(model, window)
            if hidden is None:
                continue
            
            # Get regime probability from prediction head
            with torch.no_grad():
                logit = head(hidden.unsqueeze(0)).squeeze(0)
                prob = float(torch.sigmoid(logit).item())
            
            prob = float(np.clip(prob, 0.0, 1.0))
            score = float(2.0 * prob - 1.0)
            cov = float(window_cov.mean())
            conf = float(np.clip(abs(prob - 0.5) * 2.0 * cov * self.confidence_scale, 0.0, 1.0))
            
            # Also compute z-score composite for interpretability
            z_composite = float(-zpanel.iloc[idx].mean())  # Negative = bearish for stocks
            
            # Store transformer embedding (first 8 dimensions for features)
            embedding_features = hidden.cpu().numpy()[:8] if len(hidden) >= 8 else hidden.cpu().numpy()
            
            scores.append(score)
            confs.append(conf)
            z_composites.append(z_composite)
            embeddings_list.append(embedding_features)
            valid_dates.append(date)
        
        if not valid_dates:
            return ModuleSignal(name=self.NAME, horizon=int(horizon), df=pd.DataFrame(columns=["score", "conf"]), symbol=symbol)
        
        # Build DataFrame with BOTH z-score composite AND transformer embeddings
        df_data = {
            "score": scores,  # Transformer regime probability score
            "conf": confs,    # Confidence based on coverage
            "z_composite": z_composites,  # Fallback z-score composite
        }
        
        # Add embedding dimensions as features (macro_embed_1, macro_embed_2, ...)
        embeddings_array = np.array(embeddings_list)
        for i in range(embeddings_array.shape[1]):
            df_data[f"macro_embed_{i+1}"] = embeddings_array[:, i]
        
        df = pd.DataFrame(df_data, index=pd.DatetimeIndex(valid_dates).normalize())
        
        LOGGER.info(
            f"✅ Generated {len(df)} macro signals with {embeddings_array.shape[1]} embeddings "
            f"(mean score: {df['score'].mean():.3f}, z-composite std: {df['z_composite'].std():.3f})"
        )
        
        return ModuleSignal(name=self.NAME, horizon=int(horizon), df=df, symbol=symbol)

    def _build_fallback_timeseries(
        self,
        bundle: _PanelBundle,
        start_date: Optional[str],
        end_date: Optional[str],
        horizon: int,
        symbol: str,
    ) -> ModuleSignal:
        """Generate fallback time series with BOTH z-score composite AND synthetic embeddings."""
        zpanel = self._zscore_panel(bundle)
        
        # Filter dates
        dates = bundle.session_dates
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]
        if end_date:
            dates = dates[dates <= pd.Timestamp(end_date)]
        
        if len(dates) == 0:
            return ModuleSignal(name=self.NAME, horizon=int(horizon), df=pd.DataFrame(columns=["score", "conf"]), symbol=symbol)
        
        scores = []
        confs = []
        z_composites = []
        embeddings_list = []
        valid_dates = []
        
        # Use composite macro indicator (average z-score)
        for date in dates:
            if date not in zpanel.index:
                continue
            
            row = zpanel.loc[date]
            if row.isna().all():
                continue
            
            # Z-score composite: negative = good for stocks (rising rates/dollar = bearish)
            z_composite = -row.mean()
            prob = _sigmoid(self.fallback_beta * z_composite)
            
            prob = float(np.clip(prob, 0.0, 1.0))
            score = float(2.0 * prob - 1.0)
            
            cov = float(bundle.coverage.loc[date])
            conf = float(np.clip(abs(prob - 0.5) * 2.0 * cov * self.confidence_scale, 0.0, 1.0))
            
            # Create synthetic macro embeddings from raw z-scores
            # These are interpretable: each embedding = a macro indicator z-score
            macro_values = row.fillna(0.0).values
            # Pad or truncate to 8 dimensions
            if len(macro_values) >= 8:
                embeddings = macro_values[:8]
            else:
                embeddings = np.pad(macro_values, (0, 8 - len(macro_values)), constant_values=0.0)
            
            scores.append(score)
            confs.append(conf)
            z_composites.append(z_composite)
            embeddings_list.append(embeddings)
            valid_dates.append(date)
        
        if not valid_dates:
            return ModuleSignal(name=self.NAME, horizon=int(horizon), df=pd.DataFrame(columns=["score", "conf"]), symbol=symbol)
        
        # Build DataFrame with BOTH z-score composite AND embeddings
        df_data = {
            "score": scores,  # Probability-based score
            "conf": confs,    # Confidence
            "z_composite": z_composites,  # Raw z-score composite (fallback mode)
        }
        
        # Add embedding dimensions (in fallback mode, these are just the raw z-scores)
        embeddings_array = np.array(embeddings_list)
        for i in range(embeddings_array.shape[1]):
            df_data[f"macro_embed_{i+1}"] = embeddings_array[:, i]
        
        df = pd.DataFrame(df_data, index=pd.DatetimeIndex(valid_dates).normalize())
        
        LOGGER.info(
            f"✅ Generated {len(df)} macro signals using fallback method "
            f"(mean z-composite: {df['z_composite'].mean():.3f}, std: {df['z_composite'].std():.3f})"
        )
        
        return ModuleSignal(name=self.NAME, horizon=int(horizon), df=df, symbol=symbol)


__all__ = ["MacroTSTHF"]
