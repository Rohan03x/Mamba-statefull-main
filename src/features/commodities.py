#!/usr/bin/env python3
"""
Commodities features via Tiingo FX API

Fetches daily commodity price data (metals, energy) and calculates momentum, volatility, and 
relative strength features. Commodities available through Tiingo FX endpoint.
"""

from __future__ import annotations
from typing import Dict, Optional
import os
import pandas as pd
import numpy as np
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from universal_data_fetcher import get_universal_fetcher  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    get_universal_fetcher = None  # type: ignore

logger = logging.getLogger(__name__)

FALLBACK_PROXIES: Dict[str, Dict[str, str]] = {
    # Tiingo ticker → proxy ETF ticker + friendly name override
    'xauusd': {'proxy': 'GLD', 'name': 'GOLD'},
    'xagusd': {'proxy': 'SLV', 'name': 'SILVER'},
    'xptusd': {'proxy': 'PPLT', 'name': 'PLATINUM'},
    'xpdusd': {'proxy': 'PALL', 'name': 'PALLADIUM'},
    'usoilusd': {'proxy': 'USO', 'name': 'USOIL'},
    'ukoilusd': {'proxy': 'BNO', 'name': 'UKOIL'},
    'copperusd': {'proxy': 'CPER', 'name': 'COPPER'},
}


def _friendly_name(ticker: str) -> str:
    return FALLBACK_PROXIES.get(ticker.lower(), {}).get('name', ticker.upper())


def _extract_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)
    for column in ('close', 'adjclose', 'adj_close', 'price', 'last'):
        if column in df.columns:
            return df[column].sort_index()
    if df.shape[1] == 1:
        return df.iloc[:, 0].sort_index()
    return None


def _load_proxy_price_series(ticker: str, start: Optional[str], end: Optional[str]) -> Optional[pd.Series]:
    proxy_info = FALLBACK_PROXIES.get(ticker.lower())
    if proxy_info is None:
        return None

    proxy_symbol = proxy_info['proxy']
    fetcher = get_universal_fetcher() if callable(get_universal_fetcher) else None
    df = None

    if fetcher is not None:
        try:
            df = fetcher.get_stock_data(proxy_symbol, start_date=start, end_date=end)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Proxy fetcher failed for %s via universal fetcher: %s", proxy_symbol, exc)

    if (df is None or df.empty):
        try:
            import yfinance as yf  # type: ignore

            history = yf.Ticker(proxy_symbol).history(start=start, end=end)
            if history is not None and not history.empty:
                df = history
        except ImportError:
            logger.warning("yfinance not available for commodities fallback")
        except Exception as exc:  # pragma: no cover - network variability
            logger.debug("yfinance fallback failed for %s: %s", proxy_symbol, exc)

    return _extract_price_series(df) if df is not None else None


def _fetch_proxy_prices(commodities: list[str], start: Optional[str], end: Optional[str]) -> Dict[str, pd.Series]:
    proxy_data: Dict[str, pd.Series] = {}
    for ticker in commodities:
        series = _load_proxy_price_series(ticker, start, end)
        if series is not None and not series.empty:
            proxy_data[_friendly_name(ticker)] = series
    return proxy_data


def fetch(
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    commodities: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch commodity features from Tiingo FX API.
    
    Args:
        symbol: Stock symbol (used for date alignment only)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        commodities: List of Tiingo commodity tickers. Default: precious metals + energy
                    Available: 'xauusd' (Gold), 'xagusd' (Silver), 'xptusd' (Platinum),
                              'xpdusd' (Palladium), 'usoilusd' (US Oil), 'ukoilusd' (UK Oil),
                              'copperusd' (Copper)
    
    Returns:
        DataFrame with commodity momentum and volatility features
    """
    try:
        def _env_flag(name: str, default: str = "0") -> bool:
            return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

        no_proxy = _env_flag("STAGE_B_NO_PROXY_SOURCES", "0")
        eodhd_only = _env_flag("STAGE_B_EODHD_ONLY", "0")

        if eodhd_only:
            raise RuntimeError(
                "commodities family relies on Tiingo FX; STAGE_B_EODHD_ONLY=1 forbids non-EODHD sources"
            )
        token = os.getenv('TIINGO_API_TOKEN')
        fallback_used = False
        
        # Default to major commodities
        if commodities is None:
            commodities = [
                'xauusd',      # Gold
                'xagusd',      # Silver
                'usoilusd',    # US Oil
                'copperusd'    # Copper
            ]
        
        all_data: Dict[str, pd.Series] = {}
        
        if token:
            try:
                import requests
            except ImportError:
                logger.warning("requests library not available for Tiingo commodities")
            else:
                for ticker in commodities:
                    try:
                        url = f"https://api.tiingo.com/tiingo/fx/{ticker}/prices"
                        headers = {'Authorization': f'Token {token}'}
                        params = {
                            'startDate': start if start else '2020-01-01',
                            'resampleFreq': '1day'
                        }
                        response = requests.get(url, headers=headers, params=params, timeout=15)
                        if response.status_code != 200:
                            logger.debug("Tiingo FX API returned %s for %s", response.status_code, ticker)
                            continue
                        data = response.json()
                        if not data:
                            continue
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        all_data[_friendly_name(ticker)] = df['close']
                    except Exception as exc:
                        logger.debug("Failed to fetch %s from Tiingo: %s", ticker, exc)
                        continue
        else:
            logger.warning("TIINGO_API_TOKEN not found; falling back to proxy ETFs for commodities")
        
        if not all_data:
                if no_proxy:
                    raise RuntimeError(
                        "No Tiingo commodities data available and STAGE_B_NO_PROXY_SOURCES=1 forbids proxy ETF fallback"
                    )
                fallback_used = True
                all_data = _fetch_proxy_prices(commodities, start, end)
        
        if not all_data:
            empty = pd.DataFrame()
            empty.attrs['telemetry'] = {
                'status': 'dormant:no_data',
                'source': 'CommoditiesFetcher',
                'reason': 'no_tiingo_and_no_proxy'
            }
            return empty
        
        # Combine all commodity prices
        prices = pd.DataFrame(all_data)
        
        # Calculate features
        features = pd.DataFrame(index=prices.index)
        
        for commodity in prices.columns:
            # Returns
            returns = prices[commodity].pct_change()
            
            # Momentum features
            features[f'{commodity}_return_5d'] = returns.rolling(5).sum()
            features[f'{commodity}_return_20d'] = returns.rolling(20).sum()
            features[f'{commodity}_return_60d'] = returns.rolling(60).sum()
            
            # Volatility
            features[f'{commodity}_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
            features[f'{commodity}_vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
            
            # Momentum strength
            gains = returns.clip(lower=0).rolling(14).mean()
            losses = (-returns.clip(upper=0)).rolling(14).mean()
            features[f'{commodity}_strength_14d'] = gains / (losses + 1e-8)
            
            # Price level relative to recent range
            high_60 = prices[commodity].rolling(60).max()
            low_60 = prices[commodity].rolling(60).min()
            features[f'{commodity}_pct_range_60d'] = (
                (prices[commodity] - low_60) / (high_60 - low_60 + 1e-8)
            )
        
        # Precious metals basket (if available)
        metals = [col for col in prices.columns if col in ['GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM']]
        if len(metals) >= 2:
            metals_basket = prices[metals].mean(axis=1)
            returns = metals_basket.pct_change()
            features['METALS_BASKET_return_20d'] = returns.rolling(20).sum()
            features['METALS_BASKET_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Energy basket (if available)
        energy = [col for col in prices.columns if 'OIL' in col]
        if len(energy) >= 1:
            energy_basket = prices[energy].mean(axis=1)
            returns = energy_basket.pct_change()
            features['ENERGY_BASKET_return_20d'] = returns.rolling(20).sum()
            features['ENERGY_BASKET_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Filter to requested date range
        if start:
            idx = features.index
            start_ts = pd.Timestamp(start)
            if getattr(idx, 'tz', None) is not None:
                start_ts = start_ts.tz_localize(idx.tz) if start_ts.tz is None else start_ts.tz_convert(idx.tz)
            features = features[idx >= start_ts]
        if end:
            idx = features.index
            end_ts = pd.Timestamp(end)
            if getattr(idx, 'tz', None) is not None:
                end_ts = end_ts.tz_localize(idx.tz) if end_ts.tz is None else end_ts.tz_convert(idx.tz)
            features = features[idx <= end_ts]
        
        # Drop rows with all NaN
        features = features.dropna(how='all')
        
        # Ensure timezone-naive index for compatibility
        if features.index.tz is not None:
            features.index = features.index.tz_localize(None)
        
        # Add metadata
        features.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'ProxyETF' if fallback_used else 'TiingoCommodities',
            'commodities': list(all_data.keys())
        }
        features.attrs['feature_counts'] = {'generated': len(features.columns)}
        
        return features
        
    except ImportError:
        logger.warning("requests library not available for Tiingo commodities")
        if os.environ.get("STAGE_B_NO_PROXY_SOURCES", "0").strip().lower() in {"1", "true", "yes", "y", "on"} or os.environ.get(
            "STAGE_B_EODHD_ONLY", "0"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}:
            raise
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Commodities feature generation failed: {e}")
        if os.environ.get("STAGE_B_NO_PROXY_SOURCES", "0").strip().lower() in {"1", "true", "yes", "y", "on"} or os.environ.get(
            "STAGE_B_EODHD_ONLY", "0"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}:
            raise
        return pd.DataFrame()


if __name__ == '__main__':
    # Test
    result = fetch(start='2024-10-01', end='2024-10-22')
    print(f"Commodities features shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    if not result.empty:
        print(f"\nSample data:\n{result.tail()}")
