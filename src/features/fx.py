#!/usr/bin/env python3
"""
Foreign Exchange (FX) features via Tiingo FX API

Fetches daily FX pair data and calculates currency strength, momentum, and volatility features.
Supports 136 FX pairs including majors, crosses, and metals.
"""

from __future__ import annotations
from typing import Optional
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

logger = logging.getLogger(__name__)


def fetch(
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fx_pairs: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch FX features from Tiingo FX API.
    
    Args:
        symbol: Stock symbol (used for date alignment only)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        fx_pairs: List of Tiingo FX tickers (e.g., ['eurusd', 'gbpusd', 'usdjpy'])
                 Default: Major pairs ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad']
    
    Returns:
        DataFrame with FX momentum, volatility, and carry features
    """
    try:
        import requests
        
        token = os.getenv('TIINGO_API_TOKEN')
        if not token:
            logger.warning("TIINGO_API_TOKEN not found in environment")
            return pd.DataFrame()
        
        # Default to major FX pairs
        if fx_pairs is None:
            fx_pairs = ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad']
        
        all_data = {}
        
        for pair in fx_pairs:
            try:
                url = f"https://api.tiingo.com/tiingo/fx/{pair}/prices"
                headers = {'Authorization': f'Token {token}'}
                params = {
                    'startDate': start if start else '2020-01-01',
                    'resampleFreq': '1day'
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=15)
                
                if response.status_code != 200:
                    logger.debug(f"Tiingo FX API returned {response.status_code} for {pair}")
                    continue
                
                data = response.json()
                if not data or len(data) == 0:
                    continue
                
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Store close prices
                all_data[pair.upper()] = df['close']
                
            except Exception as e:
                logger.debug(f"Failed to fetch {pair}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all FX prices
        prices = pd.DataFrame(all_data)
        
        # Calculate features
        features = pd.DataFrame(index=prices.index)
        
        for pair in prices.columns:
            # Returns
            returns = prices[pair].pct_change()
            
            # Momentum features
            features[f'{pair}_return_5d'] = returns.rolling(5).sum()
            features[f'{pair}_return_20d'] = returns.rolling(20).sum()
            features[f'{pair}_return_60d'] = returns.rolling(60).sum()
            
            # Volatility
            features[f'{pair}_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
            
            # Momentum strength (RSI-like)
            gains = returns.clip(lower=0).rolling(14).mean()
            losses = (-returns.clip(upper=0)).rolling(14).mean()
            features[f'{pair}_strength_14d'] = gains / (losses + 1e-8)
        
        # Calculate USD index (average of USD pairs)
        usd_pairs = [col for col in prices.columns if 'USD' in col]
        if len(usd_pairs) >= 3:
            # For pairs like EURUSD, GBPUSD (USD is quote), invert to get USD strength
            # For pairs like USDJPY, USDCAD (USD is base), use as-is
            usd_strength = pd.Series(0.0, index=prices.index)
            for pair in usd_pairs:
                if pair.startswith('USD'):
                    usd_strength += prices[pair].pct_change()
                else:
                    usd_strength -= prices[pair].pct_change()
            usd_strength = usd_strength / len(usd_pairs)
            features['USD_INDEX_change_5d'] = usd_strength.rolling(5).sum()
            features['USD_INDEX_change_20d'] = usd_strength.rolling(20).sum()
        
        # Filter to requested date range
        if start:
            start_ts = pd.Timestamp(start, tz='UTC')
            features = features[features.index >= start_ts]
        if end:
            end_ts = pd.Timestamp(end, tz='UTC')
            features = features[features.index <= end_ts]
        
        # Drop rows with all NaN
        features = features.dropna(how='all')
        
        # Ensure timezone-naive index for compatibility
        if features.index.tz is not None:
            features.index = features.index.tz_localize(None)
        
        # Add metadata
        features.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'TiingoFX',
            'pairs': list(all_data.keys())
        }
        features.attrs['feature_counts'] = {'generated': len(features.columns)}
        
        return features
        
    except ImportError:
        logger.warning("requests library not available for Tiingo FX")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"FX feature generation failed: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    # Test
    result = fetch(start='2024-10-01', end='2024-10-22')
    print(f"FX features shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    if not result.empty:
        print(f"\nSample data:\n{result.tail()}")
