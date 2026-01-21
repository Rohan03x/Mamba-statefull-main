#!/usr/bin/env python3
"""
Cryptocurrency features via Tiingo Crypto API

Fetches daily crypto price data and calculates momentum, volatility, and relative strength features.
Supports 7,477+ crypto pairs across multiple exchanges.
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
    crypto_tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch cryptocurrency features from Tiingo Crypto API.
    
    Args:
        symbol: Stock symbol (used for date alignment only, not for crypto selection)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        crypto_tickers: List of Tiingo crypto tickers (e.g., ['btcusd', 'ethusd', 'solusd'])
                       Default: ['btcusd', 'ethusd', 'bnbusd']
    
    Returns:
        DataFrame with crypto momentum and volatility features
    """
    try:
        import requests
        
        token = os.getenv('TIINGO_API_TOKEN')
        if not token:
            logger.warning("TIINGO_API_TOKEN not found in environment")
            return pd.DataFrame()
        
        # Default to major cryptos
        if crypto_tickers is None:
            crypto_tickers = ['btcusd', 'ethusd', 'bnbusd']
        
        # Tiingo crypto requires bulk endpoint with comma-separated tickers
        tickers_param = ','.join(crypto_tickers)
        
        try:
            url = "https://api.tiingo.com/tiingo/crypto/prices"
            headers = {'Authorization': f'Token {token}'}
            params = {
                'tickers': tickers_param,
                'startDate': start if start else '2020-01-01',
                'resampleFreq': '1day'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.debug(f"Tiingo Crypto API returned {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            if not data or len(data) == 0:
                return pd.DataFrame()
            
            all_data = {}
            
            # Process each ticker's data
            for ticker_data in data:
                ticker = ticker_data.get('ticker', '')
                price_data = ticker_data.get('priceData', [])
                
                if not price_data:
                    continue
                
                df = pd.DataFrame(price_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Store close prices
                crypto_name = ticker.replace('usd', '').replace('usdt', '').upper()
                all_data[crypto_name] = df['close']
        
        except Exception as e:
            logger.debug(f"Failed to fetch crypto data: {e}")
            return pd.DataFrame()
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all crypto prices
        prices = pd.DataFrame(all_data)
        
        # Calculate features
        features = pd.DataFrame(index=prices.index)
        
        for crypto in prices.columns:
            # Returns
            returns = prices[crypto].pct_change()
            
            # Momentum features
            features[f'{crypto}_return_5d'] = returns.rolling(5).sum()
            features[f'{crypto}_return_20d'] = returns.rolling(20).sum()
            
            # Volatility
            features[f'{crypto}_vol_20d'] = returns.rolling(20).std() * np.sqrt(365)
            
            # RSI-like momentum
            gains = returns.clip(lower=0).rolling(14).mean()
            losses = (-returns.clip(upper=0)).rolling(14).mean()
            features[f'{crypto}_momentum_14d'] = gains / (losses + 1e-8)
        
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
            'source': 'TiingoCrypto',
            'tickers': list(all_data.keys())
        }
        features.attrs['feature_counts'] = {'generated': len(features.columns)}
        
        return features
        
    except ImportError:
        logger.warning("requests library not available for Tiingo Crypto")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Crypto feature generation failed: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    # Test
    result = fetch(start='2024-10-01', end='2024-10-22')
    print(f"Crypto features shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    if not result.empty:
        print(f"\nSample data:\n{result.tail()}")
