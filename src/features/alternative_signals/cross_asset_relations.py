"""
Cross-Asset Relations Module
=============================

Extracts 5 features based on correlations and relationships with other assets.
Uses EODHD to fetch SPY, QQQ, and sector ETF prices.

Features:
1. spy_correlation_20d - 20-day rolling correlation with S&P 500
2. qqq_correlation_20d - 20-day rolling correlation with NASDAQ
3. sector_etf_correlation_20d - Correlation with relevant sector ETF
4. beta_20d - Market beta (20-day rolling)
5. beta_change_rate - Rate of change in beta
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Sector ETF mapping (simplified - expand as needed)
SECTOR_ETFS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ORCL', 'CSCO', 'AMD', 'INTC'],  # Technology
    'XLF': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB'],  # Financials
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'CMG'],  # Consumer Discretionary
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES'],  # Energy
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],  # Healthcare
    'XLI': ['BA', 'CAT', 'GE', 'UPS', 'RTX', 'HON', 'UNP', 'DE', 'LMT', 'MMM'],  # Industrials
    'XLP': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB'],  # Consumer Staples
    'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED'],  # Utilities
    'XLRE': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB'],  # Real Estate
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'DOW', 'PPG'],  # Materials
    'XLC': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'EA', 'ATVI'],  # Communication Services
}


def extract_cross_asset_relations(
    symbol: str,
    df: pd.DataFrame,
    eodhd_provider: Optional[object] = None
) -> Dict[str, pd.Series]:
    """
    Extract cross-asset correlation and beta features.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    df : pd.DataFrame
        OHLC data for the symbol with DatetimeIndex
    eodhd_provider : object, optional
        EODHD provider instance to fetch benchmark data
        
    Returns
    -------
    dict
        Dictionary of feature_name -> pd.Series
        
    Notes
    -----
    Requires EODHD provider to fetch SPY, QQQ, and sector ETF data.
    If provider not available, returns NaN series.
    """
    features = {}
    
    if df.empty or 'close' not in df.columns:
        return {f'cross_asset_{i}': pd.Series(dtype=float) for i in range(1, 6)}
    
    # Calculate symbol returns
    symbol_returns = df['close'].pct_change()
    
    if eodhd_provider is None:
        logger.warning("EODHD provider not available for cross-asset features")
        nan_series = pd.Series(np.nan, index=df.index)
        return {
            'spy_correlation_20d': nan_series.copy(),
            'qqq_correlation_20d': nan_series.copy(),
            'sector_etf_correlation_20d': nan_series.copy(),
            'beta_20d': nan_series.copy(),
            'beta_change_rate': nan_series.copy(),
        }
    
    # Fetch benchmark data
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    
    # 1. SPY Correlation
    spy_corr = _calculate_correlation(
        symbol_returns, 'SPY', start_date, end_date, eodhd_provider, window=20
    )
    features['spy_correlation_20d'] = spy_corr
    
    # 2. QQQ Correlation
    qqq_corr = _calculate_correlation(
        symbol_returns, 'QQQ', start_date, end_date, eodhd_provider, window=20
    )
    features['qqq_correlation_20d'] = qqq_corr
    
    # 3. Sector ETF Correlation
    sector_etf = _find_sector_etf(symbol)
    if sector_etf:
        sector_corr = _calculate_correlation(
            symbol_returns, sector_etf, start_date, end_date, eodhd_provider, window=20
        )
        features['sector_etf_correlation_20d'] = sector_corr
    else:
        features['sector_etf_correlation_20d'] = pd.Series(np.nan, index=df.index)
    
    # 4. Beta (20-day rolling)
    spy_returns = _get_benchmark_returns('SPY', start_date, end_date, eodhd_provider)
    if spy_returns is not None:
        beta = _calculate_rolling_beta(symbol_returns, spy_returns, window=20)
        features['beta_20d'] = beta
        
        # 5. Beta Change Rate
        # Rate of change in beta over last 5 days
        features['beta_change_rate'] = beta.pct_change(5)
    else:
        features['beta_20d'] = pd.Series(np.nan, index=df.index)
        features['beta_change_rate'] = pd.Series(np.nan, index=df.index)
    
    return features


def _find_sector_etf(symbol: str) -> Optional[str]:
    """
    Find the sector ETF for a given symbol.
    
    Returns
    -------
    str or None
        Sector ETF ticker (e.g., 'XLK') or None if not found
    """
    for etf, tickers in SECTOR_ETFS.items():
        if symbol in tickers:
            return etf
    return None


def _get_benchmark_returns(
    benchmark: str,
    start_date: str,
    end_date: str,
    eodhd_provider: object
) -> Optional[pd.Series]:
    """
    Fetch returns for a benchmark (SPY, QQQ, sector ETF).
    """
    try:
        if hasattr(eodhd_provider, 'get_eod_prices'):
            benchmark_df = eodhd_provider.get_eod_prices(
                symbol=benchmark,
                start_date=start_date,
                end_date=end_date
            )
            
            if benchmark_df is not None and not benchmark_df.empty:
                # Normalize column names to lowercase
                benchmark_df.columns = benchmark_df.columns.str.lower()
                if 'close' in benchmark_df.columns:
                    return benchmark_df['close'].pct_change()
        
        logger.warning(f"Could not fetch {benchmark} data")
        return None
        
    except Exception as e:
        logger.warning(f"Error fetching {benchmark}: {e}")
        return None


def _calculate_correlation(
    symbol_returns: pd.Series,
    benchmark: str,
    start_date: str,
    end_date: str,
    eodhd_provider: object,
    window: int = 20
) -> pd.Series:
    """
    Calculate rolling correlation with a benchmark.
    """
    benchmark_returns = _get_benchmark_returns(benchmark, start_date, end_date, eodhd_provider)
    
    if benchmark_returns is None:
        return pd.Series(np.nan, index=symbol_returns.index)
    
    # Align the two series
    aligned_df = pd.DataFrame({
        'symbol': symbol_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if aligned_df.empty:
        return pd.Series(np.nan, index=symbol_returns.index)
    
    # Calculate rolling correlation
    rolling_corr = aligned_df['symbol'].rolling(window).corr(aligned_df['benchmark'])
    
    # Reindex to original index
    return rolling_corr.reindex(symbol_returns.index)


def _calculate_rolling_beta(
    symbol_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate rolling beta using covariance method.
    
    Beta = Cov(symbol, market) / Var(market)
    """
    # Align the two series
    aligned_df = pd.DataFrame({
        'symbol': symbol_returns,
        'market': market_returns
    }).dropna()
    
    if aligned_df.empty:
        return pd.Series(np.nan, index=symbol_returns.index)
    
    # Calculate rolling covariance and variance
    rolling_cov = aligned_df['symbol'].rolling(window).cov(aligned_df['market'])
    rolling_var = aligned_df['market'].rolling(window).var()
    
    # Beta = Cov / Var
    beta = rolling_cov / rolling_var
    
    # Reindex to original index
    return beta.reindex(symbol_returns.index)
