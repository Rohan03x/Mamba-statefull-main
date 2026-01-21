"""
Technical indicators for financial analysis.
Includes trend, momentum, volatility and volume indicators for comprehensive technical analysis.
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.trend import (
    MACD,
    ADXIndicator,
    CCIIndicator,
    EMAIndicator,
    PSARIndicator,
    SMAIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
)
from ta.volume import (
    AccDistIndexIndicator,
    MFIIndicator,
    OnBalanceVolumeIndicator,
    VolumeWeightedAveragePrice,
)


def add_technical_indicators(df: pd.DataFrame,
                             include_all: bool = False) -> pd.DataFrame:
    """Add technical indicators to DataFrame.

    Args:
        df: DataFrame with OHLCV columns in lowercase
        include_all: Whether to include extended set of indicators

    Returns:
        DataFrame with additional technical indicator columns
    """
    df = df.copy()

    # Basic Trend Indicators
    df['sma_20'] = ta.trend.sma_indicator(df['adj_close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['adj_close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['adj_close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['adj_close'], window=26)

    # MACD
    df['macd'] = ta.trend.macd(df['adj_close'])
    df['macd_signal'] = ta.trend.macd_signal(df['adj_close'])
    df['macd_hist'] = ta.trend.macd_diff(df['adj_close'])

    # RSI and Stochastic
    df['rsi'] = ta.momentum.rsi(df['adj_close'])
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['adj_close'])
    df['stoch_d'] = ta.momentum.stoch_signal(
        df['high'], df['low'], df['adj_close'])

    # Bollinger Bands
    df['bb_high'] = ta.volatility.bollinger_hband(df['adj_close'])
    df['bb_mid'] = ta.volatility.bollinger_mavg(df['adj_close'])
    df['bb_low'] = ta.volatility.bollinger_lband(df['adj_close'])

    if include_all:
        # Advanced Trend Indicators
        adx = ADXIndicator(df['high'], df['low'], df['adj_close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        cci = CCIIndicator(df['high'], df['low'], df['adj_close'])
        df['cci'] = cci.cci()

        psar = PSARIndicator(df['high'], df['low'], df['adj_close'])
        df['psar'] = psar.psar()

        # Advanced Momentum Indicators
        williams = WilliamsRIndicator(df['high'], df['low'], df['adj_close'])
        df['williams_r'] = williams.williams_r()

        stoch_rsi = StochRSIIndicator(df['adj_close'])
        df['stoch_rsi'] = stoch_rsi.stochrsi()
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

        # Advanced Volatility Indicators
        dc = DonchianChannel(df['high'], df['low'], df['adj_close'])
        df['dc_high'] = dc.donchian_channel_hband()
        df['dc_mid'] = dc.donchian_channel_mband()
        df['dc_low'] = dc.donchian_channel_lband()

        kc = KeltnerChannel(df['high'], df['low'], df['adj_close'])
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_mid'] = kc.keltner_channel_mband()
        df['kc_low'] = kc.keltner_channel_lband()

        # Volume Indicators
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['adj_close'],
            volume=df['volume']
        )
        df['vwap'] = vwap.volume_weighted_average_price()

        acc_dist = AccDistIndexIndicator(
            high=df['high'],
            low=df['low'],
            close=df['adj_close'],
            volume=df['volume']
        )
        df['acc_dist'] = acc_dist.acc_dist_index()

        mfi = MFIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['adj_close'],
            volume=df['volume']
        )
        df['mfi'] = mfi.money_flow_index()

    # Fill any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df


logger = logging.getLogger(__name__)


def calculate_returns(
        df: pd.DataFrame,
        price_col: str = "close") -> pd.DataFrame:
    """
    Calculate simple and log returns.

    Args:
        df: DataFrame with price data
        price_col: Column name containing prices

    Returns:
        DataFrame with additional return columns
    """
    # Simple returns
    df["simple_ret_1d"] = df[price_col].pct_change()

    # Log returns
    df["log_ret_1d"] = np.log(df[price_col] / df[price_col].shift(1))

    return df


def calculate_rolling_stats(
    df: pd.DataFrame,
    windows: List[int] = [5, 10, 20, 50],
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Calculate rolling statistics.

    Args:
        df: DataFrame with price data
        windows: List of rolling window sizes
        price_col: Column name containing prices

    Returns:
        DataFrame with additional rolling statistics columns
    """
    for window in windows:
        # Rolling mean and std
        df[f"roll_mean_{window}d"] = df[price_col].rolling(
            window=window).mean().shift(1)
        df[f"roll_std_{window}d"] = df[price_col].rolling(
            window=window).std().shift(1)

        # Rolling skew and kurtosis
        df[f"roll_skew_{window}d"] = df[price_col].rolling(
            window=window).skew().shift(1)
        df[f"roll_kurt_{window}d"] = df[price_col].rolling(
            window=window).kurt().shift(1)

        # Rolling max drawdown
        roll_max = df[price_col].rolling(window=window).max()
        df[f"roll_drawdown_{window}d"] = (df[price_col] - roll_max) / roll_max

    return df


def add_trend_indicators(
        df: pd.DataFrame,
        price_col: str = "close") -> pd.DataFrame:
    """
    Add trend indicators: EMAs, MACD.

    Args:
        df: DataFrame with price data
        price_col: Column name containing prices

    Returns:
        DataFrame with additional indicator columns
    """
    # EMAs
    ema12 = EMAIndicator(df[price_col], window=12)
    ema26 = EMAIndicator(df[price_col], window=26)
    df["ema_12"] = ema12.ema_indicator().shift(1)
    df["ema_26"] = ema26.ema_indicator().shift(1)

    # MACD
    macd = MACD(df[price_col])
    df["macd"] = macd.macd().shift(1)
    df["macd_signal"] = macd.macd_signal().shift(1)
    df["macd_dif"] = macd.macd_diff().shift(1)

    return df


def add_momentum_indicators(df: pd.DataFrame,
                            price_col: str = "close") -> pd.DataFrame:
    """
    Add momentum indicators: RSI, Stochastic RSI.

    Args:
        df: DataFrame with price data
        price_col: Column name containing prices

    Returns:
        DataFrame with additional indicator columns
    """
    # RSI
    rsi = RSIIndicator(df[price_col], window=14)
    df["rsi_14"] = rsi.rsi().shift(1)

    # Stochastic RSI
    stoch_rsi = StochRSIIndicator(df[price_col])
    df["stoch_rsi"] = stoch_rsi.stochrsi().shift(1)
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d().shift(1)
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k().shift(1)

    return df


def add_volatility_indicators(df: pd.DataFrame,
                              price_col: str = "close") -> pd.DataFrame:
    """
    Add volatility indicators: Bollinger Bands, ATR.

    Args:
        df: DataFrame with price data
        price_col: Column name containing prices

    Returns:
        DataFrame with additional indicator columns
    """
    # Bollinger Bands
    bb = BollingerBands(df[price_col], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband().shift(1)
    df["bb_low"] = bb.bollinger_lband().shift(1)
    df["bb_mid"] = bb.bollinger_mavg().shift(1)
    df["bb_width"] = ((df["bb_high"] - df["bb_low"]) / df["bb_mid"]).shift(1)

    # Average True Range
    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df[price_col],
        window=14)
    df["atr_14"] = atr.average_true_range().shift(1)

    return df


def add_volume_indicators(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume"
) -> pd.DataFrame:
    """
    Add volume indicators: OBV, MFI.

    Args:
        df: DataFrame with price data
        price_col: Column name containing prices
        volume_col: Column name containing volume

    Returns:
        DataFrame with additional indicator columns
    """
    # On Balance Volume
    obv = OnBalanceVolumeIndicator(close=df[price_col], volume=df[volume_col])
    df["obv"] = obv.on_balance_volume().shift(1)

    # Money Flow Index
    mfi = MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df[price_col],
        volume=df[volume_col],
        window=14
    )
    df["mfi_14"] = mfi.money_flow_index().shift(1)

    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all technical indicators added
    """
    df = df.copy()

    # Calculate returns first
    df = calculate_returns(df)

    # Add rolling statistics
    df = calculate_rolling_stats(df)

    # Add all technical indicators
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)

    return df


def make_supervised(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    target: str = "log_ret_1d",
    feature_cols: Optional[List[str]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create supervised learning dataset with lookback window.

    Args:
        df: DataFrame with features
        window: Number of lookback periods
        horizon: Number of periods to forecast
        target: Target column name
        feature_cols: List of feature columns to use (if None, use all except target)

    Returns:
        tuple of (X, y) where:
            X is 3D array of shape (samples, window, features)
            y is 2D array of shape (samples, horizon)
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target]

    # Get features and target
    features = df[feature_cols].values
    targets = df[target].values

    X, y = [], []

    for i in range(window, len(df) - horizon + 1):
        # Get window of features
        X.append(features[i-window:i])

        # Get future target values
        y.append(targets[i:i+horizon])

    return np.array(X), np.array(y)


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample OHLCV data to monthly frequency.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with monthly data
    """
    # Resample rules for OHLCV
    monthly = df.resample("M").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    return monthly


def add_long_term_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add long-term features for monthly data.

    Args:
        df: DataFrame with monthly OHLCV data

    Returns:
        DataFrame with additional long-term features
    """
    df = df.copy()

    # Momentum features
    for period in [3, 6, 12]:
        df[f"momentum_{period}m"] = df["close"].pct_change(period)

    # SMAs
    for period in [3, 6, 12]:
        sma = SMAIndicator(df["close"], window=period)
        df[f"sma_{period}m"] = sma.sma_indicator()

    # Drawdown statistics
    rolling_max = df["close"].expanding().max()
    drawdown = (df["close"] - rolling_max) / rolling_max

    df["max_drawdown"] = drawdown.expanding().min()
    df["avg_drawdown"] = drawdown.expanding().mean()
    df["drawdown_std"] = drawdown.expanding().std()

    # Placeholder for fundamentals (can be replaced with actual data)
    df["pe_ratio"] = np.nan
    df["eps_growth"] = np.nan

    return df
