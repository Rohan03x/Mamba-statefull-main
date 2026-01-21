"""
Downside Feature Pack v1 - Bad Tape Prediction Features

This module provides a comprehensive set of features designed to predict
market downturns and "bad tape" conditions. Features are grouped into:

1. Momentum Breakdowns: 52w-low distance, ROC acceleration
2. Volatility State: ATR percentile
3. Gap Patterns: Gap-down frequency and severity
4. Market Breadth: % advancers, new lows, A/D line slope
5. Risk Appetite: VIX term structure, credit spread proxy
6. Relative Weakness: Performance vs sector/benchmark

Author: DCF Suite Team
Date: October 16, 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class DownsideFeaturesV1:
    """
    Downside-focused feature engineering for predicting bad tape conditions.
    
    Features designed to capture:
    - Momentum breakdowns and loss of upward momentum
    - Elevated volatility and gap-down patterns
    - Weak market breadth and deteriorating internals
    - Risk-off behavior in VIX and credit markets
    - Relative weakness vs peers/benchmarks
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize downside feature calculator.
        
        Args:
            config: Optional configuration dict with parameters:
                - lookback_52w: Lookback for 52-week calculations (default: 252)
                - atr_period: ATR calculation period (default: 14)
                - gap_window: Window for gap frequency (default: 20)
                - breadth_threshold: MA period for breadth (default: 50)
                - vix_short_term: Short-term VIX window (default: 1)
                - vix_long_term: Long-term VIX window (default: 3)
        """
        self.config = config or {}
        self.lookback_52w = self.config.get('lookback_52w', 252)
        self.atr_period = self.config.get('atr_period', 14)
        self.gap_window = self.config.get('gap_window', 20)
        self.breadth_threshold = self.config.get('breadth_threshold', 50)
        self.vix_short_term = self.config.get('vix_short_term', 1)
        self.vix_long_term = self.config.get('vix_long_term', 3)
    
    def calculate_all_features(self, 
                              df: pd.DataFrame,
                              universe_data: Optional[pd.DataFrame] = None,
                              vix_data: Optional[pd.DataFrame] = None,
                              credit_data: Optional[pd.DataFrame] = None,
                              sector_data: Optional[pd.DataFrame] = None,
                              benchmark_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all downside features.
        
        Args:
            df: Main OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            universe_data: Optional multi-symbol data for breadth calculations
            vix_data: Optional VIX data for term structure
            credit_data: Optional credit spread data (e.g., HYG/TLT)
            sector_data: Optional sector ETF data for relative performance
            benchmark_data: Optional benchmark data (e.g., SPY) for relative performance
        
        Returns:
            DataFrame with original data plus all downside features
        """
        df = df.copy()
        
        # 1. Momentum Breakdowns
        df = self._add_momentum_breakdown_features(df)
        
        # 2. Volatility State
        df = self._add_volatility_state_features(df)
        
        # 3. Gap Patterns
        df = self._add_gap_features(df)
        
        # 4. Market Breadth (if universe data provided)
        if universe_data is not None:
            df = self._add_breadth_features(df, universe_data)
        
        # 5. Risk Appetite (if VIX/credit data provided)
        if vix_data is not None:
            df = self._add_vix_term_features(df, vix_data)
        if credit_data is not None:
            df = self._add_credit_spread_features(df, credit_data)
        
        # 6. Relative Weakness (if sector/benchmark data provided)
        if sector_data is not None:
            df = self._add_relative_weakness_features(df, sector_data, benchmark_data)
        
        # ✅ FIX #7 PART 3: Euphoria & Tail Risk Features
        df['euphoria_score'] = self.compute_euphoria_index(df)
        df['tail_risk_score'] = self.compute_tail_risk_score(df, vix_data)
        
        return df
    
    def _add_momentum_breakdown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum breakdown indicators.
        
        Features:
        - dist_52w_low: Distance from 52-week low (negative = at new lows)
        - roc_accel: ROC acceleration (2nd derivative, negative = deceleration)
        - momentum_breakdown: Binary flag for momentum breakdown
        """
        # 52-week low distance
        rolling_min = df['Close'].rolling(self.lookback_52w, min_periods=20).min()
        df['dist_52w_low'] = (df['Close'] - rolling_min) / df['Close']
        
        # ROC acceleration (2nd derivative)
        roc_10 = df['Close'].pct_change(10)
        df['roc_accel'] = roc_10.diff(5)  # Negative = losing momentum
        
        # Momentum breakdown flag: near 52w low + negative acceleration
        df['momentum_breakdown'] = (
            (df['dist_52w_low'] < 0.05) & (df['roc_accel'] < -0.01)
        ).astype(int)
        
        return df
    
    def _add_volatility_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility state indicators.
        
        Features:
        - ATR: Average True Range (already common)
        - ATR_pct: ATR percentile vs 1-year history
        - vol_regime: Volatility regime (0=low, 1=normal, 2=high)
        """
        # ATR calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(self.atr_period).mean()
        
        # ATR percentile vs 252-day history
        def calc_percentile(x):
            if len(x) < 2:
                return 0.5
            return (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
        
        df['ATR_pct'] = df['ATR'].rolling(self.lookback_52w, min_periods=20).apply(
            calc_percentile, raw=False
        )
        
        # Volatility regime: 0=low (<33%), 1=normal (33-66%), 2=high (>66%)
        df['vol_regime'] = pd.cut(
            df['ATR_pct'], 
            bins=[0, 0.33, 0.66, 1.0], 
            labels=[0, 1, 2]
        ).astype(float)
        
        return df
    
    def _add_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add gap-down pattern features.
        
        Features:
        - gap_pct: Daily gap percentage (Open - Close_prev) / Close_prev
        - gap_down_freq: Frequency of gap-downs in last N days
        - gap_down_severity: Average gap-down size in last N days
        """
        # Gap calculation
        df['gap_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Gap-down frequency (% of days with gap-down)
        gap_down_binary = (df['gap_pct'] < 0).astype(int)
        df['gap_down_freq'] = gap_down_binary.rolling(self.gap_window).mean()
        
        # Gap-down severity (average size of negative gaps)
        def calc_severity(x):
            negative_gaps = x[x < 0]
            return negative_gaps.mean() if len(negative_gaps) > 0 else 0.0
        
        df['gap_down_severity'] = df['gap_pct'].rolling(self.gap_window).apply(
            calc_severity, raw=False
        )
        
        return df
    
    def _add_breadth_features(self, df: pd.DataFrame, universe_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market breadth indicators.
        
        Requires universe_data with columns: ['symbol', 'date', 'close']
        
        Features:
        - pct_above_50dma: % of symbols above their 50-day MA
        - new_lows_count: Count of 52-week lows in universe
        - advance_decline_ratio: Ratio of advancing to declining stocks
        - ad_line_slope: Advance-Decline line slope
        """
        # Calculate 50-day MA for each symbol in universe
        universe_data = universe_data.copy()
        universe_data['sma_50'] = universe_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(50).mean()
        )
        universe_data['above_50dma'] = (universe_data['close'] > universe_data['sma_50']).astype(int)
        
        # % above 50-day MA by date
        breadth_by_date = universe_data.groupby('date').agg({
            'above_50dma': 'mean'
        }).rename(columns={'above_50dma': 'pct_above_50dma'})
        
        # New 52-week lows count
        universe_data['rolling_min_252'] = universe_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(252, min_periods=20).min()
        )
        universe_data['is_new_low'] = (
            universe_data['close'] <= universe_data['rolling_min_252'] * 1.01
        ).astype(int)
        
        new_lows_by_date = universe_data.groupby('date')['is_new_low'].sum().to_frame('new_lows_count')
        
        # Advance-Decline ratio (daily)
        universe_data['daily_return'] = universe_data.groupby('symbol')['close'].pct_change()
        universe_data['advancing'] = (universe_data['daily_return'] > 0).astype(int)
        universe_data['declining'] = (universe_data['daily_return'] < 0).astype(int)
        
        ad_by_date = universe_data.groupby('date').agg({
            'advancing': 'sum',
            'declining': 'sum'
        })
        ad_by_date['advance_decline_ratio'] = ad_by_date['advancing'] / (ad_by_date['declining'] + 1)
        
        # A-D line cumulative and slope
        ad_by_date['ad_line'] = (ad_by_date['advancing'] - ad_by_date['declining']).cumsum()
        ad_by_date['ad_line_slope'] = ad_by_date['ad_line'].diff(20) / 20  # 20-day slope
        
        # Merge back to main dataframe
        df['date'] = df.index
        df = df.merge(breadth_by_date, left_on='date', right_index=True, how='left')
        df = df.merge(new_lows_by_date, left_on='date', right_index=True, how='left')
        df = df.merge(
            ad_by_date[['advance_decline_ratio', 'ad_line_slope']], 
            left_on='date', 
            right_index=True, 
            how='left'
        )
        df = df.drop(columns=['date'])
        
        # Fill NaN with neutral values
        df['pct_above_50dma'] = df['pct_above_50dma'].fillna(0.5)
        df['new_lows_count'] = df['new_lows_count'].fillna(0)
        df['advance_decline_ratio'] = df['advance_decline_ratio'].fillna(1.0)
        df['ad_line_slope'] = df['ad_line_slope'].fillna(0.0)
        
        return df
    
    def _add_vix_term_features(self, df: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add VIX term structure features.
        
        Requires vix_data with columns: ['date', 'VX1', 'VX2', 'VX3', ...] or similar
        
        Features:
        - vix_term_slope: VX1 - VX3 (positive = inverted term structure = risk-off)
        - vix_term_inverted: Binary flag for term inversion
        """
        vix_data = vix_data.copy()
        
        # Calculate term slope
        if 'VX1' in vix_data.columns and 'VX3' in vix_data.columns:
            vix_data['vix_term_slope'] = vix_data['VX1'] - vix_data['VX3']
        elif 'vix_short' in vix_data.columns and 'vix_long' in vix_data.columns:
            vix_data['vix_term_slope'] = vix_data['vix_short'] - vix_data['vix_long']
        else:
            # Fallback: use VIX levels if term structure not available
            vix_data['vix_term_slope'] = 0.0
        
        vix_data['vix_term_inverted'] = (vix_data['vix_term_slope'] > 0).astype(int)
        
        # Merge to main dataframe
        df['date'] = df.index
        df = df.merge(
            vix_data[['date', 'vix_term_slope', 'vix_term_inverted']],
            on='date',
            how='left'
        )
        df = df.drop(columns=['date'])
        
        # Fill NaN
        df['vix_term_slope'] = df['vix_term_slope'].fillna(0.0)
        df['vix_term_inverted'] = df['vix_term_inverted'].fillna(0)
        
        return df
    
    def _add_credit_spread_features(self, df: pd.DataFrame, credit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add credit spread proxy features.
        
        Requires credit_data with columns: ['date', 'spread'] or ['date', 'HYG', 'TLT']
        
        Features:
        - credit_spread: Current credit spread
        - credit_spread_widening: Binary flag for widening spreads (5-day change > 0)
        - credit_spread_pct: Percentile of spread vs 252-day history
        """
        credit_data = credit_data.copy()
        
        # Calculate spread if HYG/TLT provided
        if 'HYG' in credit_data.columns and 'TLT' in credit_data.columns:
            credit_data['credit_spread'] = credit_data['TLT'] - credit_data['HYG']
        elif 'spread' not in credit_data.columns:
            credit_data['credit_spread'] = 0.0
        
        # Spread widening (5-day change)
        credit_data['credit_spread_widening'] = (
            credit_data['credit_spread'].diff(5) > 0
        ).astype(int)
        
        # Spread percentile
        def calc_percentile(x):
            if len(x) < 2:
                return 0.5
            return (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
        
        credit_data['credit_spread_pct'] = credit_data['credit_spread'].rolling(
            252, min_periods=20
        ).apply(calc_percentile, raw=False)
        
        # Merge to main dataframe
        df['date'] = df.index
        df = df.merge(
            credit_data[['date', 'credit_spread', 'credit_spread_widening', 'credit_spread_pct']],
            on='date',
            how='left'
        )
        df = df.drop(columns=['date'])
        
        # Fill NaN
        df['credit_spread'] = df['credit_spread'].fillna(0.0)
        df['credit_spread_widening'] = df['credit_spread_widening'].fillna(0)
        df['credit_spread_pct'] = df['credit_spread_pct'].fillna(0.5)
        
        return df
    
    def _add_relative_weakness_features(self, 
                                       df: pd.DataFrame, 
                                       sector_data: Optional[pd.DataFrame] = None,
                                       benchmark_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add relative performance features.
        
        Features:
        - rel_perf_sector: 20-day return vs sector ETF
        - rel_perf_benchmark: 20-day return vs benchmark (e.g., SPY)
        - beta_20: 20-day rolling beta vs benchmark
        - beta_drift: 5-day change in beta (increasing beta = more systematic risk)
        """
        # Calculate symbol returns
        df['return_20'] = df['Close'].pct_change(20)
        
        # Relative performance vs sector
        if sector_data is not None:
            sector_data = sector_data.copy()
            sector_data['sector_return_20'] = sector_data['Close'].pct_change(20)
            
            df['date'] = df.index
            df = df.merge(
                sector_data[['date', 'sector_return_20']].rename(columns={'date': 'date'}),
                left_on='date',
                right_on='date',
                how='left'
            )
            df = df.drop(columns=['date'])
            df['sector_return_20'] = df['sector_return_20'].fillna(0)
            df['rel_perf_sector'] = df['return_20'] - df['sector_return_20']
        else:
            df['rel_perf_sector'] = 0.0
        
        # Relative performance vs benchmark + beta
        if benchmark_data is not None:
            benchmark_data = benchmark_data.copy()
            benchmark_data['benchmark_return_20'] = benchmark_data['Close'].pct_change(20)
            benchmark_data['benchmark_return_daily'] = benchmark_data['Close'].pct_change()
            
            df['date'] = df.index
            df['symbol_return_daily'] = df['Close'].pct_change()
            
            df = df.merge(
                benchmark_data[['date', 'benchmark_return_20', 'benchmark_return_daily']],
                left_on='date',
                right_on='date',
                how='left'
            )
            df = df.drop(columns=['date'])
            
            df['benchmark_return_20'] = df['benchmark_return_20'].fillna(0)
            df['benchmark_return_daily'] = df['benchmark_return_daily'].fillna(0)
            df['rel_perf_benchmark'] = df['return_20'] - df['benchmark_return_20']
            
            # Rolling 20-day beta
            df['beta_20'] = df['symbol_return_daily'].rolling(20).cov(
                df['benchmark_return_daily']
            ) / df['benchmark_return_daily'].rolling(20).var()
            
            # Beta drift
            df['beta_drift'] = df['beta_20'].diff(5)
            
            # Clean up temporary columns
            df = df.drop(columns=['symbol_return_daily', 'benchmark_return_daily', 
                                 'benchmark_return_20', 'sector_return_20'], errors='ignore')
        else:
            df['rel_perf_benchmark'] = 0.0
            df['beta_20'] = 1.0
            df['beta_drift'] = 0.0
        
        # Clean up
        df = df.drop(columns=['return_20'], errors='ignore')
        
        return df
    
    # ========================================================================
    # ✅ FIX #7 PART 3: Euphoria & Tail Risk Features
    # ========================================================================
    
    def compute_euphoria_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute euphoria index (0-100) based on overbought conditions.
        
        Euphoria signals extreme bullishness:
            - Price >> MA200 (e.g., +20% above)
            - RSI > 70 (overbought)
            - Low volatility (vol_T3 low = complacency)
        
        Formula:
            euphoria_score = 0.4 * price_vs_ma200_score + 
                            0.4 * rsi_score + 
                            0.2 * vol_complacency_score
        
        Args:
            df: DataFrame with columns: Close, (optional: MA200, RSI, vol_T3)
        
        Returns:
            Series: euphoria_score (0-100, higher = more euphoric)
        
        Environment:
            ENABLE_EUPHORIA_FEATURES: Set to 1 to enable (default: 0)
        
        Examples:
            >>> df['euphoria_score'] = calculator.compute_euphoria_index(df)
        """
        import os
        
        if not int(os.environ.get('ENABLE_EUPHORIA_FEATURES', '0')):
            return pd.Series(0.0, index=df.index)
        
        # Calculate MA200 if not present
        if 'MA200' not in df.columns:
            df['MA200'] = df['Close'].rolling(200).mean()
        
        # Calculate RSI if not present
        if 'RSI' not in df.columns:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate vol_T3 proxy (20-day rolling volatility) if not present
        if 'vol_T3' not in df.columns:
            df['vol_T3'] = df['Close'].pct_change().rolling(20).std()
        
        # Score 1: Price vs MA200 (0-100)
        # 0% above MA200 → 0, +20%+ above MA200 → 100
        price_vs_ma200 = ((df['Close'] - df['MA200']) / df['MA200']).fillna(0)
        price_score = (price_vs_ma200 * 5 * 100).clip(0, 100)  # 20% = 100
        
        # Score 2: RSI (0-100)
        # RSI 50 → 0, RSI 70+ → 100
        rsi_score = ((df['RSI'] - 50) / 20 * 100).clip(0, 100).fillna(0)
        
        # Score 3: Vol complacency (0-100)
        # Low vol_T3 → high score (complacency)
        # High vol_T3 → low score
        vol_percentile = df['vol_T3'].rolling(252).rank(pct=True).fillna(0.5)
        vol_complacency_score = (1 - vol_percentile) * 100  # Invert: low vol = high score
        
        # Combined euphoria score
        euphoria_score = (
            0.4 * price_score +
            0.4 * rsi_score +
            0.2 * vol_complacency_score
        )
        
        return euphoria_score.fillna(0)
    
    def compute_tail_risk_score(self, 
                                df: pd.DataFrame,
                                vix_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Compute tail risk score (0-100) combining VIX, volatility, and drawdown proximity.
        
        Tail risk captures extreme downside risk conditions:
            - VIX spike (VIX > 30 or sudden jump)
            - vol_T3 uptrend (sustained volatility rise)
            - Proximity to drawdown levels (near 52w low)
        
        Formula:
            tail_risk = 0.4 * vix_risk_score + 
                       0.3 * vol_trend_score + 
                       0.3 * drawdown_proximity_score
        
        Args:
            df: DataFrame with columns: Close, High, Low
            vix_data: Optional VIX DataFrame with 'Close' column
        
        Returns:
            Series: tail_risk_score (0-100, higher = more tail risk)
        
        Environment:
            ENABLE_TAIL_RISK_FEATURES: Set to 1 to enable (default: 0)
        
        Examples:
            >>> df['tail_risk'] = calculator.compute_tail_risk_score(df, vix_df)
        """
        import os
        
        if not int(os.environ.get('ENABLE_TAIL_RISK_FEATURES', '0')):
            return pd.Series(0.0, index=df.index)
        
        # Score 1: VIX risk (0-100)
        if vix_data is not None and 'Close' in vix_data.columns:
            # Align VIX to df index
            vix = vix_data['Close'].reindex(df.index, method='ffill')
            
            # VIX 15 → 0, VIX 30+ → 100
            vix_risk = ((vix - 15) / 15 * 100).clip(0, 100).fillna(0)
            
            # Add VIX jump component: sudden +5pt jump → +50 to score
            vix_jump = vix.diff().clip(-10, 10)  # Limit to ±10pts
            vix_jump_score = (vix_jump / 10 * 50).clip(0, 50).fillna(0)
            
            vix_risk_score = vix_risk + vix_jump_score
        else:
            vix_risk_score = pd.Series(0.0, index=df.index)
        
        # Score 2: Vol trend (0-100)
        # Calculate vol_T3 if not present
        if 'vol_T3' not in df.columns:
            df['vol_T3'] = df['Close'].pct_change().rolling(20).std()
        
        # Count rising vol_T3 days in last 10 days
        vol_rising = (df['vol_T3'].diff() > 0).rolling(10).sum()
        vol_trend_score = (vol_rising / 10 * 100).fillna(0)  # 10/10 rising → 100
        
        # Score 3: Drawdown proximity (0-100)
        # Calculate 52w high
        high_52w = df['High'].rolling(252).max()
        drawdown = ((df['Close'] - high_52w) / high_52w).fillna(0)
        
        # -5% DD → 0, -20%+ DD → 100
        drawdown_proximity_score = ((-drawdown - 0.05) / 0.15 * 100).clip(0, 100)
        
        # Combined tail risk score
        tail_risk = (
            0.4 * vix_risk_score +
            0.3 * vol_trend_score +
            0.3 * drawdown_proximity_score
        )
        
        return tail_risk.fillna(0)
    
    # ========================================================================
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names in this bundle."""
        return [
            # Momentum breakdowns
            'dist_52w_low',
            'roc_accel',
            'momentum_breakdown',
            
            # Volatility state
            'ATR',
            'ATR_pct',
            'vol_regime',
            
            # Gap patterns
            'gap_pct',
            'gap_down_freq',
            'gap_down_severity',
            
            # Market breadth
            'pct_above_50dma',
            'new_lows_count',
            'advance_decline_ratio',
            'ad_line_slope',
            
            # Risk appetite
            'vix_term_slope',
            'vix_term_inverted',
            'credit_spread',
            'credit_spread_widening',
            'credit_spread_pct',
            
            # Relative weakness
            'rel_perf_sector',
            'rel_perf_benchmark',
            'beta_20',
            'beta_drift',
            
            # ✅ FIX #7 PART 3: Euphoria & Tail Risk
            'euphoria_score',
            'tail_risk_score',
        ]


# Feature bundle registry
FEATURE_BUNDLES = {
    'downside_v1': DownsideFeaturesV1,
    'baseline_v1': None,  # Use default features from adaptive system
}


def get_feature_bundle(name: str, config: Optional[Dict] = None):
    """
    Get feature bundle calculator by name.
    
    Args:
        name: Feature bundle name (e.g., 'downside_v1', 'baseline_v1')
        config: Optional configuration dict
    
    Returns:
        Feature calculator instance or None for baseline
    
    Example:
        >>> calculator = get_feature_bundle('downside_v1')
        >>> df_with_features = calculator.calculate_all_features(df)
    """
    if name not in FEATURE_BUNDLES:
        raise ValueError(f"Unknown feature bundle: {name}. Available: {list(FEATURE_BUNDLES.keys())}")
    
    bundle_class = FEATURE_BUNDLES[name]
    if bundle_class is None:
        return None  # baseline uses default features
    
    return bundle_class(config=config)


if __name__ == '__main__':
    # Simple test
    print("✅ Downside Features v1 module loaded successfully")
    print(f"   Available bundles: {list(FEATURE_BUNDLES.keys())}")
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    test_df = pd.DataFrame({
        'Open': np.random.randn(300).cumsum() + 100,
        'High': np.random.randn(300).cumsum() + 102,
        'Low': np.random.randn(300).cumsum() + 98,
        'Close': np.random.randn(300).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)
    
    # Test feature calculation
    calculator = get_feature_bundle('downside_v1')
    result_df = calculator.calculate_all_features(test_df)
    
    print(f"\n📊 Test calculation complete:")
    print(f"   Input shape: {test_df.shape}")
    print(f"   Output shape: {result_df.shape}")
    print(f"   New features added: {result_df.shape[1] - test_df.shape[1]}")
    print(f"\n   Feature names: {calculator.get_feature_names()}")
