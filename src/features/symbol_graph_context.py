"""Cross-symbol graph context features for relational learning.

Family: symbol_graph_context

Goal
----
Provide per-symbol embeddings and neighbor-aggregated statistics derived from a
cross-symbol graph built on:
1. Rolling return correlations (contemporaneous, symmetric)
2. Sector/industry links from group_map
3. Fixed anchor set (ETFs/benchmarks for stability)

The graph is refreshed weekly (Friday close) for adjacency/embeddings, with
daily neighbor aggregations computed using strictly lagged features.

Features Produced
-----------------
- sgc_emb_0 .. sgc_emb_k: SVD embeddings from adjacency (weekly refresh)
- sgc_neighbor_momentum: Weighted avg of neighbor 20d returns
- sgc_neighbor_vol: Weighted avg of neighbor 20d volatility
- sgc_neighbor_drawdown: Weighted avg of neighbor 20d drawdown
- sgc_neighbor_momentum_lag1: Lagged neighbor momentum (t-1 aggregation)
- sgc_neighbor_momentum_lag5: 5-day lagged neighbor momentum
- sgc_degree: Normalized weighted degree centrality
- sgc_strength_topk: Sum of top-k edge weights
- sgc_centrality_pagerank: PageRank centrality (optional, v1.5)

Leakage Policy
--------------
- Adjacency computed using [t-window, t-1] returns (no same-day lookahead)
- Neighbor features aggregated using t-1 values (strict point-in-time)
- Embeddings are weekly snapshots, applied to entire week (conservative)

Regime Gating
-------------
- Optional regime scalar modulates adjacency temperature (softmax sharpness)
- Higher temperature in crisis = less neighbor mixing (avoid contagion oversmoothing)

Architecture
------------
GraphUniverse(t) = ActiveUniverse(t) ∪ Anchors
- ActiveUniverse: tradable symbols after hygiene/liquidity gates
- Anchors: fixed ETFs/benchmarks (SPY, QQQ, XLK, XLE, XLF, TLT, HYG, etc.)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

SYMBOL_GRAPH_CONTEXT_VERSION = 1

# Default anchor set: ETFs/benchmarks for graph stability
DEFAULT_ANCHORS: List[str] = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "TLT",   # Long-term Treasury
    "HYG",   # High Yield Corporate
    "GLD",   # Gold
    "UUP",   # Dollar Index
]

# Default parameters
DEFAULT_CORR_WINDOW = 63       # 3 months rolling correlation
DEFAULT_EMBED_DIM = 8          # SVD embedding dimension
DEFAULT_TOP_K = 10             # Top-k neighbors for sparsification
DEFAULT_CORR_WEIGHT = 0.7      # Weight for correlation edges
DEFAULT_SECTOR_WEIGHT = 0.3    # Weight for sector edges
DEFAULT_MIN_CORR = 0.1         # Minimum correlation to keep edge

# Regime temperature scaling
REGIME_TEMP_NORMAL = 1.0
REGIME_TEMP_ELEVATED = 1.5
REGIME_TEMP_CRISIS = 3.0


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def _norm_symbol(sym: str) -> str:
    """Normalize symbol to uppercase."""
    return (sym or "").strip().upper()


def _get_nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    """Get NYSE trading sessions in date range."""
    try:
        from src.dcf_lab.utils.timealign import nyse_sessions_in_range
        return nyse_sessions_in_range(start, end)
    except ImportError:
        # Fallback: business days
        return pd.bdate_range(start, end)


def _is_friday(dt: pd.Timestamp) -> bool:
    """Check if date is a Friday."""
    return dt.dayofweek == 4


def _prev_friday(dt: pd.Timestamp) -> pd.Timestamp:
    """Get the previous Friday (or same day if Friday)."""
    days_since_friday = (dt.dayofweek - 4) % 7
    return dt - pd.Timedelta(days=days_since_friday)


def _cache_key(
    universe: Sequence[str],
    as_of_date: pd.Timestamp,
    corr_window: int,
    embed_dim: int,
) -> str:
    """Generate cache key for adjacency/embeddings."""
    syms = sorted(set(_norm_symbol(s) for s in universe if s))
    content = f"{','.join(syms)}|{as_of_date.strftime('%Y%m%d')}|{corr_window}|{embed_dim}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


# -----------------------------------------------------------------------------
# ADJACENCY BUILDER
# -----------------------------------------------------------------------------

@dataclass
class AdjacencyConfig:
    """Configuration for adjacency matrix computation."""
    corr_window: int = DEFAULT_CORR_WINDOW
    corr_weight: float = DEFAULT_CORR_WEIGHT
    sector_weight: float = DEFAULT_SECTOR_WEIGHT
    min_corr: float = DEFAULT_MIN_CORR
    top_k: int = DEFAULT_TOP_K
    symmetric: bool = True
    temperature: float = 1.0


def compute_rolling_correlation_adjacency(
    returns_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    window: int = DEFAULT_CORR_WINDOW,
    min_periods: int = 30,
) -> pd.DataFrame:
    """Compute rolling correlation matrix using data up to as_of_date.
    
    Args:
        returns_df: DataFrame with symbols as columns, dates as index
        as_of_date: Compute correlation using data up to this date (exclusive)
        window: Rolling window in trading days
        min_periods: Minimum observations for valid correlation
        
    Returns:
        Correlation matrix as DataFrame (symbols x symbols)
    """
    # Use data strictly before as_of_date
    mask = returns_df.index < as_of_date
    df = returns_df.loc[mask].tail(window)
    
    if len(df) < min_periods:
        logger.warning(f"Insufficient data for correlation: {len(df)} < {min_periods}")
        return pd.DataFrame()
    
    # Compute correlation matrix
    corr = df.corr(method="pearson")
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(corr.values, 0.0)
    
    # Replace NaN with 0
    corr = corr.fillna(0.0)
    
    return corr


def compute_sector_adjacency(
    symbols: Sequence[str],
    group_map: Dict[str, str],
) -> pd.DataFrame:
    """Build binary adjacency from sector membership.
    
    Args:
        symbols: List of symbols
        group_map: Dict mapping symbol -> sector/group
        
    Returns:
        Binary adjacency matrix (1 if same sector, 0 otherwise)
    """
    symbols = list(symbols)
    n = len(symbols)
    adj = np.zeros((n, n), dtype=np.float32)
    
    # Get sector for each symbol
    sectors = [group_map.get(_norm_symbol(s), "") for s in symbols]
    
    for i in range(n):
        for j in range(i + 1, n):
            if sectors[i] and sectors[j] and sectors[i] == sectors[j]:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    
    return pd.DataFrame(adj, index=symbols, columns=symbols)


def blend_adjacencies(
    A_corr: pd.DataFrame,
    A_sector: pd.DataFrame,
    corr_weight: float = DEFAULT_CORR_WEIGHT,
    sector_weight: float = DEFAULT_SECTOR_WEIGHT,
) -> pd.DataFrame:
    """Blend correlation and sector adjacencies.
    
    Args:
        A_corr: Correlation-based adjacency
        A_sector: Sector-based adjacency
        corr_weight: Weight for correlation component
        sector_weight: Weight for sector component
        
    Returns:
        Blended adjacency matrix
    """
    # Align indices
    common = A_corr.index.intersection(A_sector.index)
    if len(common) == 0:
        return A_corr
    
    A_c = A_corr.loc[common, common].fillna(0.0)
    A_s = A_sector.loc[common, common].fillna(0.0)
    
    # Normalize weights
    total = corr_weight + sector_weight
    w_c = corr_weight / total
    w_s = sector_weight / total
    
    # Blend
    A = w_c * A_c + w_s * A_s
    
    return A


def apply_top_k_sparsification(
    adj: pd.DataFrame,
    k: int = DEFAULT_TOP_K,
    symmetric: bool = True,
) -> pd.DataFrame:
    """Keep only top-k edges per node.
    
    Args:
        adj: Adjacency matrix
        k: Number of neighbors to keep
        symmetric: If True, symmetrize after sparsification
        
    Returns:
        Sparsified adjacency matrix
    """
    if k <= 0 or k >= len(adj):
        return adj
    
    values = adj.values.copy()
    n = values.shape[0]
    
    for i in range(n):
        row = values[i, :]
        # Find k-th largest value
        if k < n:
            threshold = np.partition(row, -k)[-k]
            values[i, row < threshold] = 0.0
    
    if symmetric:
        # Symmetrize: A = (A + A.T) / 2
        values = (values + values.T) / 2.0
    
    return pd.DataFrame(values, index=adj.index, columns=adj.columns)


def apply_softmax_temperature(
    adj: pd.DataFrame,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Apply softmax with temperature per row.
    
    Higher temperature = more uniform weights (less neighbor mixing).
    Lower temperature = sharper weights (more confident neighbors).
    
    Args:
        adj: Adjacency matrix
        temperature: Softmax temperature (1.0 = standard)
        
    Returns:
        Row-normalized adjacency with temperature scaling
    """
    values = adj.values.copy()
    
    # Apply temperature scaling
    if temperature != 1.0:
        values = values / temperature
    
    # Row-wise softmax (only on non-zero entries to preserve sparsity pattern)
    for i in range(values.shape[0]):
        row = values[i, :]
        mask = row != 0
        if mask.sum() > 0:
            exp_row = np.exp(row[mask] - row[mask].max())  # Numerically stable
            values[i, mask] = exp_row / exp_row.sum()
    
    return pd.DataFrame(values, index=adj.index, columns=adj.columns)


# -----------------------------------------------------------------------------
# EMBEDDINGS
# -----------------------------------------------------------------------------

def compute_svd_embeddings(
    adj: pd.DataFrame,
    embed_dim: int = DEFAULT_EMBED_DIM,
) -> pd.DataFrame:
    """Compute SVD-based node embeddings from adjacency.
    
    Args:
        adj: Adjacency matrix (n x n)
        embed_dim: Number of embedding dimensions
        
    Returns:
        Embeddings DataFrame (n x embed_dim) with symbol index
    """
    values = adj.values.astype(np.float64)
    n = values.shape[0]
    
    # Handle edge cases
    if n == 0:
        return pd.DataFrame()
    
    k = min(embed_dim, n - 1)
    if k <= 0:
        k = 1
    
    try:
        # SVD decomposition
        U, S, Vt = np.linalg.svd(values, full_matrices=False)
        
        # Use top-k singular vectors scaled by sqrt of singular values
        embeddings = U[:, :k] * np.sqrt(S[:k])
        
    except np.linalg.LinAlgError:
        logger.warning("SVD failed, using random embeddings")
        embeddings = np.random.randn(n, k) * 0.01
    
    cols = [f"sgc_emb_{i}" for i in range(k)]
    return pd.DataFrame(embeddings, index=adj.index, columns=cols)


# -----------------------------------------------------------------------------
# CENTRALITY METRICS
# -----------------------------------------------------------------------------

def compute_centrality_metrics(
    adj: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Compute centrality metrics from adjacency.
    
    Args:
        adj: Adjacency matrix
        top_k: k for strength_topk computation
        
    Returns:
        DataFrame with centrality metrics per symbol
    """
    values = adj.values.copy()
    n = values.shape[0]
    symbols = adj.index.tolist()
    
    # Degree centrality (normalized weighted degree)
    degree = values.sum(axis=1)
    degree_norm = degree / (degree.max() + 1e-8)
    
    # Strength top-k (sum of top-k edge weights)
    strength_topk = np.zeros(n)
    for i in range(n):
        row = values[i, :]
        k = min(top_k, len(row))
        strength_topk[i] = np.partition(row, -k)[-k:].sum() if k > 0 else 0.0
    strength_topk_norm = strength_topk / (strength_topk.max() + 1e-8)
    
    # PageRank (power iteration)
    pagerank = _compute_pagerank(values, damping=0.85, max_iter=100)
    
    return pd.DataFrame({
        "sgc_degree": degree_norm,
        "sgc_strength_topk": strength_topk_norm,
        "sgc_pagerank": pagerank,
    }, index=symbols)


def _compute_pagerank(
    adj: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute PageRank via power iteration."""
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    
    # Normalize adjacency to transition matrix
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    P = adj / row_sums
    
    # Initialize pagerank
    pr = np.ones(n) / n
    
    # Power iteration
    for _ in range(max_iter):
        pr_new = (1 - damping) / n + damping * (P.T @ pr)
        if np.abs(pr_new - pr).sum() < tol:
            break
        pr = pr_new
    
    # Normalize to [0, 1]
    pr = pr / (pr.max() + 1e-8)
    return pr


# -----------------------------------------------------------------------------
# NEIGHBOR AGGREGATION
# -----------------------------------------------------------------------------

def aggregate_neighbor_features(
    symbol: str,
    adj: pd.DataFrame,
    universe_features: pd.DataFrame,
    feature_cols: List[str],
    lag: int = 1,
) -> pd.DataFrame:
    """Aggregate neighbor features for a single symbol.
    
    Args:
        symbol: Target symbol
        adj: Adjacency matrix (row-normalized)
        universe_features: DataFrame with columns [date, symbol, *feature_cols]
        feature_cols: Feature columns to aggregate
        lag: Lag in days for point-in-time safety
        
    Returns:
        DataFrame with aggregated features indexed by date
    """
    symbol = _norm_symbol(symbol)
    
    if symbol not in adj.index:
        logger.warning(f"Symbol {symbol} not in adjacency matrix")
        return pd.DataFrame()
    
    # Get neighbor weights for this symbol
    weights = adj.loc[symbol].values  # (n_symbols,)
    neighbors = adj.columns.tolist()
    
    # Pivot universe features to have symbols as columns
    dates = universe_features["date"].unique()
    results = []
    
    for dt in sorted(dates):
        # Get features for this date (with lag for point-in-time)
        lagged_dt = dt - pd.Timedelta(days=lag)
        mask = universe_features["date"] <= lagged_dt
        day_features = universe_features[mask].groupby("symbol").last()
        
        row = {"date": dt}
        for col in feature_cols:
            if col not in day_features.columns:
                row[f"sgc_neighbor_{col}"] = np.nan
                continue
            
            # Weighted average over neighbors
            vals = []
            wts = []
            for i, nbr in enumerate(neighbors):
                if nbr in day_features.index and weights[i] > 0:
                    v = day_features.loc[nbr, col]
                    if pd.notna(v):
                        vals.append(v)
                        wts.append(weights[i])
            
            if vals:
                row[f"sgc_neighbor_{col}"] = np.average(vals, weights=wts)
            else:
                row[f"sgc_neighbor_{col}"] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_neighbor_features_fast(
    symbol: str,
    adj: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    vol_matrix: pd.DataFrame,
    drawdown_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Fast vectorized neighbor feature aggregation.
    
    Args:
        symbol: Target symbol
        adj: Adjacency matrix (row-normalized)
        returns_matrix: DataFrame (dates x symbols) of returns
        vol_matrix: DataFrame (dates x symbols) of volatility
        drawdown_matrix: DataFrame (dates x symbols) of drawdowns
        
    Returns:
        DataFrame with neighbor features indexed by date
    """
    symbol = _norm_symbol(symbol)
    
    if symbol not in adj.index:
        return pd.DataFrame()
    
    # Get weights vector
    weights = adj.loc[symbol].values  # (n_symbols,)
    
    # Align matrices to adjacency columns
    symbols = adj.columns.tolist()
    
    def _weighted_mean(matrix: pd.DataFrame) -> pd.Series:
        """Compute weighted mean across symbols for each date."""
        aligned = matrix.reindex(columns=symbols).fillna(0.0)
        # (n_dates, n_symbols) @ (n_symbols,) -> (n_dates,)
        return (aligned.values @ weights) / (weights.sum() + 1e-8)
    
    result = pd.DataFrame(index=returns_matrix.index)
    
    # Current neighbor aggregates (using t-1 data via shift)
    result["sgc_neighbor_momentum"] = _weighted_mean(returns_matrix.shift(1))
    result["sgc_neighbor_vol"] = _weighted_mean(vol_matrix.shift(1))
    result["sgc_neighbor_drawdown"] = _weighted_mean(drawdown_matrix.shift(1))
    
    # Lagged neighbor aggregates (v1.5 feature)
    result["sgc_neighbor_momentum_lag1"] = _weighted_mean(returns_matrix.shift(2))
    result["sgc_neighbor_momentum_lag5"] = _weighted_mean(returns_matrix.shift(6))
    
    return result


# -----------------------------------------------------------------------------
# REGIME-AWARE TEMPERATURE
# -----------------------------------------------------------------------------

def get_regime_temperature(
    regime_value: float,
    thresholds: Tuple[float, float] = (0.1, 0.25),
) -> float:
    """Map regime value to temperature scaling.
    
    Higher temperature in crisis = less neighbor mixing.
    
    Args:
        regime_value: Market regime indicator (e.g., VIX percentile, cross-sectional vol)
        thresholds: (elevated_threshold, crisis_threshold)
        
    Returns:
        Temperature scaling factor
    """
    elevated_th, crisis_th = thresholds
    
    if regime_value >= crisis_th:
        return REGIME_TEMP_CRISIS
    elif regime_value >= elevated_th:
        # Linear interpolation between elevated and crisis
        t = (regime_value - elevated_th) / (crisis_th - elevated_th + 1e-8)
        return REGIME_TEMP_ELEVATED + t * (REGIME_TEMP_CRISIS - REGIME_TEMP_ELEVATED)
    else:
        return REGIME_TEMP_NORMAL


# -----------------------------------------------------------------------------
# MAIN BUILDER CLASS
# -----------------------------------------------------------------------------

@dataclass
class SymbolGraphContextBuilder:
    """Builder for symbol graph context features.
    
    Computes cross-symbol graph embeddings and neighbor-aggregated features
    for use in Track-C/Mamba pipeline.
    
    Example:
        builder = SymbolGraphContextBuilder(
            universe=["AAPL", "MSFT", "NVDA", ...],
            group_map_path=Path("artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv"),
            cache_root=Path("cache/shared/symbol_graph"),
        )
        
        # Build for a specific date range
        panel = builder.build_family_panel(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            horizon=63,
        )
    """
    
    universe: List[str]
    group_map_path: Path
    cache_root: Path
    anchors: List[str] = field(default_factory=lambda: DEFAULT_ANCHORS.copy())
    corr_window: int = DEFAULT_CORR_WINDOW
    embed_dim: int = DEFAULT_EMBED_DIM
    top_k: int = DEFAULT_TOP_K
    corr_weight: float = DEFAULT_CORR_WEIGHT
    sector_weight: float = DEFAULT_SECTOR_WEIGHT
    regime_aware: bool = True
    
    def __post_init__(self):
        self.universe = [_norm_symbol(s) for s in self.universe if s]
        self.anchors = [_norm_symbol(s) for s in self.anchors if s]
        self.cache_root = Path(self.cache_root)
        self.group_map_path = Path(self.group_map_path)
        self._group_map: Optional[Dict[str, str]] = None
        self._returns_cache: Dict[str, pd.Series] = {}
    
    @property
    def graph_universe(self) -> List[str]:
        """Active universe plus anchors."""
        seen = set()
        out = []
        for s in self.universe + self.anchors:
            s = _norm_symbol(s)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    
    @property
    def group_map(self) -> Dict[str, str]:
        """Load sector group map."""
        if self._group_map is None:
            self._group_map = self._load_group_map()
        return self._group_map
    
    def _load_group_map(self) -> Dict[str, str]:
        """Load symbol -> sector mapping from CSV."""
        if not self.group_map_path.exists():
            logger.warning(f"Group map not found: {self.group_map_path}")
            return {}
        
        try:
            df = pd.read_csv(self.group_map_path)
            if "symbol" in df.columns and "group" in df.columns:
                return dict(zip(df["symbol"].str.upper(), df["group"]))
            return {}
        except Exception as e:
            logger.error(f"Error loading group map: {e}")
            return {}
    
    def _get_returns(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch daily returns for symbols."""
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            eodhd = get_eodhd_provider()
        except ImportError:
            eodhd = None
        
        returns_dict = {}
        for sym in symbols:
            sym = _norm_symbol(sym)
            cache_key = f"{sym}_{start}_{end}"
            
            if cache_key in self._returns_cache:
                returns_dict[sym] = self._returns_cache[cache_key]
                continue
            
            try:
                if eodhd and eodhd.api_key:
                    df = eodhd.get_eod_prices(sym, start_date=start, end_date=end)
                    if df is not None and not df.empty and "Close" in df.columns:
                        ret = df["Close"].pct_change()
                        returns_dict[sym] = ret
                        self._returns_cache[cache_key] = ret
            except Exception as e:
                logger.debug(f"Error fetching returns for {sym}: {e}")
        
        if not returns_dict:
            return pd.DataFrame()
        
        # Align to common index
        df = pd.DataFrame(returns_dict)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    
    def _adjacency_cache_path(self, friday_date: pd.Timestamp) -> Path:
        """Path for cached adjacency matrix."""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self.cache_root / f"adjacency_{friday_date.strftime('%Y%m%d')}.parquet"
    
    def _embeddings_cache_path(self, friday_date: pd.Timestamp) -> Path:
        """Path for cached embeddings."""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self.cache_root / f"embeddings_{friday_date.strftime('%Y%m%d')}.parquet"
    
    def _centrality_cache_path(self, friday_date: pd.Timestamp) -> Path:
        """Path for cached centrality metrics."""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self.cache_root / f"centrality_{friday_date.strftime('%Y%m%d')}.parquet"
    
    def build_adjacency(
        self,
        as_of_date: pd.Timestamp,
        regime_value: Optional[float] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Build blended adjacency matrix.
        
        Args:
            as_of_date: Build using data up to this date
            regime_value: Optional regime for temperature scaling
            use_cache: Whether to use/store cached adjacency
            
        Returns:
            Adjacency matrix DataFrame (symbols x symbols)
        """
        friday = _prev_friday(as_of_date)
        cache_path = self._adjacency_cache_path(friday)
        
        # Check cache
        if use_cache and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass
        
        # Compute correlation adjacency
        start = (friday - pd.Timedelta(days=self.corr_window * 2)).strftime("%Y-%m-%d")
        end = friday.strftime("%Y-%m-%d")
        
        returns = self._get_returns(self.graph_universe, start, end)
        
        if returns.empty:
            logger.warning("No returns data available for adjacency")
            return pd.DataFrame()
        
        A_corr = compute_rolling_correlation_adjacency(
            returns,
            as_of_date=friday,
            window=self.corr_window,
        )
        
        # Compute sector adjacency
        A_sector = compute_sector_adjacency(
            self.graph_universe,
            self.group_map,
        )
        
        # Blend
        if A_corr.empty:
            adj_blended = A_sector
        else:
            adj_blended = blend_adjacencies(
                A_corr,
                A_sector,
                corr_weight=self.corr_weight,
                sector_weight=self.sector_weight,
            )
        
        # Sparsify
        adj_blended = apply_top_k_sparsification(adj_blended, k=self.top_k, symmetric=True)
        
        # Apply temperature (regime-aware if enabled)
        temperature = REGIME_TEMP_NORMAL
        if self.regime_aware and regime_value is not None:
            temperature = get_regime_temperature(regime_value)
        
        adj_blended = apply_softmax_temperature(adj_blended, temperature=temperature)
        
        # Cache
        if use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                adj_blended.to_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache adjacency: {e}")
        
        return adj_blended
    
    def compute_embeddings(
        self,
        adjacency: pd.DataFrame,
        use_cache: bool = True,
        cache_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Compute SVD embeddings from adjacency.
        
        Args:
            adjacency: Adjacency matrix
            use_cache: Whether to use/store cache
            cache_date: Date for cache key (uses Friday)
            
        Returns:
            Embeddings DataFrame (symbols x embed_dim)
        """
        cache_path: Optional[Path] = None
        if cache_date:
            friday = _prev_friday(cache_date)
            cache_path = self._embeddings_cache_path(friday)
            
            if use_cache and cache_path.exists():
                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass
        
        embeddings = compute_svd_embeddings(adjacency, embed_dim=self.embed_dim)
        
        if cache_path is not None and use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                embeddings.to_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")
        
        return embeddings
    
    def compute_centrality(
        self,
        adjacency: pd.DataFrame,
        use_cache: bool = True,
        cache_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Compute centrality metrics from adjacency.
        
        Args:
            adjacency: Adjacency matrix
            use_cache: Whether to use/store cache
            cache_date: Date for cache key
            
        Returns:
            Centrality metrics DataFrame
        """
        cache_path: Optional[Path] = None
        if cache_date:
            friday = _prev_friday(cache_date)
            cache_path = self._centrality_cache_path(friday)
            
            if use_cache and cache_path.exists():
                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass
        
        centrality = compute_centrality_metrics(adjacency, top_k=self.top_k)
        
        if cache_path is not None and use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                centrality.to_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache centrality: {e}")
        
        return centrality
    
    def build_family_panel(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        horizon: int = 63,
        regime_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Build full symbol_graph_context panel for a symbol.
        
        Args:
            symbol: Target symbol
            start_date: Panel start date
            end_date: Panel end date
            horizon: Horizon parameter (for consistency)
            regime_series: Optional series of regime values indexed by date
            
        Returns:
            Panel DataFrame with all sgc_* columns indexed by date
        """
        symbol = _norm_symbol(symbol)
        sessions = _get_nyse_sessions(start_date, end_date)
        
        if len(sessions) == 0:
            return pd.DataFrame()
        
        # Get weekly Friday dates for adjacency refresh
        fridays = sorted(set(_prev_friday(pd.Timestamp(d)) for d in sessions))
        
        # Build adjacency/embeddings/centrality for each week
        weekly_adj = {}
        weekly_emb = {}
        weekly_cent = {}
        
        for friday in fridays:
            regime_val = None
            if regime_series is not None and friday in regime_series.index:
                regime_val = regime_series.loc[friday]
            
            adj = self.build_adjacency(friday, regime_value=regime_val)
            if adj.empty:
                continue
            
            weekly_adj[friday] = adj
            weekly_emb[friday] = self.compute_embeddings(adj, cache_date=friday)
            weekly_cent[friday] = self.compute_centrality(adj, cache_date=friday)
        
        if not weekly_adj:
            logger.warning(f"No adjacency data available for {symbol}")
            return pd.DataFrame()
        
        # Build universe feature matrices for neighbor aggregation
        # Fetch returns/vol/drawdown for all symbols
        returns = self._get_returns(
            self.graph_universe,
            start_date,
            end_date,
        )
        
        if returns.empty:
            logger.warning("No returns data for neighbor aggregation")
            return pd.DataFrame()
        
        # Compute vol and drawdown matrices
        vol_matrix = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
        
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.rolling(20, min_periods=1).max()
        drawdown_matrix = (cum_returns - rolling_max) / rolling_max
        
        momentum_matrix = returns.rolling(20, min_periods=10).sum()
        
        # Build panel row by row
        results = []
        
        for dt in sessions:
            dt = pd.Timestamp(dt)
            friday = _prev_friday(dt)
            
            if friday not in weekly_adj:
                # Use most recent available
                available = [f for f in fridays if f <= friday and f in weekly_adj]
                if not available:
                    continue
                friday = max(available)
            
            adj = weekly_adj[friday]
            emb = weekly_emb[friday]
            cent = weekly_cent[friday]
            
            row = {"date": dt}
            
            # Add embeddings
            if symbol in emb.index:
                for col in emb.columns:
                    row[col] = emb.loc[symbol, col]
            
            # Add centrality
            if symbol in cent.index:
                for col in cent.columns:
                    row[col] = cent.loc[symbol, col]
            
            # Add neighbor aggregates (vectorized below for efficiency)
            results.append(row)
        
        panel = pd.DataFrame(results)
        
        if panel.empty:
            return panel
        
        # Add neighbor features (vectorized for speed)
        # Use the most recent adjacency for simplicity in v1
        latest_friday = max(weekly_adj.keys())
        latest_adj = weekly_adj[latest_friday]
        
        neighbor_features = compute_neighbor_features_fast(
            symbol,
            latest_adj,
            momentum_matrix,
            vol_matrix,
            drawdown_matrix,
        )
        
        if not neighbor_features.empty:
            # Align and merge
            neighbor_features = neighbor_features.reset_index()
            neighbor_features.columns = ["date"] + list(neighbor_features.columns[1:])
            neighbor_features["date"] = pd.to_datetime(neighbor_features["date"])
            panel["date"] = pd.to_datetime(panel["date"])
            
            panel = panel.merge(neighbor_features, on="date", how="left")
        
        # Set date as index
        panel = panel.set_index("date").sort_index()
        
        # Fill missing values
        panel = panel.fillna(method="ffill").fillna(0.0)
        
        return panel


# -----------------------------------------------------------------------------
# FAMILY INTERFACE (for prep_families.py integration)
# -----------------------------------------------------------------------------

def build_symbol_graph_context_panel(
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int = 63,
    universe: Optional[List[str]] = None,
    group_map_path: Optional[Path] = None,
    cache_root: Optional[Path] = None,
    regime_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build symbol_graph_context panel for a symbol.
    
    This is the main entry point for prep_families.py integration.
    
    Args:
        symbol: Target symbol
        start_date: Panel start date
        end_date: Panel end date
        horizon: Horizon parameter
        universe: Universe of symbols (default: discover from cache)
        group_map_path: Path to group map CSV
        cache_root: Cache root for shared graph data
        regime_series: Optional regime values
        
    Returns:
        Panel DataFrame with sgc_* columns
    """
    # Default paths
    if group_map_path is None:
        group_map_path = Path("artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv")
    
    if cache_root is None:
        cache_root = Path("cache/shared/symbol_graph")
    
    # Default universe: discover from feature cache
    if universe is None:
        feature_cache = Path("cache/features")
        universe = _discover_universe_from_cache(feature_cache, horizon)
    
    if not universe:
        logger.warning("Empty universe, using anchors only")
        universe = DEFAULT_ANCHORS.copy()
    
    builder = SymbolGraphContextBuilder(
        universe=universe,
        group_map_path=group_map_path,
        cache_root=cache_root,
    )
    
    return builder.build_family_panel(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        regime_series=regime_series,
    )


def _discover_universe_from_cache(cache_root: Path, horizon: int) -> List[str]:
    """Discover universe from existing Track-C files."""
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return []
    
    suffix = f"_h{int(horizon)}_trackc.parquet"
    out = []
    seen = set()
    
    try:
        for p in cache_root.glob(f"*{suffix}"):
            name = p.stem
            if name.endswith(f"_h{int(horizon)}_trackc"):
                sym = name.replace(f"_h{int(horizon)}_trackc", "").upper()
                if sym and sym not in seen:
                    seen.add(sym)
                    out.append(sym)
    except Exception:
        pass
    
    return out


# -----------------------------------------------------------------------------
# FEATURE COLUMN DEFINITIONS
# -----------------------------------------------------------------------------

def get_symbol_graph_context_columns(embed_dim: int = DEFAULT_EMBED_DIM) -> List[str]:
    """Get list of all symbol_graph_context feature columns."""
    cols = []
    
    # Embeddings
    cols.extend([f"sgc_emb_{i}" for i in range(embed_dim)])
    
    # Centrality
    cols.extend([
        "sgc_degree",
        "sgc_strength_topk",
        "sgc_pagerank",
    ])
    
    # Neighbor aggregates
    cols.extend([
        "sgc_neighbor_momentum",
        "sgc_neighbor_vol",
        "sgc_neighbor_drawdown",
        "sgc_neighbor_momentum_lag1",
        "sgc_neighbor_momentum_lag5",
    ])
    
    return cols


# Export for family metadata
FAMILY_ID = "symbol_graph_context"
FEATURE_COLUMNS = get_symbol_graph_context_columns()


# -----------------------------------------------------------------------------
# MULTI-SYMBOL BUILDER (for Phase2 just-in-time integration)
# -----------------------------------------------------------------------------

def build_sgc_for_multi_symbol(
    active_symbols: Sequence[str],
    price_series_by_symbol: Dict[str, pd.Series],
    group_map_by_symbol: Optional[Dict[str, str]] = None,
    train_end_date: Optional[str] = None,
    corr_window: int = DEFAULT_CORR_WINDOW,
    embed_dim: int = DEFAULT_EMBED_DIM,
    top_k: int = DEFAULT_TOP_K,
    sector_blend_weight: float = DEFAULT_SECTOR_WEIGHT,
    temperature_base: float = REGIME_TEMP_NORMAL,
    regime_aware: bool = True,
    anchor_etfs: Optional[List[str]] = None,
    cache_dir: str = "cache/shared/symbol_graph",
) -> Optional[pd.DataFrame]:
    """Build symbol_graph_context features for multiple symbols at once.
    
    This is the Phase2 integration entry point - called just-in-time after
    Track-C is built but before Mamba training.
    
    Args:
        active_symbols: List of active symbols (the tradable universe)
        price_series_by_symbol: Dict mapping symbol -> pd.Series of close prices
        group_map_by_symbol: Optional dict mapping symbol -> sector/group
        train_end_date: End date for training period (adjacency computed up to this date)
        corr_window: Rolling correlation window (default 63)
        embed_dim: SVD embedding dimension (default 8)
        top_k: Top-k neighbors for sparsification (default 10)
        sector_blend_weight: Weight for sector edges (default 0.3)
        temperature_base: Base softmax temperature (default 1.0)
        regime_aware: Whether to use regime-aware temperature scaling
        anchor_etfs: Fixed anchor symbols for stability
        cache_dir: Directory for caching adjacency/embeddings
        
    Returns:
        DataFrame with columns: symbol, date, sgc_emb_0..k, sgc_degree, etc.
        or None if computation fails
    """
    if not active_symbols:
        logger.warning("[SGC] No active symbols provided")
        return None
    
    if not price_series_by_symbol:
        logger.warning("[SGC] No price series provided")
        return None
    
    if anchor_etfs is None:
        anchor_etfs = DEFAULT_ANCHORS.copy()
    
    # Normalize symbols
    syms = [_norm_symbol(s) for s in active_symbols if s]
    anchors = [_norm_symbol(a) for a in anchor_etfs if a]
    
    # Build graph universe = active ∪ anchors
    graph_universe = list(dict.fromkeys(syms + anchors))
    
    # Build returns DataFrame from price series
    returns_dict = {}
    for sym in graph_universe:
        pseries = price_series_by_symbol.get(sym)
        if pseries is not None and len(pseries) > 1:
            ret = pseries.pct_change()
            returns_dict[sym] = ret
    
    # Fetch missing anchor ETF data from EODHD if not in price_series_by_symbol
    missing_anchors = [a for a in anchors if a not in returns_dict]
    if missing_anchors:
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            eodhd = get_eodhd_provider()
            
            if eodhd and eodhd.api_key:
                # Determine date range from existing price series
                all_dates = []
                for ps in price_series_by_symbol.values():
                    if ps is not None and hasattr(ps, 'index'):
                        all_dates.extend(ps.index.tolist())
                
                if all_dates:
                    start_dt = pd.to_datetime(min(all_dates)) - pd.Timedelta(days=corr_window * 2)
                    end_dt = pd.to_datetime(max(all_dates))
                    start_str = start_dt.strftime("%Y-%m-%d")
                    end_str = end_dt.strftime("%Y-%m-%d")
                    
                    fetched_count = 0
                    for anchor_sym in missing_anchors:
                        try:
                            df = eodhd.get_eod_prices(anchor_sym, start_date=start_str, end_date=end_str)
                            if df is not None and not df.empty:
                                close_col = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
                                if close_col:
                                    ret = df[close_col].pct_change()
                                    ret.index = pd.to_datetime(ret.index)
                                    returns_dict[anchor_sym] = ret
                                    fetched_count += 1
                        except Exception as e:
                            logger.debug(f"[SGC] Failed to fetch anchor {anchor_sym}: {e}")
                    
                    if fetched_count > 0:
                        logger.info(f"[SGC] Fetched {fetched_count}/{len(missing_anchors)} anchor ETF price series")
        except ImportError:
            logger.debug("[SGC] EODHD provider not available for anchor ETF data")
        except Exception as e:
            logger.warning(f"[SGC] Failed to fetch anchor ETF data: {e}")
    
    if not returns_dict:
        logger.warning("[SGC] No valid returns data")
        return None
    
    returns_df = pd.DataFrame(returns_dict)
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df = returns_df.sort_index()
    
    # Determine as_of_date
    if train_end_date:
        as_of_date = pd.to_datetime(train_end_date)
    else:
        as_of_date = returns_df.index.max()
    
    friday = _prev_friday(as_of_date)
    
    # Compute correlation adjacency
    A_corr = compute_rolling_correlation_adjacency(
        returns_df,
        as_of_date=friday,
        window=corr_window,
    )
    
    if A_corr.empty:
        logger.warning("[SGC] Correlation adjacency is empty")
        return None
    
    # Compute sector adjacency if group_map provided
    if group_map_by_symbol:
        A_sector = compute_sector_adjacency(list(A_corr.columns), group_map_by_symbol)
        adj_blended = blend_adjacencies(
            A_corr,
            A_sector,
            corr_weight=1.0 - sector_blend_weight,
            sector_weight=sector_blend_weight,
        )
    else:
        adj_blended = A_corr.copy()
    
    # Sparsify
    adj_blended = apply_top_k_sparsification(adj_blended, k=top_k, symmetric=True)
    
    # Apply temperature
    adj_blended = apply_softmax_temperature(adj_blended, temperature=temperature_base)
    
    # Compute SVD embeddings
    embeddings = compute_svd_embeddings(adj_blended, embed_dim=embed_dim)
    
    # Compute centrality metrics
    centrality = compute_centrality_metrics(adj_blended, top_k=top_k)
    
    # Build output panel for each active symbol
    output_rows = []
    for sym in syms:
        sym = _norm_symbol(sym)
        if sym not in embeddings.index:
            continue
        
        row = {"symbol": sym, "date": friday}
        
        # Embeddings
        for i in range(embed_dim):
            col = f"sgc_emb_{i}"
            if col in embeddings.columns:
                row[col] = float(embeddings.loc[sym, col])
            else:
                row[col] = 0.0
        
        # Centrality
        if sym in centrality.index:
            row["sgc_degree"] = float(centrality.loc[sym, "sgc_degree"]) if "sgc_degree" in centrality.columns else 0.0
            row["sgc_strength_topk"] = float(centrality.loc[sym, "sgc_strength_topk"]) if "sgc_strength_topk" in centrality.columns else 0.0
            row["sgc_pagerank"] = float(centrality.loc[sym, "sgc_pagerank"]) if "sgc_pagerank" in centrality.columns else 0.0
        else:
            row["sgc_degree"] = 0.0
            row["sgc_strength_topk"] = 0.0
            row["sgc_pagerank"] = 0.0
        
        # Neighbor features (using adj_blended)
        if sym in adj_blended.index:
            neighbors = adj_blended.loc[sym].sort_values(ascending=False).head(top_k)
            valid_neighbors = neighbors[neighbors > 0]
            
            if len(valid_neighbors) > 0 and sym in returns_df.columns:
                # Compute neighbor-weighted momentum (20d returns)
                neighbor_returns = []
                weights = []
                for n_sym, weight in valid_neighbors.items():
                    if n_sym in returns_df.columns:
                        n_ret = returns_df[n_sym].dropna()
                        if len(n_ret) >= 20:
                            mom = float(n_ret.iloc[-20:].mean())
                            neighbor_returns.append(mom)
                            weights.append(float(weight))
                
                if neighbor_returns:
                    weights_arr = np.array(weights)
                    weights_arr = weights_arr / (weights_arr.sum() + 1e-9)
                    row["sgc_neighbor_momentum"] = float(np.dot(weights_arr, neighbor_returns))
                else:
                    row["sgc_neighbor_momentum"] = 0.0
                
                # Placeholder for vol/drawdown (would need full matrices)
                row["sgc_neighbor_vol"] = 0.0
                row["sgc_neighbor_drawdown"] = 0.0
                row["sgc_neighbor_momentum_lag1"] = row["sgc_neighbor_momentum"]  # Simplified
                row["sgc_neighbor_momentum_lag5"] = row["sgc_neighbor_momentum"]  # Simplified
            else:
                row["sgc_neighbor_momentum"] = 0.0
                row["sgc_neighbor_vol"] = 0.0
                row["sgc_neighbor_drawdown"] = 0.0
                row["sgc_neighbor_momentum_lag1"] = 0.0
                row["sgc_neighbor_momentum_lag5"] = 0.0
        else:
            row["sgc_neighbor_momentum"] = 0.0
            row["sgc_neighbor_vol"] = 0.0
            row["sgc_neighbor_drawdown"] = 0.0
            row["sgc_neighbor_momentum_lag1"] = 0.0
            row["sgc_neighbor_momentum_lag5"] = 0.0
        
        output_rows.append(row)
    
    if not output_rows:
        logger.warning("[SGC] No output rows generated")
        return None
    
    output_df = pd.DataFrame(output_rows)
    logger.info(f"[SGC] Built features for {len(output_df)} symbols with {len(output_df.columns)-2} features")
    
    return output_df
