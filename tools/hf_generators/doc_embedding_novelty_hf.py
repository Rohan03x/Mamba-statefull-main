"""
tools/hf_generators/doc_embedding_novelty_hf.py
================================================

Wrapper for DocumentEmbeddingNoveltyHF module that fetches GDELT global events,
embeds them with sentence-transformers, and computes novelty via rolling baseline.

Output Format:
- date: Trading date
- score: Novelty score (cosine distance from 30-day baseline, weighted by theme relevance)
- conf: Confidence (1.0 if data available, 0.0 otherwise)

Features Generated (13 columns):
1. doc_emb_novelty_hf_{h}_score - Overall novelty
2. doc_emb_novelty_hf_{h}_conf - Confidence
3. doc_emb_novelty_hf_{h}_n_events - Event count
4. doc_emb_novelty_hf_{h}_n_articles - Article count
5. doc_emb_novelty_hf_{h}_cluster_0_novelty - Technology cluster
6. doc_emb_novelty_hf_{h}_cluster_1_novelty - Finance cluster
7. doc_emb_novelty_hf_{h}_cluster_2_novelty - Geopolitics cluster
8. doc_emb_novelty_hf_{h}_cluster_3_novelty - Disaster cluster
9. doc_emb_novelty_hf_{h}_cluster_4_novelty - Social cluster
10. doc_emb_novelty_hf_{h}_cluster_5_novelty - Other cluster
11. doc_emb_novelty_hf_{h}_top_theme - Most relevant theme
12. doc_emb_novelty_hf_{h}_theme_weight - Theme weight
13. doc_emb_novelty_hf_{h}_baseline_mean - Baseline mean embedding norm

Data Source:
- GDELT 1.0: Parallel download from http://data.gdeltproject.org/events
- GDELT 2.0: BigQuery google-cloud-public-data.gdelt_v2.events
- Combined: ~5000 events per day, deduplicated
- Events filtered to major global activity (not symbol-specific)

Model:
- sentence-transformers/all-mpnet-base-v2 (768-dim embeddings)
- GPU-accelerated (cuda:0 if available, else CPU)
- Batch size: 32 events per batch

Novelty Computation:
- Rolling 30-day baseline (mean of historical embeddings)
- Cosine distance: 1 - cosine_similarity(current, baseline)
- Theme-based relevance weighting (technology/finance keywords)
- Per-cluster novelty scores (6 event types)

Author: Default Beta
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Fix import path - use src.dcf_lab instead of dcf_lab
try:
    from src.dcf_lab.modules.doc_embedding_novelty_hf import DocumentEmbeddingNoveltyHF
except ModuleNotFoundError:
    # Fallback for direct import if src is in sys.path
    from dcf_lab.modules.doc_embedding_novelty_hf import DocumentEmbeddingNoveltyHF

LOGGER = logging.getLogger(__name__)


def build(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    out_path: str,
    raw_source_cfg: Optional[Dict[str, Any]] = None,
    compute_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Build doc_embedding_novelty_hf signal using GDELT global events.
    
    SYMBOL-INDEPENDENT: GDELT data is global (not symbol-specific).
    If cache already exists for this horizon from another symbol, copy it.
    
    Args:
        symbol: Ticker symbol (e.g., "AAPL")
        horizon: Forecast horizon in days (e.g., 63)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        out_path: Path to write parquet file
        raw_source_cfg: Optional config for data sources (unused - GDELT is hardcoded)
        compute_cfg: Optional config dict with:
            - model_id: SentenceTransformer model name (default: all-mpnet-base-v2)
            - baseline_days: Rolling baseline window (default: 30)
            - batch_size: Embedding batch size (default: 32)
    
    Returns:
        None (writes parquet to out_path)
    """
    out_path_obj = Path(out_path)
    
    # OPTIMIZATION: GDELT is symbol-independent, check if cache exists from another symbol
    # Look for any existing doc_embedding_novelty_hf cache with same horizon
    cache_root = out_path_obj.parent.parent  # e.g., data/local_cache
    if cache_root.exists():
        # Search pattern: */block_hf/*/horizon/doc_embedding_novelty_hf.parquet
        search_pattern = f"*/block_hf/*/{horizon}/doc_embedding_novelty_hf.parquet"
        existing_caches = list(cache_root.glob(search_pattern))
        if existing_caches:
            source_cache = existing_caches[0]
            LOGGER.info(
                f"✅ SYMBOL-INDEPENDENT: Copying GDELT cache from {source_cache.parent.parent.name} → {symbol}"
            )
            out_path_obj.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source_cache, out_path_obj)
            return
    
    LOGGER.info(
        f"Building doc_embedding_novelty_hf: {symbol} h{horizon} [{start}, {end}) [GDELT global events]"
    )
    
    # Initialize module with config
    compute_cfg = compute_cfg or {}
    model_id = compute_cfg.get("model_id", "sentence-transformers/all-mpnet-base-v2")
    baseline_days = compute_cfg.get("baseline_days", 30)
    batch_size = compute_cfg.get("batch_size", 32)
    
    module = DocumentEmbeddingNoveltyHF(
        model_id=model_id,
        baseline_days=baseline_days,
        batch_size=batch_size,
    )
    
    # Generate signal
    signal = module.emit_signal(
        symbol=symbol,
        horizon=horizon,
        start_date=start,
        end_date=end
    )
    
    # Extract DataFrame
    signal_df = signal.df
    
    if signal_df.empty:
        LOGGER.warning(
            f"No doc embedding novelty data for {symbol} in [{start}, {end})"
        )
        # Create empty signal with proper date range
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        signal_df = pd.DataFrame({
            'date': date_range,
            'score': [0.0] * len(date_range),
            'conf': [0.0] * len(date_range)
        })
    else:
        # Reset index to have date as column
        signal_df = signal_df.reset_index()
        # Handle both named and unnamed indices
        if 'index' in signal_df.columns:
            signal_df = signal_df.rename(columns={'index': 'date'})
        elif 'date' not in signal_df.columns:
            # If index had a different name, rename first column to 'date'
            index_col = signal_df.columns[0]
            signal_df = signal_df.rename(columns={index_col: 'date'})
    
    # Validate output format (must have date, score, conf columns)
    required_cols = {'date', 'score', 'conf'}
    if not required_cols.issubset(signal_df.columns):
        LOGGER.error(f"Missing required columns. Have: {signal_df.columns.tolist()}, need: {required_cols}")
        raise ValueError(f"Signal missing required columns: {required_cols - set(signal_df.columns)}")
    
    # Write to parquet
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(out_path, index=False)
    
    LOGGER.info(
        f"✅ Wrote doc_embedding_novelty_hf signal: {len(signal_df)} bars → {out_path}"
    )
