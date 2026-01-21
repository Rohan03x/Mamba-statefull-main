"""
Document Embedding Novelty Module (HF)

Detects global macro/geopolitical/regulatory regime changes using GDELT events:
- Pulls GDELT GKG (themes, entities, tone, locations)
- Pulls GDELT Event 2.0 (event codes, actors, Goldstein scale)
- Embeds events with sentence-transformers/all-mpnet-base-v2
- Computes rolling semantic baseline (30-day mean)
- Generates novelty scores: global, symbol-specific, and cluster-specific

Output: 15 features per symbol tracking regime shifts before prices reflect them
- score, conf: Overall novelty and confidence
- n_events, n_articles: Event/article counts
- 6 cluster novelties: macro, geopolitical, regulatory, energy, conflict, tech
- top_theme, theme_weight, baseline_mean: Metadata
- novelty_spike_flag: Binary flag for >2σ novelty spikes
- novelty_persistence_5d: Fraction of last 5 days with high novelty

Author: DCF Lab Team
Updated: 2025-11-21
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..signal_bus import ModuleSignal
from ..gdelt_global_fetcher import GDELTGlobalFetcher, EVENT_CLUSTERS

# Import sentence_transformers (should work in .venv)
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    _ST_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

_DATA_CACHE = Path(__file__).resolve().parents[3] / "data_cache"

# Symbol profile keywords for relevance scoring
SYMBOL_PROFILES = {
    'AAPL': {
        'themes': ['TECH', 'CONSUMER', 'MOBILE', 'INNOVATION', 'RETAIL'],
        'countries': ['US', 'CHN', 'JPN', 'EUR'],
        'sector': 'technology'
    },
    'MSFT': {
        'themes': ['TECH', 'CLOUD', 'ENTERPRISE', 'SOFTWARE', 'AI'],
        'countries': ['US', 'EUR', 'CHN'],
        'sector': 'technology'
    },
    'GOOGL': {
        'themes': ['TECH', 'ADVERTISING', 'AI', 'CLOUD', 'MOBILE'],
        'countries': ['US', 'EUR', 'CHN'],
        'sector': 'technology'
    },
    'TSLA': {
        'themes': ['AUTO', 'ENERGY', 'TECH', 'BATTERY', 'RENEWABLE'],
        'countries': ['US', 'CHN', 'EUR', 'DEU'],
        'sector': 'automotive'
    },
    'NVDA': {
        'themes': ['TECH', 'AI', 'CHIP', 'GAMING', 'DATACENTER'],
        'countries': ['US', 'TWN', 'CHN'],
        'sector': 'technology'
    },
}


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors (1 - cosine_similarity)"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    similarity = float(np.dot(a, b) / (na * nb))
    return 1.0 - similarity


@dataclass
class EventEmbedding:
    """Container for event embedding and metadata"""
    date: pd.Timestamp
    embedding: np.ndarray
    event_text: str
    clusters: Dict[str, bool]
    tone: float
    goldstein: float
    num_articles: int


class DocumentEmbeddingNoveltyHF:
    """Detect global regime changes via GDELT event embeddings"""

    NAME = "doc_embedding_novelty_hf"
    metadata = {
        "group": "hf",
        "group_cap": 0.5,
        "group_penalty": 1.0,
        "w_min": 0.0,
        "w_max": 0.4,
    }

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        baseline_days: int = 90,
        novelty_threshold: float = 0.15,
        batch_size: int = 32,
        max_events_per_period: int = 1500,  # Sample events stratified across clusters for coverage
        fetcher_workers: int = 40,
        **kwargs
    ):
        """
        Initialize DocumentEmbeddingNoveltyHF module
        
        Args:
            model_name: HuggingFace model for embeddings
            baseline_days: Days BEFORE training period for fixed baseline (60-90 recommended)
            novelty_threshold: Threshold for confidence (novelty > threshold → conf=1)
            batch_size: Batch size for encoding (default 32)
            max_events_per_period: Max GDELT events to sample (stratified across clusters)
        """
        self.model_name = model_name
        self.baseline_days = baseline_days
        self.novelty_threshold = novelty_threshold
        self.batch_size = batch_size
        self.max_events_per_period = max_events_per_period
        self.fetcher_workers = max(1, int(fetcher_workers or 1))
        self._model = None
        
        # Initialize GDELT fetcher with merged cache (parallel workers configurable)
        try:
            from src.dcf_lab.gdelt_global_fetcher import GDELTGlobalFetcher
            # Use centralized cache paths
            try:
                from src.cache_paths import gdelt_global_merged_dir
                gdelt_cache = str(gdelt_global_merged_dir())
            except ImportError:
                gdelt_cache = "data/cache/gdelt_global/merged"
            self.gdelt_fetcher = GDELTGlobalFetcher(
                cache_dir=gdelt_cache,
                max_workers=self.fetcher_workers
            )
            LOGGER.info(
                "✅ GDELT Global Fetcher initialized with merged cache (max %d events/period, %d workers)",
                max_events_per_period,
                self.fetcher_workers,
            )
        except Exception as exc:
            LOGGER.warning(f"GDELT fetcher unavailable: {exc}")
            self.gdelt_fetcher = None

    def _resolve_model(self) -> Optional[SentenceTransformer]:
        """Lazy load SentenceTransformer model"""
        if self._model is not None:
            return self._model
        
        try:
            from sentence_transformers import SentenceTransformer
            # Some environments can end up with torch default device set to 'meta'
            # (lazy-init), which breaks SentenceTransformer weight materialization.
            try:
                import torch  # type: ignore
                if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
                    if str(torch.get_default_device()) == "meta":
                        torch.set_default_device("cpu")
            except Exception:
                pass

            device = "cpu"
            try:
                import torch  # type: ignore
                if hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
                    device = "cuda"
            except Exception:
                device = "cpu"

            LOGGER.info(f"Loading SentenceTransformer: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=device)
            LOGGER.info(f"✅ Loaded SentenceTransformer: {self.model_name}")
            return self._model
        except Exception as exc:
            LOGGER.error(f"Failed to load SentenceTransformer {self.model_name}: {exc}")
            return None

    def _encode_events(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode event texts to embeddings"""
        model = self._resolve_model()
        if model is None or not texts:
            return None
        
        try:
            LOGGER.info(f"Encoding {len(texts)} texts with batch_size={self.batch_size}")
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            LOGGER.info(f"✅ Encoded {len(embeddings)} embeddings, shape: {embeddings.shape}")
            return embeddings.astype(np.float32)
        except Exception as exc:
            LOGGER.error(f"❌ Encoding failed: {exc}")
            import traceback
            LOGGER.error(traceback.format_exc())
            return None

    def _load_global_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load global GDELT events"""
        # Extend range for baseline computation
        start = pd.Timestamp(start_date) - pd.Timedelta(days=self.baseline_days)
        end = pd.Timestamp(end_date)
        
        df = self.gdelt_fetcher.fetch(
            start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d')
        )
        
        return df

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert value to float"""
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default: int = 1) -> int:
        """Safely convert value to int"""
        if pd.isna(value):
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _get_primary_cluster(self, clusters: Dict[str, bool]) -> str:
        """Get primary cluster from cluster dict (first True value or 'other')"""
        for cluster_name, is_match in clusters.items():
            if is_match:
                return cluster_name
        return 'other'  # Fallback for uncategorized events

    def _build_event_embeddings(self, events_df: pd.DataFrame) -> List[EventEmbedding]:
        """Build embeddings with month-by-month stratified sampling across clusters"""
        if events_df.empty:
            return []
        
        original_count = len(events_df)
        
        # Ensure date column is datetime
        if 'date' in events_df.columns:
            events_df['date'] = pd.to_datetime(events_df['date'])
            events_df['year_month'] = events_df['date'].dt.to_period('M')
        else:
            LOGGER.warning("No date column found, using simple sampling")
            target_sample = min(self.max_events_per_period, original_count)
            if original_count > target_sample:
                events_df = events_df.sample(n=target_sample, random_state=42)
            LOGGER.info(f"📊 Sampled {len(events_df):,} events from {original_count:,} total")
            events_df['_cluster'] = 'other'
        
        if 'year_month' in events_df.columns:
            # Month-by-month stratified sampling
            LOGGER.info(f"📊 Month-by-month stratified sampling from {original_count:,} total events")
            
            # Classify events into clusters
            events_df['_cluster'] = events_df.apply(
                lambda row: self._get_primary_cluster(self.gdelt_fetcher.classify_event_cluster(row)),
                axis=1
            )
            
            # Sample per month: ~350 events per cluster minimum for better coverage
            min_per_cluster_per_month = 350
            months = events_df['year_month'].unique()
            LOGGER.info(f"   Processing {len(months)} months")
            
            sampled_dfs = []
            total_sampled = 0
            
            for month in sorted(months):
                month_df = events_df[events_df['year_month'] == month]
                month_count = len(month_df)
                
                # Get cluster distribution for this month
                cluster_counts = month_df['_cluster'].value_counts()
                clusters_with_data = [c for c in EVENT_CLUSTERS.keys() if c in cluster_counts.index]
                
                if not clusters_with_data:
                    # No recognized clusters, sample proportionally
                    month_sample = min(1500, month_count)
                    sampled_month = month_df.sample(n=month_sample, random_state=42)
                    sampled_dfs.append(sampled_month)
                    total_sampled += month_sample
                    continue
                
                # Stratified sampling within month
                month_sampled = []
                for cluster in clusters_with_data:
                    cluster_df = month_df[month_df['_cluster'] == cluster]
                    available = len(cluster_df)
                    
                    # Target: min_per_cluster_per_month or all available if less
                    target = min(min_per_cluster_per_month, available)
                    
                    sampled = cluster_df.sample(n=target, random_state=42)
                    month_sampled.append(sampled)
                
                if month_sampled:
                    month_combined = pd.concat(month_sampled, ignore_index=True)
                    sampled_dfs.append(month_combined)
                    total_sampled += len(month_combined)
            
            if sampled_dfs:
                events_df = pd.concat(sampled_dfs, ignore_index=True)
                
                # Log final distribution
                final_cluster_counts = events_df['_cluster'].value_counts()
                LOGGER.info(f"   ✅ Sampled {len(events_df):,} events across {len(months)} months")
                LOGGER.info(f"   Cluster distribution:")
                for cluster in sorted(final_cluster_counts.index):
                    count = final_cluster_counts[cluster]
                    pct = count / len(events_df) * 100
                    LOGGER.info(f"     {cluster:15s}: {count:5d} ({pct:5.1f}%)")
            else:
                LOGGER.warning("No events sampled!")
        else:
            # Fallback: classify and log
            events_df['_cluster'] = events_df.apply(
                lambda row: self._get_primary_cluster(self.gdelt_fetcher.classify_event_cluster(row)),
                axis=1
            )
            cluster_counts = events_df['_cluster'].value_counts()
            LOGGER.info(f"📊 Using all {original_count:,} events")
            LOGGER.info(f"   Cluster distribution: {dict(cluster_counts)}")
        
        LOGGER.info(f"Building embeddings for {len(events_df):,} events...")
        
        # Build event texts
        event_texts = []
        event_metadata = []
        skipped = 0
        
        for idx, row in events_df.iterrows():
            try:
                text = self.gdelt_fetcher.build_event_text(row)
                
                # Validate event text quality
                if len(text) < 20:  # Too short
                    skipped += 1
                    continue
                
                # Check if text looks corrupted (too many numbers)
                words = text.split()
                numeric_words = sum(1 for w in words if w.replace('.','').replace('-','').isdigit())
                if len(words) > 0 and numeric_words / len(words) > 0.5:  # >50% numbers = likely corrupt
                    skipped += 1
                    continue
                
                clusters = self.gdelt_fetcher.classify_event_cluster(row)
                
                # Safely extract metadata with validation
                event_metadata.append({
                    'date': row['date'],
                    'clusters': clusters,
                    'tone': self._safe_float(row.get('AvgTone'), 0.0),
                    'goldstein': self._safe_float(row.get('GoldsteinScale'), 0.0),
                    'num_articles': self._safe_int(row.get('NumArticles'), 1),
                    'text': text
                })
                event_texts.append(text)
            except Exception as e:
                skipped += 1
                continue
        
        if skipped > 0:
            LOGGER.warning(f"⚠️ Skipped {skipped}/{original_count} events (invalid/corrupted data)")
        
        if not event_texts:
            LOGGER.warning(f"⚠️ No valid event texts after filtering {original_count} events")
            return []
        
        # Log data quality
        text_lengths = [len(t) for t in event_texts]
        LOGGER.info(f"📊 Event text quality: {len(event_texts)} events, "
                   f"lengths {min(text_lengths)}-{max(text_lengths)} chars "
                   f"(mean: {sum(text_lengths)/len(text_lengths):.0f})")
        
        # Encode
        LOGGER.info(f"Encoding {len(event_texts)} event texts with {self.model_name}...")
        embeddings = self._encode_events(event_texts)
        if embeddings is None:
            LOGGER.warning("Encoding failed")
            return []
        
        LOGGER.info(f"✅ Generated {len(embeddings)} embeddings")
        
        # Build EventEmbedding objects
        result = []
        for emb, meta in zip(embeddings, event_metadata):
            result.append(EventEmbedding(
                date=meta['date'],
                embedding=emb,
                event_text=meta['text'],
                clusters=meta['clusters'],
                tone=meta['tone'],
                goldstein=meta['goldstein'],
                num_articles=meta['num_articles']
            ))
        
        return result

    def _compute_fixed_baseline(
        self, 
        event_embeddings: List[EventEmbedding],
        baseline_start: pd.Timestamp,
        baseline_end: pd.Timestamp
    ) -> Dict[str, np.ndarray]:
        """
        Compute FIXED baseline from pre-training period (60-90 days BEFORE training)
        
        Returns baseline per cluster:
        - 'global': Mean of all events in baseline period
        - 'macro': Mean of macro cluster events
        - 'geopolitical': Mean of geopolitical events
        - etc.
        """
        # Filter to baseline period only
        baseline_events = [
            ev for ev in event_embeddings 
            if baseline_start <= pd.Timestamp(ev.date).normalize() <= baseline_end
        ]
        
        # ADAPTIVE: If no baseline events, use earliest available events
        if not baseline_events and event_embeddings:
            LOGGER.warning(f"No baseline events found in {baseline_start.date()} to {baseline_end.date()}")
            LOGGER.info(f"📊 Using earliest {min(100, len(event_embeddings))} events as adaptive baseline")
            baseline_events = sorted(event_embeddings, key=lambda ev: ev.date)[:min(100, len(event_embeddings))]
        
        if not baseline_events:
            LOGGER.warning(f"No events available for baseline - using zero vector")
            # Get embedding dimension from any event, or default to 768
            dim = len(event_embeddings[0].embedding) if event_embeddings else 768
            zero_vec = np.zeros(dim)
            return {
                'global': zero_vec,
                **{cluster: zero_vec for cluster in EVENT_CLUSTERS.keys()}
            }
        
        LOGGER.info(f"📊 Computing fixed baseline from {len(baseline_events)} events ({baseline_start.date()} to {baseline_end.date()})")
        
        # Global baseline (all events)
        global_embeddings = [ev.embedding for ev in baseline_events]
        global_baseline = np.mean(global_embeddings, axis=0)
        
        # Per-cluster baselines
        cluster_baselines = {}
        for cluster_name in EVENT_CLUSTERS.keys():
            cluster_events = [ev for ev in baseline_events if ev.clusters.get(cluster_name, False)]
            if cluster_events:
                cluster_embeddings = [ev.embedding for ev in cluster_events]
                cluster_baselines[cluster_name] = np.mean(cluster_embeddings, axis=0)
                LOGGER.debug(f"  {cluster_name}: {len(cluster_events)} events")
            else:
                # Fall back to global baseline if no events in this cluster
                cluster_baselines[cluster_name] = global_baseline
                LOGGER.debug(f"  {cluster_name}: 0 events (using global baseline)")
        
        return {
            'global': global_baseline,
            **cluster_baselines
        }

    def _compute_novelty_scores(
        self,
        event_embeddings: List[EventEmbedding],
        baselines: Dict[str, np.ndarray],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Compute global and per-cluster novelty scores
        
        Args:
            event_embeddings: All event embeddings (including baseline period)
            baselines: Fixed baselines per cluster
            start_date: Start of scoring period (after baseline)
            end_date: End of scoring period
        """
        rows = []
        
        # Group events by date
        events_by_date = {}
        for ev in event_embeddings:
            date = pd.Timestamp(ev.date).normalize()
            if date not in events_by_date:
                events_by_date[date] = []
            events_by_date[date].append(ev)
        
        # Compute novelty for each day in scoring period
        for date, events in sorted(events_by_date.items()):
            # Only score dates within requested range
            if date < start_date or date > end_date:
                continue
            
            # Global novelty (weighted by num_articles)
            novelties = []
            weights = []
            for ev in events:
                novelty = _cosine_distance(ev.embedding, baselines['global'])
                novelties.append(novelty)
                weights.append(ev.num_articles)
            
            weights_arr = np.array(weights)
            weights_norm = weights_arr / (weights_arr.sum() + 1e-9)
            global_novelty = np.sum(np.array(novelties) * weights_norm)
            
            # Per-cluster novelties
            cluster_novelties = {}
            for cluster_name in EVENT_CLUSTERS.keys():
                cluster_events = [ev for ev in events if ev.clusters.get(cluster_name, False)]
                if cluster_events:
                    cluster_novelty_vals = [
                        _cosine_distance(ev.embedding, baselines[cluster_name]) 
                        for ev in cluster_events
                    ]
                    cluster_novelties[cluster_name] = np.mean(cluster_novelty_vals)
                else:
                    cluster_novelties[cluster_name] = 0.0  # No events → no novelty
            
            rows.append({
                'date': date,
                'global_novelty': global_novelty,
                'num_events': len(events),
                **{f'{cluster}_novelty': cluster_novelties[cluster] for cluster in EVENT_CLUSTERS.keys()}
            })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df = df.set_index('date').sort_index()
        
        # Compute multi-day smoothed averages (helps reduce daily noise)
        df['global_novelty_1d'] = df['global_novelty']
        df['global_novelty_7d'] = df['global_novelty'].rolling(7, min_periods=1).mean()
        df['global_novelty_30d'] = df['global_novelty'].rolling(30, min_periods=1).mean()
        
        return df

    def _compute_symbol_relevance(
        self,
        symbol: str,
        event_embeddings: List[EventEmbedding]
    ) -> Dict[pd.Timestamp, float]:
        """Compute symbol relevance scores for events"""
        profile = SYMBOL_PROFILES.get(symbol.upper().replace('.US', ''), {
            'themes': [],
            'countries': ['US'],
            'sector': 'general'
        })
        
        relevance_by_date = {}
        
        # Group events by date
        events_by_date = {}
        for ev in event_embeddings:
            date = pd.Timestamp(ev.date).normalize()
            if date not in events_by_date:
                events_by_date[date] = []
            events_by_date[date].append(ev)
        
        # Compute relevance for each day
        for date, events in events_by_date.items():
            relevances = []
            
            for ev in events:
                # Theme match (0.4 weight)
                theme_match = 0.0
                event_text_upper = ev.event_text.upper()
                for theme in profile['themes']:
                    if theme in event_text_upper:
                        theme_match = 1.0
                        break
                
                # Country match (0.3 weight) - simplified, would need GKG location parsing
                country_match = 0.5  # Default
                
                # Sector match (0.2 weight) - based on cluster membership
                sector_match = 0.0
                if profile['sector'] == 'technology' and ev.clusters.get('tech', False):
                    sector_match = 1.0
                elif profile['sector'] == 'automotive' and ev.clusters.get('energy', False):
                    sector_match = 1.0
                
                # Semantic similarity (0.1 weight) - would need symbol profile embedding
                semantic_sim = 0.0
                
                relevance = (
                    0.4 * theme_match +
                    0.3 * country_match +
                    0.2 * sector_match +
                    0.1 * semantic_sim
                )
                relevances.append(relevance)
            
            # Average relevance for the day
            relevance_by_date[date] = np.mean(relevances) if relevances else 0.0
        
        return relevance_by_date

    def emit_signal(
        self,
        *,
        symbol: str,
        horizon: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> ModuleSignal:
        """
        Generate document embedding novelty signal
        
        Returns 13 features:
        - 3 global novelty (1d, 7d, 30d)
        - 1 symbol relevance
        - 3 symbol-weighted novelty (1d, 7d, 30d)
        - 6 cluster novelties (macro, geopolitical, regulatory, energy, conflict, tech)
        """
        if not start_date or not end_date:
            LOGGER.error("start_date and end_date are required")
            return ModuleSignal(name=self.NAME, horizon=horizon, df=pd.DataFrame(), symbol=symbol)
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Define baseline period: baseline_days BEFORE training start
        baseline_end = start - pd.Timedelta(days=1)
        baseline_start = baseline_end - pd.Timedelta(days=self.baseline_days - 1)
        
        LOGGER.info(f"📊 Baseline period: {baseline_start.date()} to {baseline_end.date()} ({self.baseline_days} days)")
        LOGGER.info(f"📈 Scoring period: {start.date()} to {end.date()}")
        
        # Load global events (including baseline period)
        events_df = self._load_global_events(
            baseline_start.strftime('%Y-%m-%d'),
            end_date
        )
        
        if events_df.empty:
            LOGGER.warning(f"⚠️  No GDELT events found for baseline+scoring period ({baseline_start.date()} to {end.date()})")
            LOGGER.warning(f"    This may be due to the 2014 gap (2014-01-01 to 2015-02-17).")
            LOGGER.warning(f"    Returning empty signal with null values.")
            return ModuleSignal(name=self.NAME, horizon=horizon, df=pd.DataFrame(), symbol=symbol)
        
        # Build embeddings for all events
        event_embeddings = self._build_event_embeddings(events_df)
        
        if not event_embeddings:
            LOGGER.warning("No event embeddings generated")
            return ModuleSignal(name=self.NAME, horizon=horizon, df=pd.DataFrame(), symbol=symbol)
        
        # Compute FIXED baseline from pre-training period
        baselines = self._compute_fixed_baseline(event_embeddings, baseline_start, baseline_end)
        
        # Compute novelty scores for scoring period only
        novelty_df = self._compute_novelty_scores(event_embeddings, baselines, start, end)
        
        if novelty_df.empty:
            LOGGER.warning(f"⚠️  No novelty scores computed for {start.date()} to {end.date()}")
            LOGGER.warning(f"    Creating null-filled DataFrame for date range.")
            
            # Create null-filled DataFrame spanning the requested date range
            null_df = pd.DataFrame({
                'date': pd.date_range(start, end, freq='D'),
                'score': 0.0,
                'conf': 0.0,
                'n_events': 0,
                'n_articles': 0,
                'macro_novelty': 0.0,
                'geopolitical_novelty': 0.0,
                'regulatory_novelty': 0.0,
                'energy_novelty': 0.0,
                'conflict_novelty': 0.0,
                'tech_novelty': 0.0,
                'top_theme_numeric': 0.0,
                'theme_weight': 0.0,
                'baseline_mean': 0.0,
                'novelty_spike_flag': 0.0,
                'novelty_persistence_5d': 0.0,
            }).set_index('date')
            
            LOGGER.info(f"✅ Generated null-filled signal for {len(null_df)} days (no GDELT data available)")
            return ModuleSignal(name=self.NAME, horizon=horizon, df=null_df, symbol=symbol)
        
        # Build comprehensive result DataFrame with all 15 features
        result_rows = []
        
        # Compute statistics for spike detection
        novelty_values = novelty_df['global_novelty_1d'].values
        novelty_mean = float(np.mean(novelty_values))
        novelty_std = float(np.std(novelty_values))
        spike_threshold = novelty_mean + 2.0 * novelty_std  # 2σ threshold
        
        for date, row in novelty_df.iterrows():
            novelty_1d = row['global_novelty_1d']
            conf = 1.0 if novelty_1d > self.novelty_threshold else 0.0
            
            # Event/article counts
            n_events = int(row.get('num_events', 0))
            n_articles = n_events  # Approximation - GDELT doesn't always have separate article counts
            
            # 6 cluster novelties
            macro_novelty = row.get('macro_novelty', 0.0)
            geopolitical_novelty = row.get('geopolitical_novelty', 0.0)
            regulatory_novelty = row.get('regulatory_novelty', 0.0)
            energy_novelty = row.get('energy_novelty', 0.0)
            conflict_novelty = row.get('conflict_novelty', 0.0)
            tech_novelty = row.get('tech_novelty', 0.0)
            
            # Top theme: Cluster with highest novelty
            cluster_scores = {
                'macro': macro_novelty,
                'geopolitical': geopolitical_novelty,
                'regulatory': regulatory_novelty,
                'energy': energy_novelty,
                'conflict': conflict_novelty,
                'tech': tech_novelty,
            }
            top_theme = max(cluster_scores.items(), key=lambda x: x[1])[0] if any(cluster_scores.values()) else 'none'
            theme_weight = max(cluster_scores.values()) if cluster_scores else 0.0
            
            # Baseline mean embedding norm (approximation)
            baseline_mean = float(baselines.get('global', np.zeros(1))[0] if isinstance(baselines.get('global'), np.ndarray) else 0.0)
            
            # Spike flag: 1 if novelty > mean + 2σ
            novelty_spike_flag = 1.0 if novelty_1d > spike_threshold else 0.0
            
            result_rows.append({
                'date': date,
                'score': novelty_1d,
                'conf': conf,
                'n_events': n_events,
                'n_articles': n_articles,
                'macro_novelty': macro_novelty,
                'geopolitical_novelty': geopolitical_novelty,
                'regulatory_novelty': regulatory_novelty,
                'energy_novelty': energy_novelty,
                'conflict_novelty': conflict_novelty,
                'tech_novelty': tech_novelty,
                'top_theme': top_theme,
                'theme_weight': theme_weight,
                'baseline_mean': baseline_mean,
                'novelty_spike_flag': novelty_spike_flag,
            })
        
        result_df = pd.DataFrame(result_rows).set_index('date')
        
        # Compute novelty_persistence_5d: fraction of last 5 days with high novelty
        # High novelty = score > threshold
        high_novelty_mask = (result_df['score'] > self.novelty_threshold).astype(float)
        result_df['novelty_persistence_5d'] = high_novelty_mask.rolling(
            window=5, 
            min_periods=1
        ).mean()
        
        # Convert top_theme to numeric for compatibility (use theme_weight as proxy)
        # Keep original for debugging, but aggregator will use numeric features
        result_df['top_theme_numeric'] = result_df['theme_weight']
        
        # Forward-fill to cover all days in range
        if not result_df.empty:
            full_range = pd.date_range(start, end, freq='D')
            result_df = result_df.reindex(full_range)
            
            # Forward-fill numeric columns
            numeric_cols = [
                'score', 'conf', 'n_events', 'n_articles',
                'macro_novelty', 'geopolitical_novelty', 'regulatory_novelty',
                'energy_novelty', 'conflict_novelty', 'tech_novelty',
                'theme_weight', 'baseline_mean', 'novelty_spike_flag',
                'novelty_persistence_5d', 'top_theme_numeric'
            ]
            for col in numeric_cols:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(method='ffill').fillna(0.0)
            
            # Forward-fill categorical
            if 'top_theme' in result_df.columns:
                result_df['top_theme'] = result_df['top_theme'].fillna(method='ffill').fillna('none')
        
        # Drop non-numeric top_theme for aggregator compatibility
        if 'top_theme' in result_df.columns:
            result_df = result_df.drop(columns=['top_theme'])
        
        LOGGER.info(f"✅ Generated {len(result_df)} days of doc embedding novelty features for {symbol}")
        LOGGER.info(f"  Mean novelty: {result_df['score'].mean():.4f} ± {result_df['score'].std():.4f}")
        LOGGER.info(f"  Spike threshold (mean + 2σ): {spike_threshold:.4f}")
        LOGGER.info(f"  Spike days: {int(result_df['novelty_spike_flag'].sum())}/{len(result_df)}")
        LOGGER.info(f"  High-persistence days (>0.8): {(result_df['novelty_persistence_5d'] > 0.8).sum()}/{len(result_df)}")
        
        return ModuleSignal(name=self.NAME, horizon=horizon, df=result_df, symbol=symbol)


__all__ = ["DocumentEmbeddingNoveltyHF"]
