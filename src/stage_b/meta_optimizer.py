"""Meta-Optimizer: Self-Learning Layer for Optuna Hyperparameter Optimization.

This module implements a meta-learning layer on top of Optuna:

ARCHITECTURE:
┌────────────────────────────┐
│     Optuna Trial Runner    │
│   (runs model each trial)  │
└──────────────┬─────────────┘
               │ results
┌──────────────▼─────────────┐
│        META-ANALYZER       │
│  - Memory                  │
│  - Attribution             │
│  - Interaction learning    │
│  - Elite trial selection   │
└──────────────┬─────────────┘
               │ adjustments
┌──────────────▼─────────────┐
│     SEARCH-SPACE UPDATER   │
│ modifies parameter ranges  │
└──────────────┬─────────────┘
               │ new priors
┌──────────────▼─────────────┐
│      Next Optuna trial     │
└─────────────────────────────┘

COMPONENTS:
1. TrialMemory - Stores trial history and elite trials
2. AttributionEngine - Computes parameter importance via ΔSharpe/ΔStability correlation
3. InteractionLearner - Learns non-linear param relationships via LightGBM
4. SearchSpaceUpdater - Modifies Optuna priors based on learnings
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Deque, Union
import time

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

LOGGER = logging.getLogger("stage_b.meta_optimizer")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrialRecord:
    """Record of a single trial's results."""
    
    trial_id: int
    params: Dict[str, Any]
    
    # Scores
    sharpe: float
    stability: float
    coverage: float = 0.0
    rwa: float = 0.0
    hitrate: float = 0.0
    final_score: float = 0.0
    
    # Metadata
    fold_count: int = 0
    elapsed_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrialRecord":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class ParamAttribution:
    """Attribution scores for a single parameter."""
    
    param_name: str
    impact_sharpe: float = 0.0       # corr(Δparam, ΔSharpe)
    impact_stability: float = 0.0    # corr(Δparam, ΔStability)
    impact_combined: float = 0.0     # impact_sharpe + impact_stability
    
    # Statistics
    n_samples: int = 0
    mean_value: float = 0.0
    std_value: float = 0.0
    
    # Elite statistics (from top trials)
    elite_mean: float = 0.0
    elite_std: float = 0.0
    elite_count: int = 0


@dataclass
class ParamInteraction:
    """Detected interaction between two parameters."""
    
    param_a: str
    param_b: str
    interaction_strength: float = 0.0
    
    # Interaction type: "synergistic" or "antagonistic"
    interaction_type: str = "unknown"
    
    # Optimal conditions
    optimal_a_range: Tuple[float, float] = (0.0, 1.0)
    optimal_b_range: Tuple[float, float] = (0.0, 1.0)
    
    # SHAP interaction value
    shap_interaction: float = 0.0


@dataclass
class MetaOptimizerConfig:
    """Configuration for the meta-optimizer."""
    
    # Memory settings
    history_size: int = 200        # Last N trials to keep
    elite_percentile: float = 0.10  # Top 10% = elite trials
    
    # Attribution settings
    min_samples_for_attribution: int = 10
    attribution_update_frequency: int = 5  # Update every N trials
    
    # Interaction learning settings
    enable_interaction_learning: bool = True
    interaction_update_frequency: int = 20  # Retrain model every N trials
    top_k_interactions: int = 10  # Store top-K interactions
    
    # Search space adaptation
    enable_search_space_adaptation: bool = True
    adaptation_strength: float = 0.3  # How much to shift distributions
    min_trials_before_adaptation: int = 30
    
    # Persistence
    save_path: Optional[Path] = None
    auto_save_frequency: int = 10


# =============================================================================
# PART 2: Trial Memory
# =============================================================================

class TrialMemory:
    """Memory system for storing trial history and elite trials.
    
    Maintains:
    - history: Last N trials (configurable, default 200)
    - elite: Top 10% trials based on Sharpe * Stability
    - baseline_stats: Rolling mean/std of Sharpe, Stability
    """
    
    def __init__(self, config: MetaOptimizerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or LOGGER
        
        # Trial history (deque for O(1) append/popleft)
        self._history: Deque[TrialRecord] = deque(maxlen=config.history_size)
        
        # Elite trials (top percentile)
        self._elite: List[TrialRecord] = []
        
        # Baseline statistics (rolling)
        self._sharpe_history: List[float] = []
        self._stability_history: List[float] = []
    
    @property
    def history(self) -> List[TrialRecord]:
        """Get trial history as list."""
        return list(self._history)
    
    @property
    def elite(self) -> List[TrialRecord]:
        """Get elite trials."""
        return self._elite
    
    @property
    def size(self) -> int:
        """Number of trials in memory."""
        return len(self._history)
    
    def add_trial(self, record: TrialRecord):
        """Add a trial record to memory."""
        self._history.append(record)
        self._sharpe_history.append(record.sharpe)
        self._stability_history.append(record.stability)
        
        # Update elite set
        self._update_elite()
    
    def _update_elite(self):
        """Update elite trial set based on Sharpe * Stability."""
        if len(self._history) < 5:
            return
        
        # Compute elite score: Sharpe * Stability
        scored_trials = [
            (t, t.sharpe * t.stability)
            for t in self._history
            if not np.isnan(t.sharpe) and not np.isnan(t.stability)
        ]
        
        if not scored_trials:
            return
        
        # Sort by score descending
        scored_trials.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top percentile
        elite_count = max(1, int(len(scored_trials) * self.config.elite_percentile))
        self._elite = [t for t, _ in scored_trials[:elite_count]]
        
        self.logger.debug(
            f"Elite updated: {len(self._elite)} trials "
            f"(top {self.config.elite_percentile*100:.0f}% of {len(scored_trials)})"
        )
    
    def get_baseline_stats(self) -> Dict[str, float]:
        """Get rolling baseline statistics."""
        if len(self._sharpe_history) < 2:
            return {
                "sharpe_mean": 0.0,
                "sharpe_std": 1.0,
                "stability_mean": 0.0,
                "stability_std": 1.0,
            }
        
        return {
            "sharpe_mean": float(np.nanmean(self._sharpe_history)),
            "sharpe_std": float(np.nanstd(self._sharpe_history)) + 1e-8,
            "stability_mean": float(np.nanmean(self._stability_history)),
            "stability_std": float(np.nanstd(self._stability_history)) + 1e-8,
        }
    
    def get_elite_stats(self) -> Dict[str, float]:
        """Get statistics from elite trials."""
        if not self._elite:
            return {}
        
        return {
            "elite_count": len(self._elite),
            "elite_sharpe_mean": float(np.mean([t.sharpe for t in self._elite])),
            "elite_stability_mean": float(np.mean([t.stability for t in self._elite])),
            "elite_combined_mean": float(np.mean([t.sharpe * t.stability for t in self._elite])),
        }
    
    def get_param_dataframe(self) -> pd.DataFrame:
        """Convert trial history to DataFrame for analysis."""
        if not self._history:
            return pd.DataFrame()
        
        records = []
        for t in self._history:
            row = {
                "trial_id": t.trial_id,
                "sharpe": t.sharpe,
                "stability": t.stability,
                "coverage": t.coverage,
                "rwa": t.rwa,
                "hitrate": t.hitrate,
                "final_score": t.final_score,
                **t.params
            }
            records.append(row)
        
        return pd.DataFrame(records)
    
    def save(self, path: Path):
        """Save memory to disk."""
        data = {
            "history": [t.to_dict() for t in self._history],
            "elite": [t.to_dict() for t in self._elite],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        self.logger.info(f"Saved trial memory to {path}")
    
    def load(self, path: Path):
        """Load memory from disk."""
        if not path.exists():
            self.logger.warning(f"Memory file not found: {path}")
            return
        
        data = json.loads(path.read_text())
        self._history = deque(
            [TrialRecord.from_dict(d) for d in data.get("history", [])],
            maxlen=self.config.history_size
        )
        self._elite = [TrialRecord.from_dict(d) for d in data.get("elite", [])]
        
        # Rebuild baseline stats
        self._sharpe_history = [t.sharpe for t in self._history]
        self._stability_history = [t.stability for t in self._history]
        
        self.logger.info(f"Loaded {len(self._history)} trials from {path}")


# =============================================================================
# PART 3: Attribution Engine
# =============================================================================

class AttributionEngine:
    """Computes parameter importance via delta-correlation.
    
    For trial k:
        ΔSharpe    = Sharpe_k - Sharpe_(k-1)
        ΔStability = Stability_k - Stability_(k-1)
    
    For every hyperparameter:
        Δparam_i = param_i(k) - param_i(k-1)
    
    Attribution score:
        impact_i = corr(Δparam_i, ΔSharpe) + corr(Δparam_i, ΔStability)
    
    Impact > 0 → helpful
    Impact < 0 → harmful
    """
    
    def __init__(self, config: MetaOptimizerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or LOGGER
        
        # Attribution storage
        self._attributions: Dict[str, ParamAttribution] = {}
        
        # Delta history for correlation
        self._delta_sharpe: List[float] = []
        self._delta_stability: List[float] = []
        self._delta_params: Dict[str, List[float]] = {}
        
        # Last trial for delta computation
        self._last_trial: Optional[TrialRecord] = None
    
    @property
    def attributions(self) -> Dict[str, ParamAttribution]:
        """Get current attributions."""
        return self._attributions
    
    def update(self, memory: TrialMemory):
        """Update attributions based on trial memory."""
        history = memory.history
        if len(history) < self.config.min_samples_for_attribution:
            self.logger.debug(
                f"Insufficient samples for attribution ({len(history)} < "
                f"{self.config.min_samples_for_attribution})"
            )
            return
        
        # Compute deltas between consecutive trials
        self._compute_deltas(history)
        
        # Compute correlations for each parameter
        self._compute_correlations()
        
        # Add elite statistics
        self._add_elite_stats(memory.elite)
    
    def _compute_deltas(self, history: List[TrialRecord]):
        """Compute deltas between consecutive trials."""
        self._delta_sharpe = []
        self._delta_stability = []
        self._delta_params = {}
        
        prev_trial: Optional[TrialRecord] = None
        
        for trial in history:
            if prev_trial is None:
                prev_trial = trial
                continue
            
            # Score deltas
            delta_sharpe = trial.sharpe - prev_trial.sharpe
            delta_stability = trial.stability - prev_trial.stability
            
            if np.isnan(delta_sharpe) or np.isnan(delta_stability):
                prev_trial = trial
                continue
            
            self._delta_sharpe.append(delta_sharpe)
            self._delta_stability.append(delta_stability)
            
            # Parameter deltas
            for param, value in trial.params.items():
                if param not in prev_trial.params:
                    continue
                
                prev_value = prev_trial.params[param]
                
                # Handle numeric params only
                if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
                    if param not in self._delta_params:
                        self._delta_params[param] = []
                    
                    delta = float(value) - float(prev_value)
                    self._delta_params[param].append(delta)
            
            prev_trial = trial
    
    def _compute_correlations(self):
        """Compute correlation between param deltas and score deltas."""
        if len(self._delta_sharpe) < 5:
            return
        
        delta_sharpe = np.array(self._delta_sharpe)
        delta_stability = np.array(self._delta_stability)
        
        for param, deltas in self._delta_params.items():
            if len(deltas) != len(delta_sharpe):
                # Misaligned - skip
                continue
            
            deltas_arr = np.array(deltas)
            
            # Skip constant parameters
            if np.std(deltas_arr) < 1e-8:
                continue
            
            # Compute correlations
            try:
                corr_sharpe = np.corrcoef(deltas_arr, delta_sharpe)[0, 1]
                corr_stability = np.corrcoef(deltas_arr, delta_stability)[0, 1]
            except Exception:
                corr_sharpe = 0.0
                corr_stability = 0.0
            
            if np.isnan(corr_sharpe):
                corr_sharpe = 0.0
            if np.isnan(corr_stability):
                corr_stability = 0.0
            
            # Combined impact
            impact_combined = corr_sharpe + corr_stability
            
            # Store/update attribution
            self._attributions[param] = ParamAttribution(
                param_name=param,
                impact_sharpe=float(corr_sharpe),
                impact_stability=float(corr_stability),
                impact_combined=float(impact_combined),
                n_samples=len(deltas),
                mean_value=float(np.mean(deltas_arr)),
                std_value=float(np.std(deltas_arr)),
            )
    
    def _add_elite_stats(self, elite: List[TrialRecord]):
        """Add elite statistics to attributions."""
        if not elite:
            return
        
        # Compute mean/std of each param in elite trials
        elite_params: Dict[str, List[float]] = {}
        
        for trial in elite:
            for param, value in trial.params.items():
                if isinstance(value, (int, float)):
                    if param not in elite_params:
                        elite_params[param] = []
                    elite_params[param].append(float(value))
        
        # Update attributions with elite stats
        for param, values in elite_params.items():
            if param in self._attributions:
                self._attributions[param].elite_mean = float(np.mean(values))
                self._attributions[param].elite_std = float(np.std(values))
                self._attributions[param].elite_count = len(values)
    
    def get_top_params(self, n: int = 10, positive_only: bool = False) -> List[ParamAttribution]:
        """Get top N most impactful parameters."""
        attrs = list(self._attributions.values())
        
        if positive_only:
            attrs = [a for a in attrs if a.impact_combined > 0]
        
        # Sort by absolute impact
        attrs.sort(key=lambda a: abs(a.impact_combined), reverse=True)
        
        return attrs[:n]
    
    def get_harmful_params(self) -> List[ParamAttribution]:
        """Get parameters with negative impact."""
        return [a for a in self._attributions.values() if a.impact_combined < -0.1]
    
    def get_helpful_params(self) -> List[ParamAttribution]:
        """Get parameters with positive impact."""
        return [a for a in self._attributions.values() if a.impact_combined > 0.1]


# =============================================================================
# PART 4: Interaction Learner
# =============================================================================

class InteractionLearner:
    """Learns non-linear parameter interactions via LightGBM.
    
    Trains a quick model:
        model = LightGBMRegressor()
        model.fit(X=params_history, y=sharpe_history)
    
    Extracts:
        - feature_importance
        - SHAP values
        - pair interactions
    
    Example interactions detected:
        - seq_len < 120 works only when hidden_dim < 128
        - track_b_weight > 0.3 works only when regime_threshold < 0.12
    """
    
    def __init__(self, config: MetaOptimizerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or LOGGER
        
        # Model and results
        self._model: Optional[Any] = None  # LightGBM model
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}
        self._interactions: List[ParamInteraction] = []
        self._shap_values: Optional[np.ndarray] = None
    
    @property
    def interactions(self) -> List[ParamInteraction]:
        """Get detected interactions."""
        return self._interactions
    
    @property
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model."""
        return self._feature_importance
    
    def update(self, memory: TrialMemory):
        """Train model and extract interactions."""
        if not LIGHTGBM_AVAILABLE:
            self.logger.debug("LightGBM not available for interaction learning")
            return
        
        if not self.config.enable_interaction_learning:
            return
        
        # Get param dataframe
        df = memory.get_param_dataframe()
        
        if len(df) < 30:
            self.logger.debug(f"Insufficient samples for interaction learning ({len(df)} < 30)")
            return
        
        # Prepare features (numeric params only)
        feature_cols = []
        for col in df.columns:
            if col in ["trial_id", "sharpe", "stability", "coverage", "rwa", "hitrate", "final_score"]:
                continue
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                feature_cols.append(col)
        
        if len(feature_cols) < 2:
            self.logger.debug("Not enough numeric features for interaction learning")
            return
        
        X = df[feature_cols].fillna(0).values
        y = df["sharpe"].fillna(0).values
        
        self._feature_names = feature_cols
        
        # Train LightGBM model
        self._train_model(X, y)
        
        # Extract feature importance
        self._extract_importance()
        
        # Extract interactions via SHAP
        if SHAP_AVAILABLE:
            self._extract_shap_interactions(X)
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train LightGBM regressor."""
        params = {
            # Compact model configuration (recommended)
            "objective": "regression",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": -1,           # Let LGBM find structure
            "num_leaves": 32,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "random_state": 42,
        }
        
        try:
            self._model = lgb.LGBMRegressor(**params)
            self._model.fit(X, y)
            self.logger.debug("LightGBM model trained for interaction learning")
        except Exception as e:
            self.logger.warning(f"Failed to train LightGBM model: {e}")
            self._model = None
    
    def _extract_importance(self):
        """Extract feature importance from model."""
        if self._model is None:
            return
        
        try:
            importances = self._model.feature_importances_
            self._feature_importance = {
                name: float(imp)
                for name, imp in zip(self._feature_names, importances)
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract feature importance: {e}")
    
    def _extract_shap_interactions(self, X: np.ndarray):
        """Extract SHAP interaction values."""
        if self._model is None or not SHAP_AVAILABLE:
            return
        
        try:
            # Use TreeExplainer for LightGBM
            explainer = shap.TreeExplainer(self._model)
            
            # Get SHAP values (not interactions - too expensive)
            self._shap_values = explainer.shap_values(X)
            
            # Compute approximate interactions from SHAP values
            self._compute_interactions_from_shap(X)
            
        except Exception as e:
            self.logger.debug(f"SHAP analysis skipped: {e}")
    
    def _compute_interactions_from_shap(self, X: np.ndarray):
        """Compute interactions from SHAP values and feature correlations."""
        if self._shap_values is None:
            return
        
        n_features = len(self._feature_names)
        if n_features < 2:
            return
        
        # Compute interaction strength as correlation of SHAP values
        interactions = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Correlation between SHAP values of two features
                try:
                    shap_i = self._shap_values[:, i]
                    shap_j = self._shap_values[:, j]
                    
                    # Interaction strength: correlation of SHAP values
                    if np.std(shap_i) > 1e-8 and np.std(shap_j) > 1e-8:
                        corr = np.corrcoef(shap_i, shap_j)[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                    else:
                        corr = 0.0
                    
                    # Also check feature correlation
                    feat_corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    if np.isnan(feat_corr):
                        feat_corr = 0.0
                    
                    # Interaction is when SHAP values are correlated but features are not
                    # OR when both are strongly correlated (synergy/antagonism)
                    interaction_strength = abs(corr) * (1 + abs(feat_corr))
                    
                    if interaction_strength > 0.1:  # Threshold
                        interaction = ParamInteraction(
                            param_a=self._feature_names[i],
                            param_b=self._feature_names[j],
                            interaction_strength=float(interaction_strength),
                            interaction_type="synergistic" if corr > 0 else "antagonistic",
                            shap_interaction=float(corr),
                        )
                        interactions.append(interaction)
                        
                except Exception:
                    continue
        
        # Sort by strength and keep top-K
        interactions.sort(key=lambda x: x.interaction_strength, reverse=True)
        self._interactions = interactions[:self.config.top_k_interactions]
        
        if self._interactions:
            self.logger.info(
                f"Detected {len(self._interactions)} param interactions "
                f"(top: {self._interactions[0].param_a} ↔ {self._interactions[0].param_b})"
            )


# =============================================================================
# PART 5: Search Space Updater
# =============================================================================

@dataclass
class AdaptedParam:
    """Adapted parameter with bounds and prior distribution."""
    
    param_name: str
    original_low: float
    original_high: float
    adapted_low: float
    adapted_high: float
    
    # Prior distribution from elite trials
    prior_mean: float = 0.0
    prior_std: float = 1.0
    elite_values: List[float] = field(default_factory=list)
    
    # Adaptation metadata
    rule_applied: str = "none"  # "contract", "expand", "elite_bias"
    impact_score: float = 0.0
    
    def sample_from_prior(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample a value biased toward elite distribution."""
        if rng is None:
            rng = np.random.default_rng()
        
        if len(self.elite_values) >= 3:
            # Mixture sampling: 70% from elite kernel, 30% uniform exploration
            if rng.random() < 0.7:
                # Sample from kernel density around elite values
                base = rng.choice(self.elite_values)
                noise = rng.normal(0, self.prior_std * 0.3)
                value = base + noise
            else:
                # Uniform exploration within adapted bounds
                value = rng.uniform(self.adapted_low, self.adapted_high)
        else:
            # Not enough elite data - sample uniformly
            value = rng.uniform(self.adapted_low, self.adapted_high)
        
        # Clip to adapted bounds
        return float(np.clip(value, self.adapted_low, self.adapted_high))


class SearchSpaceUpdater:
    """Modifies Optuna search space based on meta-learnings.
    
    Implements three rules:
    
    RULE A - Contract toward good values (high positive impact):
        new_low  = best_value * 0.8
        new_high = best_value * 1.2
        (clipped to original bounds)
    
    RULE B - Expand away from harmful regions (negative impact):
        Widen range away from harmful values
        Example: hidden_dim harmful at >160 → new range [32, 150]
    
    RULE C - Elite-biased sampling:
        P(param) ~ mixture of elite values
        Forces exploration around historically successful values
    """
    
    def __init__(self, config: MetaOptimizerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or LOGGER
        
        # Original search space bounds
        self._original_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Adapted parameters with full metadata
        self._adapted_params: Dict[str, AdaptedParam] = {}
        
        # Current adapted bounds (for compatibility)
        self._adapted_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Suggested values for next trial
        self._suggested_priors: Dict[str, float] = {}
        
        # Track harmful regions for Rule B
        self._harmful_regions: Dict[str, List[Tuple[float, float]]] = {}
        
        # Random generator for sampling
        self._rng = np.random.default_rng()
    
    @property
    def adapted_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get adapted parameter bounds."""
        return self._adapted_bounds
    
    @property
    def suggested_priors(self) -> Dict[str, float]:
        """Get suggested prior values."""
        return self._suggested_priors
    
    def set_original_bounds(
        self,
        bounds_or_param: Union[Dict[str, Tuple[float, float]], str],
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        """Set original search space bounds.
        
        Can be called two ways:
            set_original_bounds({'lr': (0.001, 0.1), 'hidden': (32, 512)})
            set_original_bounds('lr', 0.001, 0.1)
        """
        if isinstance(bounds_or_param, dict):
            # Dict mode
            bounds = bounds_or_param
            self._original_bounds.update(bounds)
            self._adapted_bounds.update(bounds)
            
            for param, (lo, hi) in bounds.items():
                self._adapted_params[param] = AdaptedParam(
                    param_name=param,
                    original_low=lo,
                    original_high=hi,
                    adapted_low=lo,
                    adapted_high=hi,
                )
        else:
            # Single param mode
            param = bounds_or_param
            if low is None or high is None:
                raise ValueError("Must provide low and high when setting single param bounds")
            
            self._original_bounds[param] = (low, high)
            self._adapted_bounds[param] = (low, high)
            self._adapted_params[param] = AdaptedParam(
                param_name=param,
                original_low=low,
                original_high=high,
                adapted_low=low,
                adapted_high=high,
            )
    
    def update(
        self,
        memory: TrialMemory,
        attribution: AttributionEngine,
        interaction: InteractionLearner,
    ):
        """Update search space based on meta-learnings."""
        if not self.config.enable_search_space_adaptation:
            return
        
        if memory.size < self.config.min_trials_before_adaptation:
            self.logger.debug(
                f"Insufficient trials for adaptation ({memory.size} < "
                f"{self.config.min_trials_before_adaptation})"
            )
            return
        
        # Collect elite values for each parameter
        elite_values = self._collect_elite_values(memory.elite)
        
        # Apply the three rules
        for param, attr in attribution.attributions.items():
            if param not in self._original_bounds:
                continue
            
            orig_low, orig_high = self._original_bounds[param]
            elite_vals = elite_values.get(param, [])
            
            # Determine which rule to apply based on impact
            if attr.impact_combined > 0.15:
                # RULE A: Contract toward good values
                self._apply_rule_a_contract(param, attr, elite_vals, orig_low, orig_high)
                
            elif attr.impact_combined < -0.15:
                # RULE B: Expand away from harmful regions
                self._apply_rule_b_expand(param, attr, memory.history, orig_low, orig_high)
            
            # RULE C: Always compute elite-biased prior
            self._apply_rule_c_elite_bias(param, elite_vals, orig_low, orig_high)
        
        # Update adapted_bounds dict for compatibility
        for param, ap in self._adapted_params.items():
            self._adapted_bounds[param] = (ap.adapted_low, ap.adapted_high)
        
        # Apply interaction constraints
        self._apply_interaction_constraints(interaction)
        
        # Log summary
        self._log_adaptations()
    
    def _collect_elite_values(self, elite: List[TrialRecord]) -> Dict[str, List[float]]:
        """Collect parameter values from elite trials."""
        elite_values: Dict[str, List[float]] = {}
        
        for trial in elite:
            for param, value in trial.params.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if param not in elite_values:
                        elite_values[param] = []
                    elite_values[param].append(float(value))
        
        return elite_values
    
    def _apply_rule_a_contract(
        self,
        param: str,
        attr: ParamAttribution,
        elite_vals: List[float],
        orig_low: float,
        orig_high: float,
    ):
        """RULE A: Contract search range toward good values.
        
        For high-impact parameters:
            new_low  = best_value * 0.8
            new_high = best_value * 1.2
        
        Example:
            best_seq_len = 124
            → new search range: [100, 150]
        """
        if not elite_vals or len(elite_vals) < 2:
            return
        
        # Use elite mean as "best value"
        best_value = attr.elite_mean if attr.elite_count >= 2 else np.mean(elite_vals)
        
        # Contract factor based on impact strength (stronger impact = tighter contraction)
        # Impact 0.15-0.3 → contract 0.8-1.2 (20%)
        # Impact 0.3-0.5 → contract 0.85-1.15 (15%)
        # Impact >0.5 → contract 0.9-1.1 (10%)
        if attr.impact_combined > 0.5:
            contract_low, contract_high = 0.9, 1.1
        elif attr.impact_combined > 0.3:
            contract_low, contract_high = 0.85, 1.15
        else:
            contract_low, contract_high = 0.8, 1.2
        
        # Handle zero/negative values correctly
        if best_value > 0:
            new_low = best_value * contract_low
            new_high = best_value * contract_high
        elif best_value < 0:
            # For negative values, swap the multipliers
            new_low = best_value * contract_high
            new_high = best_value * contract_low
        else:
            # best_value is 0 - use range around 0
            orig_range = orig_high - orig_low
            new_low = -orig_range * 0.1
            new_high = orig_range * 0.1
        
        # Clip to original bounds
        new_low = max(orig_low, new_low)
        new_high = min(orig_high, new_high)
        
        # Ensure valid range
        if new_low >= new_high:
            new_low = orig_low
            new_high = orig_high
        
        # Update adapted param
        if param in self._adapted_params:
            self._adapted_params[param].adapted_low = new_low
            self._adapted_params[param].adapted_high = new_high
            self._adapted_params[param].rule_applied = "contract"
            self._adapted_params[param].impact_score = attr.impact_combined
        
        self.logger.debug(
            f"RULE A (contract): {param} [{orig_low:.3f}, {orig_high:.3f}] → "
            f"[{new_low:.3f}, {new_high:.3f}] (best={best_value:.3f}, impact={attr.impact_combined:.3f})"
        )
    
    def _apply_rule_b_expand(
        self,
        param: str,
        attr: ParamAttribution,
        history: List[TrialRecord],
        orig_low: float,
        orig_high: float,
    ):
        """RULE B: Expand search range away from harmful regions.
        
        If impact < 0, widen range away from harmful values.
        
        Example:
            hidden_dim repeatedly harmful at >160
            → new search: [32, 150]
        """
        # Collect values from poor trials (bottom 25%)
        poor_trials = []
        for trial in history:
            if trial.sharpe < np.percentile([t.sharpe for t in history], 25):
                if param in trial.params:
                    value = trial.params[param]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        poor_trials.append(float(value))
        
        if len(poor_trials) < 3:
            return
        
        # Find harmful region (cluster of poor values)
        poor_mean = np.mean(poor_trials)
        poor_std = np.std(poor_trials)
        
        # Harmful region: mean ± 1.5*std
        harmful_low = poor_mean - 1.5 * poor_std
        harmful_high = poor_mean + 1.5 * poor_std
        
        # Store harmful region for future reference
        if param not in self._harmful_regions:
            self._harmful_regions[param] = []
        self._harmful_regions[param].append((harmful_low, harmful_high))
        
        # Determine new range that avoids harmful region
        orig_range = orig_high - orig_low
        
        if harmful_high < orig_high and harmful_low > orig_low:
            # Harmful region is in the middle - expand away from it
            # Use the larger non-harmful region
            left_region = harmful_low - orig_low
            right_region = orig_high - harmful_high
            
            if left_region >= right_region:
                # Use left region
                new_low = orig_low
                new_high = min(harmful_low, orig_high)
            else:
                # Use right region
                new_low = max(harmful_high, orig_low)
                new_high = orig_high
                
        elif harmful_low <= orig_low:
            # Harmful region is at the low end
            new_low = max(harmful_high, orig_low)
            new_high = orig_high
            
        elif harmful_high >= orig_high:
            # Harmful region is at the high end
            new_low = orig_low
            new_high = min(harmful_low, orig_high)
            
        else:
            # Fallback - no clear harmful region
            new_low = orig_low
            new_high = orig_high
        
        # Ensure valid range (at least 20% of original)
        min_range = orig_range * 0.2
        if new_high - new_low < min_range:
            # Can't avoid harmful region entirely - keep original
            new_low = orig_low
            new_high = orig_high
        
        # Update adapted param
        if param in self._adapted_params:
            self._adapted_params[param].adapted_low = new_low
            self._adapted_params[param].adapted_high = new_high
            self._adapted_params[param].rule_applied = "expand"
            self._adapted_params[param].impact_score = attr.impact_combined
        
        self.logger.debug(
            f"RULE B (expand): {param} [{orig_low:.3f}, {orig_high:.3f}] → "
            f"[{new_low:.3f}, {new_high:.3f}] (harmful=[{harmful_low:.3f}, {harmful_high:.3f}])"
        )
    
    def _apply_rule_c_elite_bias(
        self,
        param: str,
        elite_vals: List[float],
        orig_low: float,
        orig_high: float,
    ):
        """RULE C: Bias sampling using elite-memory.
        
        Create a prior distribution:
            P(param) ~ mixture of elite values
        
        This forces exploration around historically successful values.
        """
        if param not in self._adapted_params:
            return
        
        ap = self._adapted_params[param]
        ap.elite_values = elite_vals.copy()
        
        if len(elite_vals) >= 2:
            ap.prior_mean = float(np.mean(elite_vals))
            ap.prior_std = float(np.std(elite_vals)) + 1e-8
            
            # Store suggested prior (single value for simple API)
            self._suggested_priors[param] = ap.prior_mean
            
            # Mark rule if not already set
            if ap.rule_applied == "none":
                ap.rule_applied = "elite_bias"
    
    def _apply_interaction_constraints(self, interaction: InteractionLearner):
        """Apply discovered interaction constraints.
        
        For synergistic interactions: keep both params in similar relative position
        For antagonistic interactions: bias toward successful combinations
        """
        for inter in interaction.interactions[:5]:  # Top 5 interactions
            if inter.interaction_strength < 0.2:
                continue
            
            param_a = inter.param_a
            param_b = inter.param_b
            
            if param_a not in self._adapted_params or param_b not in self._adapted_params:
                continue
            
            # For now, just log the interaction
            # Future: implement conditional constraints
            self.logger.debug(
                f"Interaction constraint: {param_a} ↔ {param_b} "
                f"(strength={inter.interaction_strength:.3f}, type={inter.interaction_type})"
            )
    
    def _log_adaptations(self):
        """Log summary of search space adaptations."""
        contracted = []
        expanded = []
        biased = []
        
        for param, ap in self._adapted_params.items():
            if ap.rule_applied == "contract":
                contracted.append(param)
            elif ap.rule_applied == "expand":
                expanded.append(param)
            elif ap.rule_applied == "elite_bias":
                biased.append(param)
        
        if contracted or expanded or biased:
            self.logger.info(
                f"Search space adapted: {len(contracted)} contracted, "
                f"{len(expanded)} expanded, {len(biased)} elite-biased"
            )
    
    def sample_prior(self, param: str) -> Optional[float]:
        """Sample a value from the elite-biased prior distribution.
        
        RULE C implementation: Returns a value biased toward elite distribution.
        """
        if param not in self._adapted_params:
            return None
        
        return self._adapted_params[param].sample_from_prior(self._rng)
    
    def get_adapted_param(self, param: str) -> Optional[AdaptedParam]:
        """Get the full adapted parameter info."""
        return self._adapted_params.get(param)
    
    def get_optuna_suggest_kwargs(self, param: str) -> Dict[str, Any]:
        """Get kwargs for Optuna suggest_* calls with adapted bounds."""
        if param not in self._adapted_params:
            return {}
        
        ap = self._adapted_params[param]
        
        return {
            "low": ap.adapted_low,
            "high": ap.adapted_high,
        }
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of all adaptations."""
        summary = {
            "total_params": len(self._adapted_params),
            "contracted": [],
            "expanded": [],
            "elite_biased": [],
            "harmful_regions": {},
        }
        
        for param, ap in self._adapted_params.items():
            info = {
                "param": param,
                "original": (ap.original_low, ap.original_high),
                "adapted": (ap.adapted_low, ap.adapted_high),
                "impact": ap.impact_score,
            }
            
            if ap.rule_applied == "contract":
                summary["contracted"].append(info)
            elif ap.rule_applied == "expand":
                summary["expanded"].append(info)
            elif ap.rule_applied == "elite_bias":
                summary["elite_biased"].append(info)
        
        summary["harmful_regions"] = {
            param: regions
            for param, regions in self._harmful_regions.items()
        }
        
        return summary


# =============================================================================
# PART 7: Meta-Learning Feedback System
# =============================================================================

@dataclass
class FeedbackAction:
    """A feedback action to apply to search space."""
    param: str
    action_type: str  # "tighten", "widen", "reinforce", "discard"
    magnitude: float  # Strength of action
    reason: str


class MetaLearningFeedback:
    """Meta-learning feedback system for continuous improvement.
    
    PART 7 Implementation:
    
    After each trial:
        Step 1: Compare to baseline (Sharpe_z = (Sharpe - mean) / std)
        Step 2: If significantly better -> tighten ranges, reinforce multipliers
        Step 3: If worse -> widen ranges, discard harmful interactions
    
    This creates continuous improvement, not stagnation.
    """
    
    def __init__(
        self,
        config: MetaOptimizerConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or LOGGER
        
        # Significance thresholds
        self.z_threshold_good: float = 1.0   # z > 1.0 = significantly better
        self.z_threshold_bad: float = -0.5   # z < -0.5 = worse than baseline
        
        # Feedback history
        self._feedback_history: List[FeedbackAction] = []
        self._consecutive_improvements: int = 0
        self._consecutive_degradations: int = 0
        
        # Regime threshold tracking (for Part 6 integration)
        self._threshold_param_names = [
            "threshold", "bull_long_mult", "bull_short_mult",
            "bear_long_mult", "bear_short_mult",
            "crisis_long_mult", "crisis_short_mult",
            "conf_threshold", "vol_scaler"
        ]
    
    def compute_trial_feedback(
        self,
        trial: TrialRecord,
        memory: TrialMemory,
        attribution: AttributionEngine,
        search_space: SearchSpaceUpdater,
    ) -> List[FeedbackAction]:
        """Compute feedback actions for a trial.
        
        Step 1: Compare to baseline
        Step 2: Apply feedback rules based on performance
        
        Returns:
            List of FeedbackAction to apply
        """
        actions: List[FeedbackAction] = []
        
        # Step 1: Compare to baseline
        baseline = memory.get_baseline_stats()
        if baseline["sharpe_std"] < 1e-6:
            return actions  # Not enough variance yet
        
        sharpe_z = (trial.sharpe - baseline["sharpe_mean"]) / baseline["sharpe_std"]
        stability_z = (trial.stability - baseline["stability_mean"]) / max(baseline["stability_std"], 1e-6)
        
        self.logger.debug(
            f"Trial {trial.trial_id}: Sharpe_z={sharpe_z:.3f}, Stability_z={stability_z:.3f}"
        )
        
        # Step 2 & 3: Apply feedback based on performance
        if sharpe_z > self.z_threshold_good:
            # Significantly better trial
            self._consecutive_improvements += 1
            self._consecutive_degradations = 0
            
            actions.extend(self._feedback_on_improvement(
                trial, memory, attribution, search_space, sharpe_z
            ))
            
        elif sharpe_z < self.z_threshold_bad:
            # Worse than baseline
            self._consecutive_degradations += 1
            self._consecutive_improvements = 0
            
            actions.extend(self._feedback_on_degradation(
                trial, memory, attribution, search_space, sharpe_z
            ))
        
        # Store feedback history
        self._feedback_history.extend(actions)
        
        return actions
    
    def _feedback_on_improvement(
        self,
        trial: TrialRecord,
        memory: TrialMemory,
        attribution: AttributionEngine,
        search_space: SearchSpaceUpdater,
        sharpe_z: float,
    ) -> List[FeedbackAction]:
        """Apply feedback when trial is significantly better.
        
        Step 2 actions:
        - Tighten threshold ranges toward successful values
        - Reinforce multipliers near best values
        - Tune conf_threshold around successful values
        - Shrink threshold multipliers if high volatility leads to better Sharpe
        """
        actions: List[FeedbackAction] = []
        
        # Magnitude scales with how good the trial was
        magnitude = min(1.0, (sharpe_z - self.z_threshold_good) / 2.0 + 0.3)
        
        for param, value in trial.params.items():
            if not isinstance(value, (int, float)):
                continue
                
            # Check if param is in search space
            if param not in search_space._adapted_params:
                continue
            
            ap = search_space._adapted_params[param]
            
            # Tighten range toward this good value
            current_range = ap.adapted_high - ap.adapted_low
            
            if current_range > 0:
                # Contract toward good value
                tighten_factor = 0.1 * magnitude
                new_range = current_range * (1 - tighten_factor)
                
                # Center on successful value (weighted toward it)
                center = value * 0.7 + (ap.adapted_low + ap.adapted_high) / 2 * 0.3
                
                new_low = max(ap.original_low, center - new_range / 2)
                new_high = min(ap.original_high, center + new_range / 2)
                
                if new_high > new_low:
                    actions.append(FeedbackAction(
                        param=param,
                        action_type="tighten",
                        magnitude=magnitude,
                        reason=f"Sharpe_z={sharpe_z:.2f}, reinforcing toward {value:.4f}"
                    ))
                    
                    # Apply the tightening
                    ap.adapted_low = new_low
                    ap.adapted_high = new_high
        
        # Special handling for regime threshold params
        for param in self._threshold_param_names:
            if param in trial.params and param in search_space._adapted_params:
                value = trial.params[param]
                ap = search_space._adapted_params[param]
                
                # Reinforce successful threshold values
                if ap.prior_mean == 0 or ap.prior_mean is None:
                    ap.prior_mean = float(value)
                else:
                    # Exponential moving average toward successful value
                    ap.prior_mean = ap.prior_mean * 0.7 + float(value) * 0.3
                
                actions.append(FeedbackAction(
                    param=param,
                    action_type="reinforce",
                    magnitude=magnitude,
                    reason=f"Regime param reinforced to {ap.prior_mean:.4f}"
                ))
        
        if actions:
            self.logger.info(
                f"Meta-feedback: {len(actions)} tighten/reinforce actions "
                f"(consecutive improvements: {self._consecutive_improvements})"
            )
        
        return actions
    
    def _feedback_on_degradation(
        self,
        trial: TrialRecord,
        memory: TrialMemory,
        attribution: AttributionEngine,
        search_space: SearchSpaceUpdater,
        sharpe_z: float,
    ) -> List[FeedbackAction]:
        """Apply feedback when trial is worse than baseline.
        
        Step 3 actions:
        - Widen ranges to increase exploration
        - Discard harmful interactions
        - Increase diversification in next sampling
        """
        actions: List[FeedbackAction] = []
        
        # Magnitude scales with how bad the trial was
        magnitude = min(1.0, abs(sharpe_z - self.z_threshold_bad) / 2.0 + 0.2)
        
        for param, value in trial.params.items():
            if not isinstance(value, (int, float)):
                continue
                
            if param not in search_space._adapted_params:
                continue
            
            ap = search_space._adapted_params[param]
            
            # Check if this value is in a "contracted" region that's now failing
            if ap.rule_applied == "contract":
                # Widen back toward original bounds
                widen_factor = 0.15 * magnitude
                
                current_low = ap.adapted_low
                current_high = ap.adapted_high
                original_range = ap.original_high - ap.original_low
                
                # Expand toward original bounds
                new_low = current_low - original_range * widen_factor
                new_high = current_high + original_range * widen_factor
                
                new_low = max(ap.original_low, new_low)
                new_high = min(ap.original_high, new_high)
                
                if new_high > new_low:
                    actions.append(FeedbackAction(
                        param=param,
                        action_type="widen",
                        magnitude=magnitude,
                        reason=f"Sharpe_z={sharpe_z:.2f}, expanding search"
                    ))
                    
                    ap.adapted_low = new_low
                    ap.adapted_high = new_high
                    ap.rule_applied = "widen"  # Mark as widened
        
        # Track harmful value regions
        for param, value in trial.params.items():
            if not isinstance(value, (int, float)):
                continue
            
            if param not in search_space._harmful_regions:
                search_space._harmful_regions[param] = []
            
            # Add this value as potentially harmful
            search_space._harmful_regions[param].append((value * 0.95, value * 1.05))
            
            actions.append(FeedbackAction(
                param=param,
                action_type="discard",
                magnitude=magnitude,
                reason=f"Marking region around {value:.4f} as harmful"
            ))
        
        if actions:
            self.logger.info(
                f"Meta-feedback: {len(actions)} widen/discard actions "
                f"(consecutive degradations: {self._consecutive_degradations})"
            )
        
        return actions
    
    def get_diversification_boost(self) -> float:
        """Get diversification boost for next sampling.
        
        After consecutive degradations, increase exploration.
        """
        if self._consecutive_degradations >= 3:
            return 0.3  # 30% more exploration
        elif self._consecutive_degradations >= 2:
            return 0.15
        return 0.0
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback actions."""
        action_counts = {}
        for action in self._feedback_history[-50:]:  # Last 50 actions
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
        
        return {
            "total_actions": len(self._feedback_history),
            "recent_actions": action_counts,
            "consecutive_improvements": self._consecutive_improvements,
            "consecutive_degradations": self._consecutive_degradations,
            "diversification_boost": self.get_diversification_boost(),
        }


# =============================================================================
# MAIN: Meta-Optimizer
# =============================================================================

class MetaOptimizer:
    """Self-learning meta-optimizer layer for Optuna.
    
    PART 8: FULL EXECUTION LOOP
    
    for trial in optuna:
        params = suggest_hyperparameters(search_space)
        results = run_model(params)
        
        update_memory(results)
        update_attribution(results)
        update_interactions(results)
        update_search_space(results)
        apply_feedback(results)  # Part 7
        
        return results.sharpe_stability_objective
    
    Capabilities:
    - Discovering better LSTM hyperparameters
    - Learning best Track-A/B family weights
    - Tuning regime thresholds intelligently (Part 6)
    - Adapting based on past success (Part 7)
    - Converging faster and deeper
    - Avoiding over-exploration of bad regions
    
    Usage:
        meta = MetaOptimizer(config)
        
        # After each Optuna trial:
        meta.record_trial(trial_id, params, scores)
        
        # Before next trial:
        suggestions = meta.get_suggestions()
        
        # Periodically:
        meta.analyze()
    """
    
    def __init__(
        self,
        config: Optional[MetaOptimizerConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or MetaOptimizerConfig()
        self.logger = logger or LOGGER
        
        # Components
        self.memory = TrialMemory(self.config, self.logger)
        self.attribution = AttributionEngine(self.config, self.logger)
        self.interaction = InteractionLearner(self.config, self.logger)
        self.search_space = SearchSpaceUpdater(self.config, self.logger)
        self.feedback = MetaLearningFeedback(self.config, self.logger)  # Part 7
        
        # Counters
        self._trial_count: int = 0
        self._last_attribution_update: int = 0
        self._last_interaction_update: int = 0
    
    def record_trial(
        self,
        trial_id: int,
        params: Dict[str, Any],
        sharpe: float,
        stability: float,
        coverage: float = 0.0,
        rwa: float = 0.0,
        hitrate: float = 0.0,
        final_score: float = 0.0,
        fold_count: int = 0,
        elapsed_time: float = 0.0,
    ):
        """Record a completed trial.
        
        Call this after each Optuna trial completes.
        """
        record = TrialRecord(
            trial_id=trial_id,
            params=params,
            sharpe=sharpe,
            stability=stability,
            coverage=coverage,
            rwa=rwa,
            hitrate=hitrate,
            final_score=final_score,
            fold_count=fold_count,
            elapsed_time=elapsed_time,
        )
        
        self.memory.add_trial(record)
        self._trial_count += 1
        
        # Periodic updates
        if self._trial_count - self._last_attribution_update >= self.config.attribution_update_frequency:
            self.attribution.update(self.memory)
            self._last_attribution_update = self._trial_count
        
        if self._trial_count - self._last_interaction_update >= self.config.interaction_update_frequency:
            self.interaction.update(self.memory)
            self._last_interaction_update = self._trial_count
        
        # Update search space
        self.search_space.update(self.memory, self.attribution, self.interaction)
        
        # PART 7: Apply meta-learning feedback
        self.feedback.compute_trial_feedback(
            record, self.memory, self.attribution, self.search_space
        )
        
        # Auto-save
        if self.config.save_path and self._trial_count % self.config.auto_save_frequency == 0:
            self.save()
    
    def analyze(self):
        """Force a full analysis update."""
        self.attribution.update(self.memory)
        self.interaction.update(self.memory)
        self.search_space.update(self.memory, self.attribution, self.interaction)
    
    def get_suggestions(self) -> Dict[str, Any]:
        """Get suggested priors for next trial."""
        return self.search_space.suggested_priors.copy()
    
    def get_adapted_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get adapted parameter bounds."""
        return self.search_space.adapted_bounds.copy()
    
    def get_attribution_summary(self) -> Dict[str, Any]:
        """Get attribution analysis summary."""
        top_helpful = self.attribution.get_top_params(5, positive_only=True)
        top_harmful = self.attribution.get_harmful_params()[:5]
        
        return {
            "total_params_analyzed": len(self.attribution.attributions),
            "helpful": [
                {"param": a.param_name, "impact": a.impact_combined}
                for a in top_helpful
            ],
            "harmful": [
                {"param": a.param_name, "impact": a.impact_combined}
                for a in top_harmful
            ],
        }
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get interaction analysis summary."""
        return {
            "feature_importance": self.interaction.feature_importance,
            "top_interactions": [
                {
                    "params": (i.param_a, i.param_b),
                    "strength": i.interaction_strength,
                    "type": i.interaction_type,
                }
                for i in self.interaction.interactions[:5]
            ],
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        return {
            "total_trials": self.memory.size,
            "elite_count": len(self.memory.elite),
            "baseline_stats": self.memory.get_baseline_stats(),
            "elite_stats": self.memory.get_elite_stats(),
        }
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get meta-learning feedback summary (Part 7)."""
        return self.feedback.get_feedback_summary()
    
    def get_diversification_boost(self) -> float:
        """Get diversification boost for next sampling.
        
        Use this to increase exploration after consecutive bad trials.
        """
        return self.feedback.get_diversification_boost()
    
    def get_full_summary(self) -> Dict[str, Any]:
        """Get complete meta-optimizer summary.
        
        Combines all component summaries into one comprehensive view.
        """
        return {
            "memory": self.get_memory_summary(),
            "attribution": self.get_attribution_summary(),
            "interactions": self.get_interaction_summary(),
            "search_space": self.search_space.get_adaptation_summary(),
            "feedback": self.get_feedback_summary(),
            "trial_count": self._trial_count,
        }
    
    def save(self):
        """Save meta-optimizer state."""
        if not self.config.save_path:
            self.logger.debug("No save_path configured, skipping save")
            return
        
        path = Path(self.config.save_path)
        # 🔧 FIX: Create the save directory itself, not just its parent
        path.mkdir(parents=True, exist_ok=True)
        
        self.memory.save(path / "trial_memory.json")
        
        # Save attribution
        attr_data = {
            param: asdict(attr)
            for param, attr in self.attribution.attributions.items()
        }
        (path / "attributions.json").write_text(json.dumps(attr_data, indent=2, default=str))
        
        # Save interactions
        inter_data = [
            {
                "param_a": i.param_a,
                "param_b": i.param_b,
                "strength": i.interaction_strength,
                "type": i.interaction_type,
            }
            for i in self.interaction.interactions
        ]
        (path / "interactions.json").write_text(json.dumps(inter_data, indent=2))
        
        self.logger.info(f"Saved meta-optimizer state to {path}")
    
    def load(self):
        """Load meta-optimizer state."""
        if not self.config.save_path:
            return
        
        path = Path(self.config.save_path)
        
        if (path / "trial_memory.json").exists():
            self.memory.load(path / "trial_memory.json")
            self._trial_count = self.memory.size
            
            # Rebuild attribution and interactions
            self.analyze()
            
            self.logger.info(f"Loaded meta-optimizer state from {path}")


# =============================================================================
# Integration Helper
# =============================================================================

def create_meta_optimizer(
    save_dir: Optional[Path] = None,
    history_size: int = 200,
    elite_percentile: float = 0.10,
    enable_interactions: bool = True,
) -> MetaOptimizer:
    """Factory function to create a configured MetaOptimizer.
    
    Args:
        save_dir: Directory to save/load state
        history_size: Number of trials to keep in memory
        elite_percentile: Top percentile for elite selection
        enable_interactions: Whether to enable LightGBM interaction learning
    
    Returns:
        Configured MetaOptimizer instance
    """
    config = MetaOptimizerConfig(
        history_size=history_size,
        elite_percentile=elite_percentile,
        enable_interaction_learning=enable_interactions,
        save_path=save_dir,
    )
    
    meta = MetaOptimizer(config)
    
    # Try to load existing state
    if save_dir and Path(save_dir).exists():
        meta.load()
    
    return meta
