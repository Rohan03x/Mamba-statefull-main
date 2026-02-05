"""
Phase-2 Z-Score Explainability Module.

Explains portfolio decisions by decomposing z_final into interpretable
feature contributions. The target is z_final (post-threshold), which is
exactly what enters portfolio optimization.

Signal Chain (explained):
    z_raw = μ / σ_exec
    z_overlays = z_raw × risk_scale × regime_mult × split_discount × hygiene_ok
    z_blend = (1 - w_q) × z_overlays + w_q × quantile_z
    z_scaled = z_blend × vol_scaler
    z_final = threshold(z_scaled)  # TARGET TO EXPLAIN

Feature Vector (per symbol, per day):
    Per-symbol inputs:
        - z_pre_overlay: z_raw before overlays
        - sigma_exec: execution sigma (+ was_clipped flag)
        - risk_scale: from portfolio hygiene
        - regime_multiplier: from regime context
        - hygiene_ok: binary flag (1/0)
        - split_stress: split penalty factor
        - quantile_z: from portfolio parquet

    Global inputs (broadcast to all symbols):
        - day_calib_score: from MambaCalibrationTracker
        - online_trust_score: from portfolio parquet
        - cboe_panic, cboe_slope, cboe_vrp: volatility context
        - portfolio_drawdown: current drawdown
        - portfolio_realized_vol: realized volatility
        - corr_hhi: correlation concentration
        - turnover_prev: previous day turnover
        - cost_prev: previous day transaction cost

Copyright 2024-2026. All rights reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    _has_pd = True
except ImportError:
    _has_pd = False
    pd = None  # type: ignore

HAS_PANDAS: bool = _has_pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Names
# ─────────────────────────────────────────────────────────────────────────────

# Per-symbol features
PER_SYMBOL_FEATURES = [
    "z_pre_overlay",          # z_raw before overlays
    "mu_raw",                 # Raw Mamba μ prediction
    "sigma_exec",             # Execution sigma
    "sigma_was_clipped",      # 1 if sigma was clipped, 0 otherwise
    "risk_scale",             # From portfolio hygiene
    "regime_multiplier",      # From regime context
    "hygiene_ok",             # Binary flag (1/0)
    "split_stress",           # Split penalty factor
    "split_discount",         # 1 - k * split_stress
    "quantile_z",             # From portfolio parquet
    "quantile_blend_weight",  # Weight used for blending
]

# Global features (broadcast to all symbols)
GLOBAL_FEATURES = [
    "day_calib_score",        # From MambaCalibrationTracker
    "online_trust_score",     # From portfolio parquet
    "cboe_panic",             # VIX panic premium
    "cboe_slope",             # VIX term structure slope
    "cboe_vrp",               # Volatility risk premium z-score
    "vol_scaler",             # Policy vol scaler
    "portfolio_drawdown",     # Current portfolio drawdown
    "portfolio_realized_vol", # Realized volatility
    "corr_hhi",               # Correlation HHI
    "turnover_prev",          # Previous day turnover
    "cost_prev",              # Previous day cost
    "equity_level",           # Current equity (normalized)
    "market_regime",          # 0=normal, 1=bull, -1=bear, -2=crisis
]

# All features
ALL_FEATURE_NAMES = PER_SYMBOL_FEATURES + GLOBAL_FEATURES

N_PER_SYMBOL = len(PER_SYMBOL_FEATURES)
N_GLOBAL = len(GLOBAL_FEATURES)
N_TOTAL_FEATURES = N_PER_SYMBOL + N_GLOBAL


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZExplainerSample:
    """Single sample for Z explainer (one symbol, one day)."""
    
    date: str
    symbol: str
    
    # Target: z_final (post-threshold)
    z_final: float
    
    # Feature vector
    features: np.ndarray  # Shape: [N_TOTAL_FEATURES]
    
    # Intermediate z values for debugging
    z_pre_overlay: float = 0.0
    z_post_overlay: float = 0.0
    z_post_blend: float = 0.0
    z_post_scale: float = 0.0


@dataclass
class ZExplainerDay:
    """
    Captures all data needed for z-explanation on a single day.
    This is the structured input collected during Phase-2 loop.
    """
    
    date: str
    day_idx: int
    
    # Per-symbol arrays (n_assets,)
    symbols: List[str]
    mu_raw: np.ndarray
    sigma_exec: np.ndarray
    sigma_was_clipped: np.ndarray  # Boolean or float
    z_pre_overlay: np.ndarray
    risk_scale: np.ndarray
    regime_multiplier: np.ndarray
    hygiene_ok: np.ndarray
    split_stress: np.ndarray
    split_discount: np.ndarray
    quantile_z: np.ndarray
    quantile_blend_weight: float
    z_post_overlay: np.ndarray
    z_post_blend: np.ndarray
    z_post_scale: np.ndarray
    z_final: np.ndarray  # TARGET: post-threshold
    
    # Global scalars
    day_calib_score: float
    online_trust_score: float
    cboe_panic: float
    cboe_slope: float
    cboe_vrp: float
    vol_scaler: float
    portfolio_drawdown: float
    portfolio_realized_vol: float
    corr_hhi: float
    turnover_prev: float
    cost_prev: float
    equity_level: float
    market_regime: int


@dataclass
class ZExplainerBuffer:
    """
    Accumulates ZExplainerDay records during Phase-2 OOS walk-forward.
    Provides methods to build feature matrices for model fitting.
    """
    
    days: List[ZExplainerDay] = field(default_factory=list)
    
    def add(self, day: ZExplainerDay) -> None:
        """Add a day's data to the buffer."""
        self.days.append(day)
    
    def __len__(self) -> int:
        return len(self.days)
    
    def build_feature_matrix(
        self,
        symbols_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
        """
        Build feature matrix X and target vector y from accumulated days.
        
        Args:
            symbols_filter: If provided, only include these symbols.
        
        Returns:
            X: Feature matrix [n_samples, N_TOTAL_FEATURES]
            y: Target vector [n_samples] (z_final)
            index: List of (date, symbol) tuples for each sample
        """
        if not self.days:
            return np.empty((0, N_TOTAL_FEATURES)), np.empty(0), []
        
        X_list = []
        y_list = []
        index_list = []
        
        for day in self.days:
            n_assets = len(day.symbols)
            
            # Build global feature vector (broadcast)
            global_vec = np.array([
                day.day_calib_score,
                day.online_trust_score,
                day.cboe_panic,
                day.cboe_slope,
                day.cboe_vrp,
                day.vol_scaler,
                day.portfolio_drawdown,
                day.portfolio_realized_vol,
                day.corr_hhi,
                day.turnover_prev,
                day.cost_prev,
                day.equity_level,
                float(day.market_regime),
            ], dtype=np.float32)
            
            for j, sym in enumerate(day.symbols):
                if symbols_filter is not None and sym not in symbols_filter:
                    continue
                
                # Per-symbol features
                per_sym_vec = np.array([
                    day.z_pre_overlay[j],
                    day.mu_raw[j],
                    day.sigma_exec[j],
                    float(day.sigma_was_clipped[j]),
                    day.risk_scale[j],
                    day.regime_multiplier[j],
                    float(day.hygiene_ok[j]),
                    day.split_stress[j],
                    day.split_discount[j],
                    day.quantile_z[j],
                    day.quantile_blend_weight,
                ], dtype=np.float32)
                
                # Concatenate: per-symbol + global
                features = np.concatenate([per_sym_vec, global_vec])
                
                X_list.append(features)
                y_list.append(day.z_final[j])
                index_list.append((day.date, sym))
        
        if not X_list:
            return np.empty((0, N_TOTAL_FEATURES)), np.empty(0), []
        
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.float32)
        
        return X, y, index_list
    
    def to_dataframe(self) -> "pd.DataFrame":
        """Convert buffer to pandas DataFrame for analysis."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe()")
        
        X, y, index = self.build_feature_matrix()
        
        df = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
        df["z_final"] = y
        df["date"] = [idx[0] for idx in index]
        df["symbol"] = [idx[1] for idx in index]
        
        # Reorder columns
        cols = ["date", "symbol", "z_final"] + ALL_FEATURE_NAMES
        return df[cols]


# ─────────────────────────────────────────────────────────────────────────────
# Explainer Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureImportance:
    """Feature importance from explainer model."""
    
    feature_name: str
    importance: float  # Absolute importance (magnitude)
    coefficient: float = 0.0  # Signed coefficient (for linear models)
    rank: int = 0


@dataclass
class ZExplanation:
    """Explanation for a single z_final prediction."""
    
    date: str
    symbol: str
    z_final: float  # Actual z_final
    z_predicted: float  # Model prediction
    
    # Top contributing features (sorted by absolute contribution)
    contributions: List[Tuple[str, float]]  # (feature_name, contribution)
    
    # Model info
    model_r2: float = 0.0
    residual: float = 0.0


class LinearZExplainer:
    """
    Simple linear model for explaining z_final.
    
    Fits: z_final = β₀ + Σ βᵢ × xᵢ + ε
    
    Interpretable because:
    - Coefficients show direction and magnitude of each feature's effect
    - Contributions decompose z_final into additive feature effects
    """
    
    def __init__(self, regularization: float = 0.01):
        """
        Args:
            regularization: L2 regularization strength (Ridge)
        """
        self.regularization = regularization
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.feature_names: List[str] = ALL_FEATURE_NAMES.copy()
        self.r2_score: float = 0.0
        self.is_fitted: bool = False
        
        # Training stats
        self.n_samples: int = 0
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        standardize: bool = True,
    ) -> "LinearZExplainer":
        """
        Fit the linear explainer model.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target vector [n_samples]
            standardize: Whether to standardize features before fitting
        
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        
        if n_samples < 10:
            logger.warning("[ZExplainer] Insufficient samples for fitting: %d", n_samples)
            self.coefficients = np.zeros(n_features)
            self.intercept = float(np.mean(y)) if len(y) > 0 else 0.0
            self.is_fitted = False
            return self
        
        # Standardize features
        if standardize:
            self.feature_means = np.nanmean(X, axis=0)
            self.feature_stds = np.nanstd(X, axis=0)
            self.feature_stds[self.feature_stds < 1e-12] = 1.0  # Avoid division by zero
            X_std = (X - self.feature_means) / self.feature_stds
        else:
            X_std = X
            self.feature_means = np.zeros(n_features)
            self.feature_stds = np.ones(n_features)
        
        # Ridge regression: (X'X + λI)⁻¹ X'y
        XtX = X_std.T @ X_std
        XtX += self.regularization * np.eye(n_features)
        Xty = X_std.T @ y
        
        try:
            self.coefficients = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.coefficients = np.linalg.lstsq(X_std, y, rcond=None)[0]
        
        # Compute intercept (using un-standardized prediction)
        y_pred = X_std @ self.coefficients
        self.intercept = float(np.mean(y - y_pred))
        
        # R² score
        y_pred_full = y_pred + self.intercept
        ss_res = np.sum((y - y_pred_full) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r2_score = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        self.is_fitted = True
        logger.info(
            "[ZExplainer] Fitted linear model: R²=%.4f, n_samples=%d",
            self.r2_score, n_samples
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict z_final from features."""
        if not self.is_fitted or self.coefficients is None:
            return np.zeros(X.shape[0])
        
        if self.feature_means is not None and self.feature_stds is not None:
            X_std = (X - self.feature_means) / self.feature_stds
        else:
            X_std = X
        
        return X_std @ self.coefficients + self.intercept
    
    def explain(
        self,
        features: np.ndarray,
        date: str,
        symbol: str,
        z_actual: float,
    ) -> ZExplanation:
        """
        Generate explanation for a single prediction.
        
        Args:
            features: Feature vector [N_TOTAL_FEATURES]
            date: Date string
            symbol: Symbol string
            z_actual: Actual z_final value
        
        Returns:
            ZExplanation with contributions
        """
        if not self.is_fitted or self.coefficients is None:
            return ZExplanation(
                date=date,
                symbol=symbol,
                z_final=z_actual,
                z_predicted=0.0,
                contributions=[],
            )
        
        # Standardize features
        if self.feature_means is not None and self.feature_stds is not None:
            features_std = (features - self.feature_means) / self.feature_stds
        else:
            features_std = features
        
        z_pred = float(features_std @ self.coefficients + self.intercept)
        
        # Compute contributions (feature_std * coefficient)
        contributions_raw = features_std * self.coefficients
        
        # Sort by absolute contribution
        indices = np.argsort(np.abs(contributions_raw))[::-1]
        contributions = [
            (self.feature_names[i], float(contributions_raw[i]))
            for i in indices
        ]
        
        return ZExplanation(
            date=date,
            symbol=symbol,
            z_final=z_actual,
            z_predicted=z_pred,
            contributions=contributions,
            model_r2=self.r2_score,
            residual=z_actual - z_pred,
        )
    
    def get_feature_importance(self) -> List[FeatureImportance]:
        """Get feature importance ranking."""
        if not self.is_fitted or self.coefficients is None:
            return []
        
        importances = []
        for i, name in enumerate(self.feature_names):
            coef = float(self.coefficients[i])
            importances.append(FeatureImportance(
                feature_name=name,
                importance=abs(coef),
                coefficient=coef,
            ))
        
        # Sort by importance
        importances.sort(key=lambda x: x.importance, reverse=True)
        for rank, imp in enumerate(importances):
            imp.rank = rank + 1
        
        return importances
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "regularization": self.regularization,
            "coefficients": self.coefficients.tolist() if self.coefficients is not None else None,
            "intercept": self.intercept,
            "feature_names": self.feature_names,
            "r2_score": self.r2_score,
            "is_fitted": self.is_fitted,
            "n_samples": self.n_samples,
            "feature_means": self.feature_means.tolist() if self.feature_means is not None else None,
            "feature_stds": self.feature_stds.tolist() if self.feature_stds is not None else None,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinearZExplainer":
        """Deserialize from dictionary."""
        explainer = cls(regularization=d.get("regularization", 0.01))
        explainer.coefficients = np.array(d["coefficients"]) if d.get("coefficients") else None
        explainer.intercept = d.get("intercept", 0.0)
        explainer.feature_names = d.get("feature_names", ALL_FEATURE_NAMES.copy())
        explainer.r2_score = d.get("r2_score", 0.0)
        explainer.is_fitted = d.get("is_fitted", False)
        explainer.n_samples = d.get("n_samples", 0)
        explainer.feature_means = np.array(d["feature_means"]) if d.get("feature_means") else None
        explainer.feature_stds = np.array(d["feature_stds"]) if d.get("feature_stds") else None
        return explainer


# ─────────────────────────────────────────────────────────────────────────────
# Tree-Based Explainer (Optional, for non-linear interactions)
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTreeZExplainer:
    """
    Decision tree model for explaining z_final.
    
    Captures non-linear interactions but may be less interpretable
    than linear model for coefficient attribution.
    """
    
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names: List[str] = ALL_FEATURE_NAMES.copy()
        self.r2_score: float = 0.0
        self.is_fitted: bool = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeZExplainer":
        """Fit decision tree."""
        try:
            from sklearn.tree import DecisionTreeRegressor
            
            self.tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )
            self.tree.fit(X, y)
            
            y_pred = self.tree.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.r2_score = 1.0 - (ss_res / (ss_tot + 1e-12))
            
            self.is_fitted = True
            logger.info(
                "[ZExplainer.Tree] Fitted tree model: R²=%.4f, depth=%d",
                self.r2_score, self.tree.get_depth()
            )
        except ImportError:
            logger.warning("[ZExplainer.Tree] sklearn not available")
            self.is_fitted = False
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.tree is None:
            return np.zeros(X.shape[0])
        return self.tree.predict(X)
    
    def get_feature_importance(self) -> List[FeatureImportance]:
        """Get feature importance from tree."""
        if not self.is_fitted or self.tree is None:
            return []
        
        importances = self.tree.feature_importances_
        result = []
        for i, name in enumerate(self.feature_names):
            result.append(FeatureImportance(
                feature_name=name,
                importance=float(importances[i]),
                coefficient=0.0,  # Trees don't have coefficients
            ))
        
        result.sort(key=lambda x: x.importance, reverse=True)
        for rank, imp in enumerate(result):
            imp.rank = rank + 1
        
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Create ZExplainerDay from Phase-2 loop state
# ─────────────────────────────────────────────────────────────────────────────

def create_explainer_day(
    date: str,
    day_idx: int,
    symbols: List[str],
    # Per-symbol
    mu_raw: np.ndarray,
    sigma_exec: np.ndarray,
    sigma_was_clipped: np.ndarray,
    z_pre_overlay: np.ndarray,
    risk_scale: np.ndarray,
    regime_multiplier: np.ndarray,
    hygiene_ok: np.ndarray,
    split_stress: np.ndarray,
    split_discount: np.ndarray,
    quantile_z: np.ndarray,
    quantile_blend_weight: float,
    z_post_overlay: np.ndarray,
    z_post_blend: np.ndarray,
    z_post_scale: np.ndarray,
    z_final: np.ndarray,
    # Global
    day_calib_score: float,
    online_trust_score: float,
    cboe_panic: float,
    cboe_slope: float,
    cboe_vrp: float,
    vol_scaler: float,
    portfolio_drawdown: float,
    portfolio_realized_vol: float,
    corr_hhi: float,
    turnover_prev: float,
    cost_prev: float,
    equity_level: float,
    market_regime: int,
) -> ZExplainerDay:
    """Factory function to create ZExplainerDay from Phase-2 loop state."""
    return ZExplainerDay(
        date=date,
        day_idx=day_idx,
        symbols=symbols,
        mu_raw=np.asarray(mu_raw, dtype=np.float32),
        sigma_exec=np.asarray(sigma_exec, dtype=np.float32),
        sigma_was_clipped=np.asarray(sigma_was_clipped, dtype=np.float32),
        z_pre_overlay=np.asarray(z_pre_overlay, dtype=np.float32),
        risk_scale=np.asarray(risk_scale, dtype=np.float32),
        regime_multiplier=np.asarray(regime_multiplier, dtype=np.float32),
        hygiene_ok=np.asarray(hygiene_ok, dtype=np.float32),
        split_stress=np.asarray(split_stress, dtype=np.float32),
        split_discount=np.asarray(split_discount, dtype=np.float32),
        quantile_z=np.asarray(quantile_z, dtype=np.float32),
        quantile_blend_weight=float(quantile_blend_weight),
        z_post_overlay=np.asarray(z_post_overlay, dtype=np.float32),
        z_post_blend=np.asarray(z_post_blend, dtype=np.float32),
        z_post_scale=np.asarray(z_post_scale, dtype=np.float32),
        z_final=np.asarray(z_final, dtype=np.float32),
        day_calib_score=float(day_calib_score),
        online_trust_score=float(online_trust_score),
        cboe_panic=float(cboe_panic),
        cboe_slope=float(cboe_slope),
        cboe_vrp=float(cboe_vrp),
        vol_scaler=float(vol_scaler),
        portfolio_drawdown=float(portfolio_drawdown),
        portfolio_realized_vol=float(portfolio_realized_vol),
        corr_hhi=float(corr_hhi),
        turnover_prev=float(turnover_prev),
        cost_prev=float(cost_prev),
        equity_level=float(equity_level),
        market_regime=int(market_regime),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation_report(
    explainer: LinearZExplainer,
    buffer: ZExplainerBuffer,
    top_k_features: int = 10,
    top_k_samples: int = 20,
) -> Dict[str, Any]:
    """
    Generate a structured explanation report.
    
    Args:
        explainer: Fitted LinearZExplainer
        buffer: ZExplainerBuffer with accumulated data
        top_k_features: Number of top features to include
        top_k_samples: Number of sample explanations to include
    
    Returns:
        Dictionary with report sections
    """
    report = {
        "model": {
            "type": "linear_ridge",
            "r2_score": explainer.r2_score,
            "n_samples": explainer.n_samples,
            "regularization": explainer.regularization,
        },
        "feature_importance": [],
        "sample_explanations": [],
        "summary_stats": {},
    }
    
    # Feature importance
    importances = explainer.get_feature_importance()
    for imp in importances[:top_k_features]:
        report["feature_importance"].append({
            "rank": imp.rank,
            "feature": imp.feature_name,
            "importance": round(imp.importance, 4),
            "coefficient": round(imp.coefficient, 4),
        })
    
    # Sample explanations (largest |z_final| samples)
    X, y, index = buffer.build_feature_matrix()
    if len(y) > 0:
        # Sort by absolute z_final
        sorted_indices = np.argsort(np.abs(y))[::-1]
        
        for idx in sorted_indices[:top_k_samples]:
            date, symbol = index[idx]
            explanation = explainer.explain(X[idx], date, symbol, float(y[idx]))
            
            report["sample_explanations"].append({
                "date": explanation.date,
                "symbol": explanation.symbol,
                "z_final": round(explanation.z_final, 4),
                "z_predicted": round(explanation.z_predicted, 4),
                "residual": round(explanation.residual, 4),
                "top_contributions": [
                    {"feature": name, "contribution": round(contrib, 4)}
                    for name, contrib in explanation.contributions[:5]
                ],
            })
    
    # Summary stats
    if len(y) > 0:
        report["summary_stats"] = {
            "z_final_mean": round(float(np.mean(y)), 4),
            "z_final_std": round(float(np.std(y)), 4),
            "z_final_min": round(float(np.min(y)), 4),
            "z_final_max": round(float(np.max(y)), 4),
            "n_nonzero": int(np.sum(np.abs(y) > 1e-6)),
            "active_rate": round(float(np.mean(np.abs(y) > 1e-6)), 4),
        }
    
    return report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3-5: Rolling Ridge Surrogate with Fidelity Gating
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurrogateFidelity:
    """Fidelity metrics for the surrogate model."""
    
    correlation: float = 0.0  # Corr(ŷ, y) pooled across symbols
    directional_agreement: float = 0.5  # mean(sign(ŷ) == sign(y))
    r2_score: float = 0.0  # R² on validation window
    n_samples: int = 0  # Number of samples used
    is_reliable: bool = False  # True if passes fidelity gate
    
    # Thresholds
    MIN_CORR: float = 0.3
    MIN_DIR_AGREE: float = 0.55


@dataclass
class SymbolContribution:
    """Feature contributions for a single symbol."""
    
    symbol: str
    z_final: float
    z_predicted: float
    
    # Top contributors: (feature_name, feature_value, coefficient, contribution=β*x)
    top_positive: List[Tuple[str, float, float, float]]  # Increasing z
    top_negative: List[Tuple[str, float, float, float]]  # Decreasing z
    
    # Total explained contribution
    total_contribution: float = 0.0


@dataclass
class DailyExplanation:
    """Complete explanation for one day."""
    
    date: str
    day_idx: int
    
    # Fidelity metrics
    fidelity: SurrogateFidelity
    
    # Global drivers (portfolio-wide aggregate of |βx|)
    global_drivers: List[Tuple[str, float]]  # (feature_name, mean_abs_contribution)
    
    # Per-symbol explanations for top traded symbols
    symbol_contributions: List[SymbolContribution]
    
    # Model info
    n_features: int = 0
    intercept: float = 0.0
    
    # Suppression flag
    explanation_suppressed: bool = False
    suppression_reason: str = ""


class RollingSurrogate:
    """
    Rolling Ridge regression surrogate for z_final explanation.
    
    Features:
    - Rolling window of W days (default: 126-252 days)
    - Refit every K days (default: 5 = weekly)
    - Fidelity gating: only explain if corr >= 0.3 OR dir_agree >= 55%
    - Pooled across symbols/time for stable coefficients
    
    Usage:
        surrogate = RollingSurrogate(window=126, refit_interval=5)
        
        # In Phase-2 loop:
        for i, day in enumerate(oos_days):
            # After computing z_final
            surrogate.add_sample(day_data)
            
            if surrogate.should_refit(i):
                surrogate.refit()
            
            explanation = surrogate.explain_day(day_data)
            if not explanation.explanation_suppressed:
                log_explanation(explanation)
    """
    
    def __init__(
        self,
        window: int = 126,
        max_window: int = 252,
        refit_interval: int = 5,
        regularization: float = 1.0,
        min_samples_for_fit: int = 50,
        top_n_symbols: int = 10,
        top_k_features: int = 5,
    ):
        """
        Args:
            window: Minimum rolling window size (days * symbols)
            max_window: Maximum window size (trim older data)
            refit_interval: Refit every K days
            regularization: Ridge regularization strength (λ)
            min_samples_for_fit: Minimum samples before fitting
            top_n_symbols: Number of top traded symbols to explain
            top_k_features: Number of top features per symbol
        """
        self.window = window
        self.max_window = max_window
        self.refit_interval = refit_interval
        self.regularization = regularization
        self.min_samples_for_fit = min_samples_for_fit
        self.top_n_symbols = top_n_symbols
        self.top_k_features = top_k_features
        
        # Rolling buffer: list of (X_row, y_val, date, symbol)
        self._buffer: List[Tuple[np.ndarray, float, str, str]] = []
        
        # Model state
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.last_refit_idx: int = -1
        self.n_refits: int = 0
        
        # Fidelity tracking
        self.last_fidelity: SurrogateFidelity = SurrogateFidelity()
        
        # Feature names
        self.feature_names: List[str] = ALL_FEATURE_NAMES.copy()
    
    def add_day(self, day: ZExplainerDay) -> None:
        """
        Add all samples from a day to the rolling buffer.
        
        Args:
            day: ZExplainerDay with per-symbol data
        """
        n_assets = len(day.symbols)
        
        # Build global feature vector
        global_vec = np.array([
            day.day_calib_score,
            day.online_trust_score,
            day.cboe_panic,
            day.cboe_slope,
            day.cboe_vrp,
            day.vol_scaler,
            day.portfolio_drawdown,
            day.portfolio_realized_vol,
            day.corr_hhi,
            day.turnover_prev,
            day.cost_prev,
            day.equity_level,
            float(day.market_regime),
        ], dtype=np.float32)
        
        for j, sym in enumerate(day.symbols):
            # Per-symbol features
            per_sym_vec = np.array([
                day.z_pre_overlay[j],
                day.mu_raw[j],
                day.sigma_exec[j],
                float(day.sigma_was_clipped[j]),
                day.risk_scale[j],
                day.regime_multiplier[j],
                float(day.hygiene_ok[j]),
                day.split_stress[j],
                day.split_discount[j],
                day.quantile_z[j],
                day.quantile_blend_weight,
            ], dtype=np.float32)
            
            # Concatenate
            features = np.concatenate([per_sym_vec, global_vec])
            y_val = float(day.z_final[j])
            
            # Only add if z_final is finite
            if np.isfinite(y_val) and np.all(np.isfinite(features)):
                self._buffer.append((features, y_val, day.date, sym))
        
        # Trim to max_window (approximate by n_assets per day)
        max_samples = self.max_window * n_assets
        if len(self._buffer) > max_samples:
            self._buffer = self._buffer[-max_samples:]
    
    def should_refit(self, day_idx: int) -> bool:
        """Check if it's time to refit the model."""
        if len(self._buffer) < self.min_samples_for_fit:
            return False
        if self.last_refit_idx < 0:
            return True
        return (day_idx - self.last_refit_idx) >= self.refit_interval
    
    def refit(self, day_idx: int = 0) -> SurrogateFidelity:
        """
        Refit the Ridge regression model on the rolling buffer.
        
        Returns:
            SurrogateFidelity with model quality metrics
        """
        if len(self._buffer) < self.min_samples_for_fit:
            self.last_fidelity = SurrogateFidelity(is_reliable=False)
            return self.last_fidelity
        
        # Stack data
        X = np.vstack([s[0] for s in self._buffer])
        y = np.array([s[1] for s in self._buffer], dtype=np.float32)
        
        n_samples, n_features = X.shape
        
        # Standardize features
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0)
        self.feature_stds[self.feature_stds < 1e-12] = 1.0
        X_std = (X - self.feature_means) / self.feature_stds
        
        # Ridge regression: (X'X + λI)⁻¹ X'y
        XtX = X_std.T @ X_std
        XtX += self.regularization * np.eye(n_features)
        Xty = X_std.T @ y
        
        try:
            self.coefficients = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(X_std, y, rcond=None)[0]
        
        # Compute intercept
        y_pred = X_std @ self.coefficients
        self.intercept = float(np.mean(y - y_pred))
        y_pred_full = y_pred + self.intercept
        
        # Compute fidelity metrics
        # Correlation
        try:
            corr = float(np.corrcoef(y_pred_full, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        
        # Directional agreement
        # Only count where both are non-zero
        nonzero_mask = (np.abs(y) > 1e-9) | (np.abs(y_pred_full) > 1e-9)
        if np.sum(nonzero_mask) > 10:
            dir_agree = float(np.mean(
                np.sign(y_pred_full[nonzero_mask]) == np.sign(y[nonzero_mask])
            ))
        else:
            dir_agree = 0.5
        
        # R² score
        ss_res = np.sum((y - y_pred_full) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        # Fidelity gate
        is_reliable = (corr >= SurrogateFidelity.MIN_CORR) or (dir_agree >= SurrogateFidelity.MIN_DIR_AGREE)
        
        self.last_fidelity = SurrogateFidelity(
            correlation=corr,
            directional_agreement=dir_agree,
            r2_score=r2,
            n_samples=n_samples,
            is_reliable=is_reliable,
        )
        
        self.is_fitted = True
        self.last_refit_idx = day_idx
        self.n_refits += 1
        
        logger.info(
            "[RollingSurrogate] Refit #%d: corr=%.3f, dir_agree=%.1f%%, R²=%.3f, reliable=%s, n=%d",
            self.n_refits, corr, dir_agree * 100, r2, is_reliable, n_samples
        )
        
        return self.last_fidelity
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict z_final from features."""
        if not self.is_fitted or self.coefficients is None:
            return np.zeros(X.shape[0] if X.ndim > 1 else 1)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_std = (X - self.feature_means) / self.feature_stds
        return X_std @ self.coefficients + self.intercept
    
    def compute_contributions(self, features: np.ndarray) -> np.ndarray:
        """
        Compute per-feature contributions: contrib_k = β_k × x_k (standardized).
        
        Returns:
            Array of contributions [n_features]
        """
        if not self.is_fitted or self.coefficients is None:
            return np.zeros(len(self.feature_names))
        
        features_std = (features - self.feature_means) / self.feature_stds
        return features_std * self.coefficients
    
    def explain_day(
        self,
        day: ZExplainerDay,
        traded_symbols: Optional[List[str]] = None,
    ) -> DailyExplanation:
        """
        Generate explanation for a single day.
        
        Args:
            day: ZExplainerDay with current day's data
            traded_symbols: If provided, prioritize these symbols for explanation
        
        Returns:
            DailyExplanation with fidelity check and contributions
        """
        explanation = DailyExplanation(
            date=day.date,
            day_idx=day.day_idx,
            fidelity=self.last_fidelity,
            global_drivers=[],
            symbol_contributions=[],
            n_features=len(self.feature_names),
            intercept=self.intercept,
        )
        
        # Check fidelity gate
        if not self.is_fitted:
            explanation.explanation_suppressed = True
            explanation.suppression_reason = "model_not_fitted"
            return explanation
        
        if not self.last_fidelity.is_reliable:
            explanation.explanation_suppressed = True
            explanation.suppression_reason = f"fidelity_gate_failed: corr={self.last_fidelity.correlation:.3f}, dir_agree={self.last_fidelity.directional_agreement:.3f}"
            return explanation
        
        n_assets = len(day.symbols)
        
        # Build feature matrix for all symbols
        global_vec = np.array([
            day.day_calib_score,
            day.online_trust_score,
            day.cboe_panic,
            day.cboe_slope,
            day.cboe_vrp,
            day.vol_scaler,
            day.portfolio_drawdown,
            day.portfolio_realized_vol,
            day.corr_hhi,
            day.turnover_prev,
            day.cost_prev,
            day.equity_level,
            float(day.market_regime),
        ], dtype=np.float32)
        
        all_contributions = []
        symbol_features = []
        
        for j, sym in enumerate(day.symbols):
            per_sym_vec = np.array([
                day.z_pre_overlay[j],
                day.mu_raw[j],
                day.sigma_exec[j],
                float(day.sigma_was_clipped[j]),
                day.risk_scale[j],
                day.regime_multiplier[j],
                float(day.hygiene_ok[j]),
                day.split_stress[j],
                day.split_discount[j],
                day.quantile_z[j],
                day.quantile_blend_weight,
            ], dtype=np.float32)
            
            features = np.concatenate([per_sym_vec, global_vec])
            symbol_features.append((sym, features, float(day.z_final[j])))
            
            # Compute contributions
            contribs = self.compute_contributions(features)
            all_contributions.append(contribs)
        
        contributions_arr = np.vstack(all_contributions)  # [n_assets, n_features]
        
        # ─────────────────────────────────────────────────────────────────────
        # Global drivers: portfolio-wide mean |βx| per feature
        # ─────────────────────────────────────────────────────────────────────
        mean_abs_contrib = np.mean(np.abs(contributions_arr), axis=0)
        sorted_indices = np.argsort(mean_abs_contrib)[::-1]
        
        explanation.global_drivers = [
            (self.feature_names[i], float(mean_abs_contrib[i]))
            for i in sorted_indices[:self.top_k_features * 2]
        ]
        
        # ─────────────────────────────────────────────────────────────────────
        # Per-symbol explanations for top N traded symbols
        # ─────────────────────────────────────────────────────────────────────
        # Determine which symbols to explain
        if traded_symbols:
            # Use provided list (e.g., symbols with largest |w|)
            syms_to_explain = [s for s in traded_symbols if s in day.symbols][:self.top_n_symbols]
        else:
            # Use symbols with largest |z_final|
            z_abs = np.abs(day.z_final)
            top_indices = np.argsort(z_abs)[::-1][:self.top_n_symbols]
            syms_to_explain = [day.symbols[i] for i in top_indices]
        
        for sym in syms_to_explain:
            if sym not in day.symbols:
                continue
            
            j = day.symbols.index(sym)
            sym, features, z_actual = symbol_features[j]
            z_pred = float(self.predict(features)[0])
            contribs = contributions_arr[j]
            
            # Sort contributions
            sorted_contrib_idx = np.argsort(contribs)
            
            # Top positive contributors (largest positive βx)
            top_pos = []
            for idx in sorted_contrib_idx[::-1][:self.top_k_features]:
                if contribs[idx] > 0:
                    top_pos.append((
                        self.feature_names[idx],
                        float(features[idx]),
                        float(self.coefficients[idx]),
                        float(contribs[idx]),
                    ))
            
            # Top negative contributors (most negative βx)
            top_neg = []
            for idx in sorted_contrib_idx[:self.top_k_features]:
                if contribs[idx] < 0:
                    top_neg.append((
                        self.feature_names[idx],
                        float(features[idx]),
                        float(self.coefficients[idx]),
                        float(contribs[idx]),
                    ))
            
            explanation.symbol_contributions.append(SymbolContribution(
                symbol=sym,
                z_final=z_actual,
                z_predicted=z_pred,
                top_positive=top_pos,
                top_negative=top_neg,
                total_contribution=float(np.sum(contribs)),
            ))
        
        return explanation
    
    def get_global_coefficients(self) -> List[Tuple[str, float]]:
        """Get all coefficients sorted by absolute value."""
        if not self.is_fitted or self.coefficients is None:
            return []
        
        sorted_indices = np.argsort(np.abs(self.coefficients))[::-1]
        return [
            (self.feature_names[i], float(self.coefficients[i]))
            for i in sorted_indices
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize surrogate state for persistence."""
        return {
            "window": self.window,
            "max_window": self.max_window,
            "refit_interval": self.refit_interval,
            "regularization": self.regularization,
            "coefficients": self.coefficients.tolist() if self.coefficients is not None else None,
            "intercept": self.intercept,
            "feature_means": self.feature_means.tolist() if self.feature_means is not None else None,
            "feature_stds": self.feature_stds.tolist() if self.feature_stds is not None else None,
            "is_fitted": self.is_fitted,
            "n_refits": self.n_refits,
            "last_fidelity": {
                "correlation": self.last_fidelity.correlation,
                "directional_agreement": self.last_fidelity.directional_agreement,
                "r2_score": self.last_fidelity.r2_score,
                "is_reliable": self.last_fidelity.is_reliable,
            },
            "buffer_size": len(self._buffer),
        }


def format_daily_explanation_log(explanation: DailyExplanation) -> str:
    """
    Format DailyExplanation for logging output.
    
    Returns:
        Formatted string suitable for logging
    """
    lines = []
    
    # Header
    lines.append(f"═══ Z-Explainer Day {explanation.day_idx} ({explanation.date}) ═══")
    
    # Fidelity
    f = explanation.fidelity
    lines.append(f"  Fidelity: corr={f.correlation:.3f}, dir_agree={f.directional_agreement:.1%}, R²={f.r2_score:.3f}, reliable={f.is_reliable}")
    
    if explanation.explanation_suppressed:
        lines.append(f"  ⚠️ EXPLANATION SUPPRESSED: {explanation.suppression_reason}")
        return "\n".join(lines)
    
    # Global drivers
    if explanation.global_drivers:
        lines.append("  Global Drivers (mean |βx|):")
        for feat, val in explanation.global_drivers[:5]:
            lines.append(f"    {feat}: {val:.4f}")
    
    # Top symbols
    if explanation.symbol_contributions:
        lines.append("  Top Traded Symbols:")
        for sc in explanation.symbol_contributions[:5]:
            lines.append(f"    {sc.symbol}: z_final={sc.z_final:.3f}, ŷ={sc.z_predicted:.3f}")
            if sc.top_positive:
                top_pos = sc.top_positive[0]
                lines.append(f"      ↑ {top_pos[0]}: x={top_pos[1]:.3f}, β={top_pos[2]:.3f}, βx={top_pos[3]:.3f}")
            if sc.top_negative:
                top_neg = sc.top_negative[0]
                lines.append(f"      ↓ {top_neg[0]}: x={top_neg[1]:.3f}, β={top_neg[2]:.3f}, βx={top_neg[3]:.3f}")
    
    return "\n".join(lines)


def compute_portfolio_wide_drivers(
    surrogate: RollingSurrogate,
    day: ZExplainerDay,
    weights: np.ndarray,
) -> List[Tuple[str, float]]:
    """
    Compute portfolio-wide weighted feature drivers.
    
    Each feature's importance is weighted by |w| to show what's
    driving the portfolio's overall z-exposure.
    
    Args:
        surrogate: Fitted RollingSurrogate
        day: Current day's data
        weights: Portfolio weights [n_assets]
    
    Returns:
        List of (feature_name, weighted_contribution) sorted by importance
    """
    if not surrogate.is_fitted:
        return []
    
    n_assets = len(day.symbols)
    n_features = len(surrogate.feature_names)
    
    # Build feature matrix
    global_vec = np.array([
        day.day_calib_score, day.online_trust_score, day.cboe_panic,
        day.cboe_slope, day.cboe_vrp, day.vol_scaler, day.portfolio_drawdown,
        day.portfolio_realized_vol, day.corr_hhi, day.turnover_prev,
        day.cost_prev, day.equity_level, float(day.market_regime),
    ], dtype=np.float32)
    
    weighted_contribs = np.zeros(n_features)
    total_abs_weight = np.sum(np.abs(weights))
    
    if total_abs_weight < 1e-9:
        return []
    
    for j, sym in enumerate(day.symbols):
        w = abs(weights[j])
        if w < 1e-9:
            continue
        
        per_sym_vec = np.array([
            day.z_pre_overlay[j], day.mu_raw[j], day.sigma_exec[j],
            float(day.sigma_was_clipped[j]), day.risk_scale[j],
            day.regime_multiplier[j], float(day.hygiene_ok[j]),
            day.split_stress[j], day.split_discount[j],
            day.quantile_z[j], day.quantile_blend_weight,
        ], dtype=np.float32)
        
        features = np.concatenate([per_sym_vec, global_vec])
        contribs = surrogate.compute_contributions(features)
        
        # Weight by position size
        weighted_contribs += (w / total_abs_weight) * np.abs(contribs)
    
    sorted_indices = np.argsort(weighted_contribs)[::-1]
    return [
        (surrogate.feature_names[i], float(weighted_contribs[i]))
        for i in sorted_indices
    ]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Human-Readable Reason Codes
# ─────────────────────────────────────────────────────────────────────────────

# Feature → human-readable descriptions (for narratives)
FEATURE_NARRATIVES: Dict[str, Tuple[str, str]] = {
    # (low_description, high_description)
    # Per-symbol features
    "z_pre_overlay": ("weak raw signal", "strong raw signal"),
    "mu_raw": ("low Mamba prediction", "high Mamba prediction"),
    "sigma_exec": ("low execution uncertainty", "high execution uncertainty"),
    "sigma_was_clipped": ("normal sigma", "sigma clipped (extreme)"),
    "risk_scale": ("risk-off scaling", "full risk allocation"),
    "regime_multiplier": ("regime dampening", "regime amplification"),
    "hygiene_ok": ("hygiene check failed", "hygiene check passed"),
    "split_stress": ("no split stress", "high split stress"),
    "split_discount": ("heavy split discount", "minimal split discount"),
    "quantile_z": ("low quantile rank", "high quantile rank"),
    "quantile_blend_weight": ("low quantile blend", "high quantile blend"),
    
    # Global features
    "day_calib_score": ("low calibration score", "high calibration score"),
    "online_trust_score": ("low model trust", "high model trust"),
    "cboe_panic": ("low panic premium", "high panic premium"),
    "cboe_slope": ("flat VIX curve", "steep VIX curve"),
    "cboe_vrp": ("low vol risk premium", "high vol risk premium"),
    "vol_scaler": ("reduced vol target", "full vol target"),
    "portfolio_drawdown": ("shallow drawdown", "deep drawdown"),
    "portfolio_realized_vol": ("low realized vol", "high realized vol"),
    "corr_hhi": ("dispersed correlations", "concentrated correlations"),
    "turnover_prev": ("low turnover", "high turnover"),
    "cost_prev": ("low transaction costs", "high transaction costs"),
    "equity_level": ("low equity level", "high equity level"),
    "market_regime": ("crisis/bear regime", "bull/normal regime"),
}


def _describe_contribution(
    feature: str,
    value: float,
    contribution: float,
    coefficient: float,
) -> str:
    """
    Generate human-readable description of a feature's contribution.
    
    Args:
        feature: Feature name
        value: Feature value (standardized)
        contribution: β × x contribution
        coefficient: Ridge coefficient β
    
    Returns:
        Human-readable description like "high panic premium" or "low calibration score"
    """
    narratives = FEATURE_NARRATIVES.get(feature, (f"low {feature}", f"high {feature}"))
    
    # Determine if feature is "high" or "low" based on sign of contribution
    # Positive contribution = pushing z up = usually high positive feature or low negative feature
    # For simplicity: use feature value sign
    if value >= 0:
        return narratives[1]  # high
    else:
        return narratives[0]  # low


@dataclass
class ReasonCode:
    """Structured reason code for stakeholder narratives."""
    
    level: str  # "portfolio" or "symbol"
    symbol: Optional[str]  # None for portfolio-level
    direction: str  # "risk_on", "risk_off", "long", "short", "flat", "reduced"
    drivers: List[str]  # Human-readable driver descriptions
    confidence: str  # "high", "medium", "low" based on fidelity
    
    def to_narrative(self) -> str:
        """Generate stakeholder-ready narrative string."""
        if self.level == "portfolio":
            prefix = f"Portfolio {self.direction}"
        else:
            prefix = f"{self.symbol} {self.direction}"
        
        if self.drivers:
            return f"{prefix} mainly driven by: {', '.join(self.drivers[:3])}"
        else:
            return f"{prefix} (no clear drivers identified)"


def generate_portfolio_reason_code(
    explanation: DailyExplanation,
    net_exposure: float = 0.0,
    gross_exposure: float = 0.0,
) -> ReasonCode:
    """
    Generate portfolio-level reason code from daily explanation.
    
    Args:
        explanation: DailyExplanation from RollingSurrogate
        net_exposure: Portfolio net exposure (long - short)
        gross_exposure: Portfolio gross exposure (long + short)
    
    Returns:
        ReasonCode with stakeholder narrative
    """
    # Determine portfolio direction
    if gross_exposure < 0.1:
        direction = "flat"
    elif net_exposure > 0.3:
        direction = "risk_on"
    elif net_exposure < -0.3:
        direction = "risk_off"
    else:
        direction = "neutral"
    
    # Confidence from fidelity
    if explanation.explanation_suppressed:
        confidence = "low"
    elif explanation.fidelity.correlation >= 0.5:
        confidence = "high"
    elif explanation.fidelity.correlation >= 0.3:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Generate driver descriptions from global_drivers
    drivers = []
    for feat, contrib in explanation.global_drivers[:5]:
        # Infer high/low from global average (rough heuristic)
        desc = FEATURE_NARRATIVES.get(feat, (f"low {feat}", f"high {feat}"))
        # If contribution is positive and large, feature is likely "high"
        if contrib > 0.1:
            drivers.append(desc[1])
        else:
            drivers.append(desc[0])
    
    return ReasonCode(
        level="portfolio",
        symbol=None,
        direction=direction,
        drivers=drivers,
        confidence=confidence,
    )


def generate_symbol_reason_codes(
    explanation: DailyExplanation,
    max_symbols: int = 5,
) -> List[ReasonCode]:
    """
    Generate per-symbol reason codes from daily explanation.
    
    Args:
        explanation: DailyExplanation from RollingSurrogate
        max_symbols: Maximum number of symbol reason codes to generate
    
    Returns:
        List of ReasonCodes for top traded symbols
    """
    if explanation.explanation_suppressed:
        return []
    
    reason_codes = []
    
    for sc in explanation.symbol_contributions[:max_symbols]:
        # Determine direction from z_final
        if sc.z_final > 0.5:
            direction = "long"
        elif sc.z_final < -0.5:
            direction = "short"
        elif abs(sc.z_final) < 0.1:
            direction = "flat"
        else:
            direction = "reduced" if abs(sc.z_final) < 0.3 else "moderate"
        
        # Confidence from fidelity
        if explanation.fidelity.correlation >= 0.5:
            confidence = "high"
        elif explanation.fidelity.correlation >= 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate drivers from top contributors
        drivers = []
        
        # Add top positive contributors
        for feat, val, coef, contrib in sc.top_positive[:2]:
            desc = _describe_contribution(feat, val, contrib, coef)
            drivers.append(desc)
        
        # Add top negative contributors (as countervailing factors)
        for feat, val, coef, contrib in sc.top_negative[:1]:
            desc = _describe_contribution(feat, val, contrib, coef)
            # Negate description for negative contributions
            drivers.append(desc)
        
        reason_codes.append(ReasonCode(
            level="symbol",
            symbol=sc.symbol,
            direction=direction,
            drivers=drivers,
            confidence=confidence,
        ))
    
    return reason_codes


def format_reason_codes_summary(
    portfolio_reason: ReasonCode,
    symbol_reasons: List[ReasonCode],
) -> str:
    """
    Format reason codes into stakeholder-ready summary.
    
    Returns:
        Multi-line formatted summary for logging or display
    """
    lines = []
    lines.append("═══ Portfolio Reason Codes ═══")
    lines.append(f"  📊 {portfolio_reason.to_narrative()} [confidence: {portfolio_reason.confidence}]")
    
    if symbol_reasons:
        lines.append("")
        lines.append("  Top Symbols:")
        for rc in symbol_reasons[:5]:
            lines.append(f"    {rc.to_narrative()} [confidence: {rc.confidence}]")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Validation Companion (Logistic Ridge for sign(fwd_ret))
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationCompanionFidelity:
    """Fidelity metrics for the validation companion."""
    
    accuracy: float = 0.5  # Classification accuracy
    auc_roc: float = 0.5  # Area under ROC curve
    precision_long: float = 0.5  # Precision for long predictions
    precision_short: float = 0.5  # Precision for short predictions
    n_samples: int = 0
    is_reliable: bool = False
    
    MIN_ACCURACY: float = 0.52
    MIN_AUC: float = 0.52


class ValidationCompanion:
    """
    Logistic Ridge regression to validate z_final against forward returns.
    
    Predicts sign(fwd_ret_H) using:
    - Same features as z-explainer
    - z_final itself (as additional feature)
    
    Answers: "Under this regime/stress, does z_final historically work?"
    
    This is purely observational - it does not affect trading decisions.
    """
    
    def __init__(
        self,
        window: int = 126,
        max_window: int = 252,
        refit_interval: int = 10,
        regularization: float = 1.0,
        min_samples: int = 100,
    ):
        """
        Args:
            window: Minimum rolling window size
            max_window: Maximum window size
            refit_interval: Refit every K days
            regularization: L2 regularization strength
            min_samples: Minimum samples before fitting
        """
        self.window = window
        self.max_window = max_window
        self.refit_interval = refit_interval
        self.regularization = regularization
        self.min_samples = min_samples
        
        # Rolling buffer: (features_with_z, sign_of_fwd_ret, date, symbol)
        self._buffer: List[Tuple[np.ndarray, int, str, str]] = []
        
        # Model state
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.last_refit_idx: int = -1
        self.n_refits: int = 0
        
        # Fidelity
        self.last_fidelity: ValidationCompanionFidelity = ValidationCompanionFidelity()
        
        # Feature names (same as surrogate + z_final)
        self.feature_names: List[str] = ALL_FEATURE_NAMES + ["z_final"]
    
    def add_sample(
        self,
        features: np.ndarray,
        z_final: float,
        fwd_return: float,
        date: str,
        symbol: str,
    ) -> None:
        """
        Add a sample with known forward return.
        
        Args:
            features: Feature vector (same as z-explainer)
            z_final: The z_final value for this symbol
            fwd_return: Forward return over horizon H
            date: Date string
            symbol: Symbol string
        """
        # Target: sign of forward return
        if abs(fwd_return) < 1e-9:
            sign_ret = 0
        else:
            sign_ret = 1 if fwd_return > 0 else -1
        
        # Append z_final to features
        features_with_z = np.concatenate([features, [z_final]])
        
        if np.all(np.isfinite(features_with_z)):
            self._buffer.append((features_with_z, sign_ret, date, symbol))
        
        # Trim to max window
        max_samples = self.max_window * 50  # Approximate symbols per day
        if len(self._buffer) > max_samples:
            self._buffer = self._buffer[-max_samples:]
    
    def add_day_with_returns(
        self,
        day: ZExplainerDay,
        fwd_returns: np.ndarray,
    ) -> None:
        """
        Add all samples from a day with their forward returns.
        
        Args:
            day: ZExplainerDay with current day's data
            fwd_returns: Forward returns [n_assets] for horizon H
        """
        n_assets = len(day.symbols)
        
        global_vec = np.array([
            day.day_calib_score, day.online_trust_score, day.cboe_panic,
            day.cboe_slope, day.cboe_vrp, day.vol_scaler, day.portfolio_drawdown,
            day.portfolio_realized_vol, day.corr_hhi, day.turnover_prev,
            day.cost_prev, day.equity_level, float(day.market_regime),
        ], dtype=np.float32)
        
        for j, sym in enumerate(day.symbols):
            per_sym_vec = np.array([
                day.z_pre_overlay[j], day.mu_raw[j], day.sigma_exec[j],
                float(day.sigma_was_clipped[j]), day.risk_scale[j],
                day.regime_multiplier[j], float(day.hygiene_ok[j]),
                day.split_stress[j], day.split_discount[j],
                day.quantile_z[j], day.quantile_blend_weight,
            ], dtype=np.float32)
            
            features = np.concatenate([per_sym_vec, global_vec])
            
            if j < len(fwd_returns) and np.isfinite(fwd_returns[j]):
                self.add_sample(
                    features=features,
                    z_final=float(day.z_final[j]),
                    fwd_return=float(fwd_returns[j]),
                    date=day.date,
                    symbol=sym,
                )
    
    def should_refit(self, day_idx: int) -> bool:
        """Check if it's time to refit."""
        if len(self._buffer) < self.min_samples:
            return False
        if self.last_refit_idx < 0:
            return True
        return (day_idx - self.last_refit_idx) >= self.refit_interval
    
    def refit(self, day_idx: int = 0) -> ValidationCompanionFidelity:
        """
        Refit logistic ridge regression.
        
        Uses iteratively reweighted least squares (IRLS) approximation
        for logistic regression with L2 penalty.
        """
        if len(self._buffer) < self.min_samples:
            self.last_fidelity = ValidationCompanionFidelity(is_reliable=False)
            return self.last_fidelity
        
        # Stack data
        X = np.vstack([s[0] for s in self._buffer])
        y = np.array([s[1] for s in self._buffer], dtype=np.float32)
        
        # Filter to binary classification (exclude y=0)
        binary_mask = y != 0
        X = X[binary_mask]
        y = y[binary_mask]
        
        # Convert to 0/1 for logistic regression
        y_binary = (y > 0).astype(np.float32)
        
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples:
            self.last_fidelity = ValidationCompanionFidelity(is_reliable=False)
            return self.last_fidelity
        
        # Standardize
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0)
        self.feature_stds[self.feature_stds < 1e-12] = 1.0
        X_std = (X - self.feature_means) / self.feature_stds
        
        # Simple logistic regression via IRLS (few iterations)
        # Initialize with zeros
        beta = np.zeros(n_features)
        
        for _ in range(10):  # IRLS iterations
            # Predictions
            z = X_std @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
            
            # Weights
            w = p * (1 - p) + 1e-8
            
            # Weighted least squares with L2 penalty
            W = np.diag(w)
            XtWX = X_std.T @ W @ X_std + self.regularization * np.eye(n_features)
            XtWz = X_std.T @ (w * z + (y_binary - p))
            
            try:
                beta = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                break
        
        self.coefficients = beta
        
        # Compute predictions
        z_pred = X_std @ self.coefficients
        p_pred = 1.0 / (1.0 + np.exp(-np.clip(z_pred, -20, 20)))
        y_pred_class = (p_pred > 0.5).astype(int)
        
        # Metrics
        accuracy = float(np.mean(y_pred_class == y_binary))
        
        # AUC (simplified)
        try:
            # Sort by predicted probability
            sorted_idx = np.argsort(p_pred)
            y_sorted = y_binary[sorted_idx]
            n_pos = np.sum(y_binary)
            n_neg = len(y_binary) - n_pos
            
            if n_pos > 0 and n_neg > 0:
                # Wilcoxon-Mann-Whitney statistic
                rank_sum = np.sum(np.where(y_sorted == 1)[0])
                auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            else:
                auc = 0.5
        except Exception:
            auc = 0.5
        
        # Precision by class
        pred_long = y_pred_class == 1
        pred_short = y_pred_class == 0
        
        if np.sum(pred_long) > 0:
            precision_long = float(np.mean(y_binary[pred_long] == 1))
        else:
            precision_long = 0.5
        
        if np.sum(pred_short) > 0:
            precision_short = float(np.mean(y_binary[pred_short] == 0))
        else:
            precision_short = 0.5
        
        is_reliable = (accuracy >= ValidationCompanionFidelity.MIN_ACCURACY) or (auc >= ValidationCompanionFidelity.MIN_AUC)
        
        self.last_fidelity = ValidationCompanionFidelity(
            accuracy=accuracy,
            auc_roc=auc,
            precision_long=precision_long,
            precision_short=precision_short,
            n_samples=n_samples,
            is_reliable=is_reliable,
        )
        
        self.is_fitted = True
        self.last_refit_idx = day_idx
        self.n_refits += 1
        
        logger.info(
            "[ValidationCompanion] Refit #%d: acc=%.1f%%, AUC=%.3f, prec_L=%.1f%%, prec_S=%.1f%%, n=%d",
            self.n_refits, accuracy * 100, auc, precision_long * 100, precision_short * 100, n_samples
        )
        
        return self.last_fidelity
    
    def predict_proba(self, features: np.ndarray, z_final: float) -> float:
        """
        Predict probability that forward return is positive.
        
        Args:
            features: Feature vector (same as z-explainer, without z_final)
            z_final: The z_final value
        
        Returns:
            Probability of positive forward return
        """
        if not self.is_fitted or self.coefficients is None:
            return 0.5
        
        features_with_z = np.concatenate([features, [z_final]])
        features_std = (features_with_z - self.feature_means) / self.feature_stds
        
        z = float(features_std @ self.coefficients)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    
    def get_z_final_coefficient(self) -> float:
        """Get the coefficient for z_final (interpretability)."""
        if not self.is_fitted or self.coefficients is None:
            return 0.0
        
        # z_final is the last feature
        return float(self.coefficients[-1])
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate validation summary for logging."""
        if not self.is_fitted:
            return {"is_fitted": False}
        
        # Get top coefficients (excluding z_final)
        coef_abs = np.abs(self.coefficients)
        sorted_idx = np.argsort(coef_abs)[::-1]
        
        top_features = [
            (self.feature_names[i], float(self.coefficients[i]))
            for i in sorted_idx[:10]
        ]
        
        return {
            "is_fitted": True,
            "fidelity": {
                "accuracy": self.last_fidelity.accuracy,
                "auc_roc": self.last_fidelity.auc_roc,
                "precision_long": self.last_fidelity.precision_long,
                "precision_short": self.last_fidelity.precision_short,
                "is_reliable": self.last_fidelity.is_reliable,
                "n_samples": self.last_fidelity.n_samples,
            },
            "z_final_coefficient": self.get_z_final_coefficient(),
            "z_final_is_predictive": abs(self.get_z_final_coefficient()) > 0.1,
            "top_features": top_features,
            "n_refits": self.n_refits,
        }
    
    def format_validation_log(self) -> str:
        """Format validation companion for logging."""
        if not self.is_fitted:
            return "[ValidationCompanion] Not fitted"
        
        f = self.last_fidelity
        z_coef = self.get_z_final_coefficient()
        
        lines = [
            "═══ Validation Companion ═══",
            f"  Accuracy: {f.accuracy:.1%} | AUC: {f.auc_roc:.3f}",
            f"  Precision (long): {f.precision_long:.1%} | Precision (short): {f.precision_short:.1%}",
            f"  z_final coefficient: {z_coef:.4f} {'✓ predictive' if abs(z_coef) > 0.1 else '✗ weak'}",
            f"  Reliable: {f.is_reliable}",
        ]
        
        return "\n".join(lines)

