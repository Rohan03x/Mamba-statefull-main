#!/usr/bin/env python3
"""Standalone Stage A selector that consumes feature families via build_panel.

The script trains a tri-class multinomial logistic regression (softmax) with L1
penalty using rolling, time-aware cross-validation. Hyper-parameters are tuned
with Optuna (optionally Ray-backed). The best model's coefficients are converted
into family-level weights and persisted as ``family_weights_best.json`` so later
pipeline stages can reuse them without additional glue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from collections import Counter, defaultdict, deque
from time import perf_counter

# Load environment variables from .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

import numpy as np
import optuna
from optuna import pruners
import pandas as pd
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    import ray  # type: ignore
except Exception:  # pragma: no cover - ray optional
    ray = None  # type: ignore

# Allow running this file directly (without `-m`) by ensuring repo root is on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.aggregator_panel import build_panel, _fetch_price_data  # type: ignore
from src.features.lag_config import apply_lags  # type: ignore

try:
    # Default Stage-A pooled universe: CORE + satellite candidates.
    from src.stage_b.universe_selector import DEFAULT_CANDIDATE_UNIVERSE as _DEFAULT_SYMBOLS
except Exception:  # pragma: no cover - defensive fallback
    _DEFAULT_SYMBOLS = (
        "SPY",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "JPM",
        "XOM",
        "UNH",
        "COST",
        "GS",
        "BAC",
        "BLK",
        "PG",
        "KO",
        "JNJ",
        "WMT",
        "CVX",
        "CAT",
        "BA",
        "NFLX",
        "PYPL",
        "SHOP",
        "SQ",
        "ZM",
        "FCX",
        "COP",
        "NEM",
        "SCHW",
        "MS",
        "HD",
        "LOW",
        "DIS",
        "TSLA",
        "AMD",
    )

LOGGER = logging.getLogger("stage_a_selector")

COUNTER_COLUMN_PATTERNS: Tuple[str, ...] = (
    "_id",
    "_index",
    "_counter",
    "days_since",
    "since_",
    "_since",
)


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy and Optuna-compatible RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    try:  # torch optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


if ray is not None:  # pragma: no cover - requires ray runtime

    @ray.remote(num_cpus=1)
    def _ray_fold_runner(
        dataset_arrays: Dict[str, Any],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        params: Dict[str, Any],
        fold_id: int,
        seed: int,
    ) -> Optional[Dict[str, Any]]:
        return train_and_eval_fold(
            fold_id,
            train_idx,
            val_idx,
            dataset_arrays,
            params,
            seed,
        )
else:  # pragma: no cover - ray optional
    _ray_fold_runner = None

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "local_cache"
DEFAULT_PREP_OUTPUT_DIR = REPO_ROOT / "artifacts" / "prep_families"
PREP_FAMILIES_SCRIPT = REPO_ROOT / "tools" / "prep_families.py"
REBUILD_TRACKC_SCRIPT = REPO_ROOT / "tools" / "rebuild_trackc_from_cache.py"

STAGE_A_FAMILIES: Tuple[str, ...] = (
    "alternative_signals",
    "cboe_term",
    "correlation",
    "cross_asset",
    "dcf",
    "dividends",
    "doc_embedding_novelty_hf",
    "earnings",
    "earnings_transcript_hf",
    "fin_g2",
    "fin_g3",
    "fin_g4",
    "fin_g5",
    "fin_g6",
    "fin_g7",
    "finbert",
    "garch_iv",
    "macro_tst_hf",
    "microstructure",
    "ml_framework",
    "multiasset",
    "options",
    "options_anchoring",
    "regime",
    "short_interest",
    "subsidiary",
    "tft_features",
)

try:
    from src.stage_b.pipeline import STAGE_B_FAMILIES as _STAGE_B_FAMILIES  # type: ignore
except Exception:  # pragma: no cover - stage_b optional
    _STAGE_B_FAMILIES = tuple()

STAGE_B_FAMILIES: Tuple[str, ...] = tuple(_STAGE_B_FAMILIES)
ALL_FAMILIES: Tuple[str, ...] = STAGE_A_FAMILIES + tuple(
    fam for fam in STAGE_B_FAMILIES if fam not in STAGE_A_FAMILIES
)

FORCED_FAMILIES: Tuple[str, ...] = (
    "ml_framework",
    "microstructure",
    "regime",
    "macro_tst_hf",
    "tft_features",
)

DEFAULT_CACHE_ONLY_FAMILIES: Tuple[str, ...] = (
    "finbert",
    "doc_embedding_novelty_hf",
)

HF_PCA_PROTECTED_FAMILIES: Tuple[str, ...] = (
    "finbert",
    "macro_tst_hf",
    "doc_embedding_novelty_hf",
    "earnings_transcript_hf",
)

PCA_FEATURE_THRESHOLD = 999999  # Disabled - keep raw features
# Default compression used before the Optuna selector starts scoring families.
PCA_COMPONENTS_PER_FAMILY = 3
TOP_FEATURES_PER_FAMILY = 5

FEATURE_CONTRIB_EPS = 1e-8
FEATURE_USEFUL_THRESHOLD = 5e-4
FAMILY_SCORE_LOWER_THRESHOLD = 0.05
FAMILY_SCORE_HARD_CUT_THRESHOLD = -0.05
FAMILY_SCORE_WEIGHTS = {
    "mean": 0.40,
    "stability": 0.25,
    "useful": 0.20,
    "delta": 0.15,
}
FAMILY_NEGATIVE_PENALTY_WEIGHT = 0.05
FAMILY_SCORE_OBJECTIVE_WEIGHT = 0.10

def _build_column_family_lookup(family_columns: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for fam, cols in family_columns.items():
        for col in cols:
            lookup[col] = fam
    return lookup


def build_group_metadata(
    columns: Sequence[str],
    family_columns: Mapping[str, Sequence[str]],
    families: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, List[int]]]:
    family_lookup = _build_column_family_lookup(family_columns)
    fam_to_idx = {fam: idx for idx, fam in enumerate(families)}
    group_ids: List[int] = []
    fam_indices: Dict[str, List[int]] = {fam: [] for fam in families}
    for idx, col in enumerate(columns):
        fam = family_lookup.get(col)
        gid = fam_to_idx.get(fam, -1)
        group_ids.append(gid)
        if fam is not None and gid >= 0:
            fam_indices[fam].append(idx)
    return np.array(group_ids, dtype=int), fam_to_idx, fam_indices


class GroupLassoSoftmax:
    """Simple proximal-gradient solver for multi-class softmax with group sparsity."""

    def __init__(
        self,
        groups: Sequence[int],
        group_reg: float,
        l1_reg: float = 0.0,
        learning_rate: float = 0.1,
        max_iter: int = 200,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        self.groups = np.asarray(groups, dtype=int)
        self.group_reg = float(max(group_reg, 0.0))
        self.l1_reg = float(max(l1_reg, 0.0))
        self.learning_rate = float(max(learning_rate, 1e-6))
        self.max_iter = int(max(1, max_iter))
        self.tol = float(max(tol, 1e-8))
        self.fit_intercept = fit_intercept
        self._group_indices: Dict[int, np.ndarray] = {}

    def _prepare_groups(self) -> None:
        unique = np.unique(self.groups)
        for gid in unique:
            if gid < 0:
                continue
            self._group_indices[int(gid)] = np.where(self.groups == gid)[0]

    def _one_hot(self, y_idx: np.ndarray, n_classes: int) -> np.ndarray:
        eye = np.eye(n_classes, dtype=np.float32)
        return eye[y_idx]

    def _prox_sparse_group(self, weights: np.ndarray) -> np.ndarray:
        updated = weights
        if self.l1_reg > 0:
            updated = np.sign(updated) * np.maximum(0.0, np.abs(updated) - self.learning_rate * self.l1_reg)
        if self.group_reg > 0:
            for gid, idx in self._group_indices.items():
                if idx.size == 0:
                    continue
                block = updated[:, idx]
                norm = np.linalg.norm(block)
                if norm == 0:
                    continue
                shrink = max(0.0, 1.0 - (self.learning_rate * self.group_reg) / norm)
                updated[:, idx] = block * shrink
        return updated

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupLassoSoftmax":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        self.classes_ = np.sort(classes)
        class_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([class_index[int(label)] for label in y], dtype=int)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if self.groups.size != n_features:
            raise ValueError("Group id array must match number of features")
        self._prepare_groups()
        weights = np.zeros((n_classes, n_features), dtype=np.float32)
        intercept = np.zeros(n_classes, dtype=np.float32)
        targets = self._one_hot(y_idx, n_classes)
        last_change = np.inf
        for it in range(self.max_iter):
            logits = X @ weights.T
            if self.fit_intercept:
                logits += intercept
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-9, None)
            diff = probs - targets
            grad_w = diff.T @ X / n_samples
            grad_b = diff.mean(axis=0)
            tentative = weights - self.learning_rate * grad_w
            tentative = self._prox_sparse_group(tentative)
            new_intercept = intercept - self.learning_rate * grad_b if self.fit_intercept else intercept
            last_change = float(np.max(np.abs(tentative - weights)))
            weights = tentative
            intercept = new_intercept
            if last_change < self.tol:
                break
        self.coef_ = weights
        self.intercept_ = intercept
        self.n_iter_ = it + 1
        self.converged_ = last_change < self.tol
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.coef_.T
        if self.fit_intercept:
            logits += self.intercept_
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        denom = np.clip(exp.sum(axis=1, keepdims=True), 1e-9, None)
        return exp / denom

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


def resolve_families(requested: Optional[Sequence[str]]) -> List[str]:
    """Return canonical family list (Stage A + Stage B), optionally filtered by user input."""

    lookup = {fam.lower(): fam for fam in ALL_FAMILIES}
    if not requested:
        return list(ALL_FAMILIES)
    resolved: List[str] = []
    for raw in requested:
        key = raw.strip().lower()
        if key not in lookup:
            raise ValueError(
                f"Family '{raw}' is not allowed. Allowed: {', '.join(ALL_FAMILIES)}"
            )
        canonical = lookup[key]
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def resolve_required_cache_families(requested: Optional[Sequence[str]]) -> List[str]:
    """Normalize list of families that must come from cache when cache guard enabled."""

    if not requested:
        return list(DEFAULT_CACHE_ONLY_FAMILIES)
    lookup = {fam.lower(): fam for fam in ALL_FAMILIES}
    normalized: List[str] = []
    for raw in requested:
        if raw is None:
            continue
        key = raw.strip().lower()
        if not key or key == "none":
            continue
        if key not in lookup:
            raise ValueError(
                f"Family '{raw}' is not allowed for --require-cached-families. "
                f"Allowed: {', '.join(ALL_FAMILIES)} or 'none'"
            )
        canonical = lookup[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.Series
    forward_returns: pd.Series
    family_columns: Dict[str, List[str]]
    families: List[str]
    family_coverage: Dict[str, float]
    feature_metadata: Dict[str, Dict[str, Any]]
    family_raw_columns: Dict[str, List[str]]
    family_component_projections: Dict[str, Dict[str, Any]]


def build_dataset_arrays(dataset: Dataset) -> Dict[str, Any]:
    columns = list(dataset.features.columns)
    variances = (
        dataset.features.var(axis=0, ddof=0).fillna(0.0)
        if not dataset.features.empty
        else pd.Series(dtype=float)
    )
    column_variances = {col: float(variances.get(col, 0.0)) for col in columns}
    group_ids, family_to_group, family_column_indices = build_group_metadata(
        columns,
        dataset.family_columns,
        dataset.families,
    )
    component_counts = {fam: len(indices) for fam, indices in family_column_indices.items()}
    non_zero_counts = [(fam, count) for fam, count in component_counts.items() if count > 0]
    non_zero_counts.sort(key=lambda kv: kv[0])
    LOGGER.info(
        "📐 Family component counts (<=%d PCA cols): %s",
        PCA_COMPONENTS_PER_FAMILY,
        non_zero_counts,
    )

    feature_matrix = dataset.features.to_numpy(dtype=np.float32, copy=True)
    return {
        "features": feature_matrix,
        "labels": dataset.labels.to_numpy(),
        "returns": dataset.forward_returns.to_numpy(),
        "columns": columns,
        "family_columns": dataset.family_columns,
        "families": list(dataset.families),
        "column_variances": column_variances,
        "group_ids": group_ids,
        "family_to_group": family_to_group,
        "family_column_indices": family_column_indices,
        "family_component_counts": component_counts,
        "family_raw_columns": dataset.family_raw_columns,
        "family_component_projections": dataset.family_component_projections,
    }
def parse_args() -> argparse.Namespace:
    """Configure and parse CLI arguments for the Stage A selector."""

    today = datetime.utcnow().strftime("%Y-%m-%d")
    default_outdir = REPO_ROOT / "artifacts" / "stage_a" / "{symbol}_h{horizon}"

    parser = argparse.ArgumentParser(
        description="Stage A selector that fits logistic models and emits family weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run grid / I/O fundamentals
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Primary ticker symbol (used as run label in --universe-mode=global)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(_DEFAULT_SYMBOLS),
        help="List of symbols to run (defaults to CORE+satellite candidate universe)",
    )
    parser.add_argument("--horizon", type=int, default=63, help="Primary forecast horizon (days)")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of horizons (days) for batch runs",
    )
    parser.add_argument(
        "--outdir",
        default=str(default_outdir),
        help="Output directory (supports {symbol}/{horizon} placeholders)",
    )
    parser.add_argument(
        "--outdir-template",
        default=None,
        help="Template path used only when batching symbols/horizons",
    )

    # Selector method
    parser.add_argument(
        "--selector-method",
        dest="selector_method",
        choices=("ridge_proxy", "elasticnet_proxy", "legacy_optuna"),
        default="legacy_optuna",
        help="Stage-A learning method",
    )

    parser.add_argument(
        "--universe-mode",
        choices=("grid", "global"),
        default="global",
        help="When 'global', treat --symbols as a pooled universe and run once (using --symbol only as the run label).",
    )

    # Date and walk-forward controls
    parser.add_argument("--start", default="2005-07-02", help="Panel start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-06-20", help="Panel end date (YYYY-MM-DD)")
    parser.add_argument(
        "--wf-start",
        dest="wf_start",
        default=None,
        help="Walk-forward start override (defaults to --start)",
    )
    parser.add_argument(
        "--wf-end",
        dest="wf_end",
        default=None,
        help="Walk-forward end override (defaults to --end)",
    )
    parser.add_argument(
        "--wf-train-years",
        type=int,
        default=5,
        help="Training window (years) for walk-forward cache prep",
    )
    parser.add_argument(
        "--wf-step-years",
        type=int,
        default=1,
        help="Walk-forward step in years (when --wf-step-days unset)",
    )
    parser.add_argument(
        "--wf-step-days",
        type=int,
        default=None,
        help="Walk-forward step in days (takes precedence over years)",
    )

    # Family / cache controls
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Optional subset of Stage A + Stage B families to evaluate",
    )
    parser.add_argument(
        "--panel-cache-dir",
        default=None,
        help="Base directory for cached panel parquet files",
    )
    parser.add_argument(
        "--prep-output-dir",
        default=None,
        help="Directory to store prep_families manifests (defaults to artifacts/prep_families)",
    )
    parser.add_argument(
        "--panel-horizon",
        type=int,
        default=None,
        help="Override horizon passed to build_panel for cache lookup",
    )
    parser.add_argument(
        "--require-cached-families",
        nargs="+",
        default=list(DEFAULT_CACHE_ONLY_FAMILIES),
        help="Families that must come from cache when enforcement is on (use 'none' to disable)",
    )
    parser.add_argument(
        "--require-cached-panel",
        dest="require_cached_panel",
        action="store_true",
        help="Require cached panel artifacts before running",
    )

    # Proxy (Ridge/ElasticNet) learning controls
    # Default ON: Phase-2 can consume scheduled Stage-A weights to apply a causal,
    # slowly-evolving governance prior. Disable explicitly if you need the fastest
    # possible Stage-A run.
    parser.add_argument(
        "--proxy-time-adaptive",
        dest="proxy_time_adaptive",
        action="store_true",
        default=True,
        help="Emit a walk-forward schedule of weights (causal, piecewise-constant by update boundary)",
    )
    parser.add_argument(
        "--no-proxy-time-adaptive",
        dest="proxy_time_adaptive",
        action="store_false",
        help="Disable scheduled Stage-A weights (static weights only)",
    )
    parser.add_argument(
        "--proxy-update-every",
        type=int,
        default=21,
        help="Schedule update cadence in sessions (U)",
    )
    parser.add_argument(
        "--proxy-lookback",
        type=int,
        default=1008,
        help="Schedule lookback window in sessions (L), e.g. ~4y = 1008",
    )
    parser.add_argument(
        "--proxy-smooth-alpha",
        type=float,
        default=0.2,
        help="EWMA smoothing alpha for governance weights (0.1-0.3 recommended)",
    )
    parser.add_argument(
        "--proxy-weight-normalization",
        choices=("mean1", "sum1"),
        default="mean1",
        help="How to normalize family weights before clipping",
    )
    parser.add_argument(
        "--proxy-folds",
        type=int,
        default=5,
        help="Number of purged time-series CV folds",
    )
    parser.add_argument(
        "--proxy-gap",
        type=int,
        default=None,
        help="Purge gap in rows between train and validation (defaults to horizon)",
    )
    parser.add_argument(
        "--proxy-val-size",
        type=int,
        default=252,
        help="Validation block size per fold (rows)",
    )
    parser.add_argument(
        "--proxy-min-train",
        type=int,
        default=756,
        help="Minimum train rows required per fold",
    )
    parser.add_argument(
        "--proxy-alpha-grid",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        help="Grid of regularization strengths to try for the proxy model",
    )
    parser.add_argument(
        "--proxy-l1-ratio",
        type=float,
        default=0.10,
        help="ElasticNet l1_ratio (only used when selector-method=elasticnet_proxy)",
    )
    parser.add_argument(
        "--proxy-shrink-to-uniform",
        type=float,
        default=0.10,
        help="Blend weights toward uniform by this fraction (shrinkage)",
    )
    parser.add_argument(
        "--proxy-clip-min",
        type=float,
        default=0.25,
        help="Minimum per-family weight before renormalization (governance clip)",
    )
    parser.add_argument(
        "--proxy-clip-max",
        type=float,
        default=4.0,
        help="Maximum per-family weight before renormalization (governance clip)",
    )
    parser.add_argument(
        "--allow-panel-regen",
        dest="require_cached_panel",
        action="store_false",
        help="Permit build_panel to regenerate missing panel data",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=32,
        help="Parallel workers for prep_families cache preparation",
    )
    parser.add_argument(
        "--cache-hf-workers",
        type=int,
        default=8,
        help="Parallel workers for HF/FinBERT families when prepping cache",
    )
    parser.add_argument(
        "--skip-cache-prep",
        action="store_true",
        help="Assume cache already prepared; only validate manifest",
    )
    parser.add_argument(
        "--force-cache-refresh",
        action="store_true",
        help="Force rerunning prep_families even if manifests look valid",
    )
    parser.add_argument(
        "--auto-regen-missing-families",
        dest="auto_regen_missing_families",
        action="store_true",
        help="Automatically rerun prep_families for families dropped after filtering",
    )
    parser.add_argument(
        "--no-auto-regen-missing-families",
        dest="auto_regen_missing_families",
        action="store_false",
        help="Disable automatic prep_families reruns when families disappear",
    )
    parser.add_argument(
        "--auto-regen-max-attempts",
        type=int,
        default=2,
        help="Maximum dataset rebuild attempts triggered by auto regeneration",
    )
    parser.add_argument(
        "--prep-mode",
        choices=["stage-a", "walkforward"],
        default="stage-a",
        help="Cache prep strategy passed to prep_families",
    )
    parser.add_argument(
        "--cache-strict",
        dest="cache_strict",
        action="store_true",
        help="Run prep_families in strict mode",
    )
    parser.add_argument(
        "--no-cache-strict",
        dest="cache_strict",
        action="store_false",
        help="Allow prep_families to proceed even if non-fatal errors occur",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip run if family_weights_best.json exists and is fresh",
    )

    parser.add_argument(
        "--attach-proxy-schedule-only",
        action="store_true",
        help="Do not run Stage-A optimization; only (re)build the dataset and attach a time-adaptive proxy schedule to family_weights_best.json in --outdir.",
    )
    parser.add_argument(
        "--reuse-max-age-hours",
        type=float,
        default=24.0,
        help="Maximum age in hours for reuse when --reuse-existing is set",
    )

    # Label / threshold controls
    parser.add_argument(
        "--disable-lags",
        action="store_true",
        help="Skip applying lag_config rules (default applies registered lags)",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=0.02,
        help="Fixed upper threshold when threshold-mode=fixed",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=-0.02,
        help="Fixed lower threshold when threshold-mode=fixed",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "percentile", "adaptive"],
        default="percentile",
        help="Choose how long/short thresholds are derived",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=None,
        help="Legacy symmetric percentile applied to |returns| (0-100)",
    )
    parser.add_argument(
        "--percentile-long",
        type=float,
        default=0.65,
        help="Upper quantile (0-1) when using percentile thresholds",
    )
    parser.add_argument(
        "--percentile-short",
        type=float,
        default=0.35,
        help="Lower quantile (0-1) when using percentile thresholds",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=50,
        help="Minimum samples required per class after filtering",
    )

    # Feature filtering controls
    parser.add_argument(
        "--max-cols-per-family",
        type=int,
        default=None,
        help="Cap columns per family by variance ranking",
    )
    parser.add_argument(
        "--max-nan-ratio",
        type=float,
        default=0.4,
        help="Drop columns whose NaN ratio exceeds this threshold",
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=1e-8,
        help="Drop columns with variance <= threshold",
    )
    parser.add_argument(
        "--family-min-coverage",
        type=float,
        default=0.6,
        help="Drop families whose average column coverage is below this threshold",
    )

    # Cross-validation configuration
    parser.add_argument("--folds", type=int, default=3, help="Number of rolling CV folds")
    parser.add_argument(
        "--fold-size",
        type=int,
        default=120,
        help="Validation window length (business days) per fold",
    )
    parser.add_argument(
        "--min-train-window",
        type=int,
        default=252,
        help="Minimum training days before first validation fold",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.005,
        help="Std-dev multiplier for Gaussian noise injected into training folds",
    )

    # Search configuration
    parser.add_argument(
        "--stagea-lambda",
        type=float,
        default=0.0024,
        help="Base λ (1/C) around which trials sample (default targets ≈0.001–0.002)",
    )
    parser.add_argument(
        "--stagea-lambda-span",
        type=float,
        default=1.5,
        help="Multiplicative span for λ sampling (1.5 ⇒ ~0.0007–0.0015)",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="Parallel Optuna jobs when Ray is disabled",
    )
    parser.add_argument(
        "--ray-parallelism",
        type=str,
        default="16",
        help="Ray worker count (integer or 'auto' for cpu_count)",
    )
    parser.add_argument(
        "--fold-workers",
        type=str,
        default="1",
        help="Parallel workers per trial for fold evaluation (integer or 'auto')",
    )
    parser.add_argument(
        "--objective-weights",
        nargs=3,
        type=float,
        metavar=("W_SHARPE", "W_RWA", "W_STABILITY"),
        default=(0.6, 0.15, 0.1),
        help="Weights for the composite objective (sharpe, rwa, stability)",
    )
    parser.add_argument(
        "--family-sparsity-coef",
        type=float,
        default=0.0,
        help="Penalty multiplier encouraging fewer active families",
    )

    # Selection / weighting controls
    parser.add_argument(
        "--selection-min-weight",
        type=float,
        default=0.001,
        help="Minimum normalized weight for auto-selected families",
    )
    parser.add_argument(
        "--selection-min-coverage",
        type=float,
        default=0.6,
        help="Minimum coverage required to keep a family selected",
    )
    parser.add_argument(
        "--stability-alpha",
        type=float,
        default=1.0,
        help="Penalty multiplier applied to unstable families during selection",
    )
    parser.add_argument(
        "--weight-temp",
        type=float,
        default=2.0,
        help="Softmax temperature applied to family logits (higher spreads mass)",
    )
    parser.add_argument(
        "--normalize-family-variance",
        action="store_true",
        help="Divide family scores by mean feature variance before weighting",
    )
    parser.add_argument(
        "--group-l1-ratio",
        type=float,
        default=0.2,
        help="Fraction of stagea_lambda applied as element-wise L1 (0 disables sparse term)",
    )
    parser.add_argument(
        "--group-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for the custom group-lasso softmax solver",
    )
    parser.add_argument(
        "--group-max-iter",
        type=int,
        default=200,
        help="Maximum optimization steps for the group-lasso solver",
    )
    parser.add_argument(
        "--group-tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for the group-lasso solver",
    )
    parser.add_argument(
        "--pca-cache-dir",
        default="/tmp/autoopt_stagea_pca",
        help="Directory for caching per-family PCA projections",
    )
    parser.add_argument(
        "--no-pca-cache",
        dest="pca_cache",
        action="store_false",
        help="Disable PCA caching (enabled by default)",
    )

    # Pruner / logging controls
    parser.add_argument(
        "--pruner",
        choices=["hyperband", "none"],
        default="hyperband",
        help="Optuna pruner strategy",
    )
    parser.add_argument(
        "--pruner-max-resource",
        type=int,
        default=4,
        help="Hyperband max resource (typically number of folds)",
    )
    parser.add_argument(
        "--pruner-reduction-factor",
        type=int,
        default=3,
        help="Hyperband reduction factor controlling pruning aggressiveness",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URI for study persistence",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optional Optuna study name",
    )

    parser.set_defaults(
        auto_regen_missing_families=False,
        cache_strict=True,
        require_cached_panel=True,
        normalize_family_variance=False,
        pca_cache=True,
    )

    return parser.parse_args()


def resolve_base_cache_dir(cache_dir: Optional[str]) -> Path:
    return Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR


def resolve_output_dir(output_dir: Optional[str]) -> Path:
    return Path(output_dir).expanduser() if output_dir else DEFAULT_PREP_OUTPUT_DIR


def find_unified_panel_parquet(
    symbol: str,
    horizon: int,
    cache_dir: Optional[str],
) -> Optional[Path]:
    """Locate a unified per-symbol parquet panel (e.g. <SYM>_h<H>_trackc.parquet).

    This repo frequently stores a consolidated panel per symbol/horizon under
    cache/features. When present, it is preferred over per-family cache shards.
    """

    candidates: List[Path] = []
    if cache_dir:
        base = Path(cache_dir).expanduser()
        candidates.extend(
            [
                base / f"{symbol.upper()}_h{horizon}_trackc.parquet",
                base / f"{symbol.lower()}_h{horizon}_trackc.parquet",
            ]
        )
    default_root = REPO_ROOT / "cache" / "features"
    candidates.extend(
        [
            default_root / f"{symbol.upper()}_h{horizon}_trackc.parquet",
            default_root / f"{symbol.lower()}_h{horizon}_trackc.parquet",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_ray_parallelism(value: Any) -> int:
    """Convert ray parallelism argument to an integer worker count."""

    if isinstance(value, int):
        return max(0, value)
    if value is None:
        return 0
    text = str(value).strip().lower()
    if text == "auto":
        return max(1, os.cpu_count() or 1)
    try:
        return max(0, int(text))
    except ValueError:
        raise ValueError("--ray-parallelism must be an integer or 'auto'") from None


def resolve_fold_workers(value: Any) -> int:
    """Resolve fold worker argument supporting 'auto' shorthand."""

    if isinstance(value, int):
        return max(1, value)
    if value is None:
        return 1
    text = str(value).strip().lower()
    if text == "auto":
        return max(1, os.cpu_count() or 1)
    try:
        return max(1, int(text))
    except ValueError:
        raise ValueError("--fold-workers must be an integer or 'auto'") from None


def compute_step_suffix(step_years: Optional[int], step_days: Optional[int]) -> str:
    if step_days is not None:
        if step_days <= 0:
            raise ValueError("wf-step-days must be positive when provided")
        return f"step{step_days}d"
    if step_years is None or step_years <= 0:
        raise ValueError("wf-step-years must be positive when --wf-step-days is not provided")
    return f"step{step_years}y"


def resolve_cache_paths(
    args: argparse.Namespace,
    symbol: str,
    horizon: int,
) -> Tuple[Path, Path, Path, Path]:
    base_cache = resolve_base_cache_dir(args.panel_cache_dir)
    output_dir = resolve_output_dir(args.prep_output_dir)
    prep_mode = getattr(args, "prep_mode", "stage-a").lower()
    if prep_mode == "stage-a":
        suffix = "stagea"
    else:
        suffix = compute_step_suffix(args.wf_step_years, args.wf_step_days)
    step_cache = base_cache / f"{symbol.lower()}_h{horizon}_{suffix}"
    manifest = output_dir / f"{symbol.lower()}_h{horizon}_completeness.json"
    return base_cache, step_cache, output_dir, manifest


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - best effort logging
        LOGGER.warning("Failed to parse manifest %s: %s", path, exc)
        return None


def manifest_is_valid(
    manifest: Dict[str, Any],
    symbol: str,
    horizon: int,
    desired_start: str,
    desired_end: str,
    expected_mode: str,
    required_families: Optional[Sequence[str]] = None,
) -> bool:
    try:
        if manifest.get("symbol", "").upper() != symbol.upper():
            return False
        if int(manifest.get("horizon", -1)) != int(horizon):
            return False
        mode = str(manifest.get("mode", "walkforward")).lower()
        if mode != expected_mode.lower():
            return False
        result = manifest.get("result", {})
        if not result.get("success"):
            return False
        checksum = manifest.get("checksum_stats", {})
        if checksum.get("missing", 0) > 0 or checksum.get("errors", 0) > 0:
            return False
        coverage_start_raw = manifest.get("coverage_start")
        coverage_end_raw = manifest.get("coverage_end")
        windows = manifest.get("windows") or []
        if coverage_start_raw and coverage_end_raw:
            coverage_start = pd.to_datetime(coverage_start_raw)
            coverage_end = pd.to_datetime(coverage_end_raw)
        elif windows:
            coverage_start = min(pd.to_datetime(win["train_start"]) for win in windows)
            coverage_end = max(pd.to_datetime(win["valid_end"]) for win in windows)
        else:
            return False
        desired_start_ts = pd.to_datetime(desired_start)
        desired_end_ts = pd.to_datetime(desired_end)
        if coverage_start > desired_start_ts:
            return False
        if coverage_end < desired_end_ts:
            return False
        if required_families:
            recorded = manifest.get("families") or {}
            manifest_families = set()
            for bucket in ("base", "hf"):
                for fam in recorded.get(bucket, []) or []:
                    if isinstance(fam, str):
                        manifest_families.add(fam.strip().lower())
            missing = [fam for fam in required_families if fam and fam.lower() not in manifest_families]
            if missing:
                LOGGER.info(
                    "Manifest missing %d required families: %s",
                    len(missing),
                    ", ".join(missing),
                )
                return False
        return True
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Manifest validation error: %s", exc)
        return False


def run_prep_families(
    args: argparse.Namespace,
    symbol: str,
    horizon: int,
    base_cache_dir: Path,
    output_dir: Path,
    families_override: Optional[Sequence[str]] = None,
) -> None:
    if not PREP_FAMILIES_SCRIPT.exists():
        raise FileNotFoundError(f"prep_families script missing at {PREP_FAMILIES_SCRIPT}")
    cmd = [
        sys.executable,
        str(PREP_FAMILIES_SCRIPT),
        "--symbol",
        symbol,
        "--horizon",
        str(horizon),
        "--wf-start",
        args.wf_start,
        "--wf-end",
        args.wf_end,
        "--wf-train-years",
        str(args.wf_train_years),
    ]
    cmd.extend(["--mode", args.prep_mode])
    if args.wf_step_days is not None:
        cmd.extend(["--wf-step-days", str(args.wf_step_days)])
    else:
        cmd.extend(["--wf-step-years", str(args.wf_step_years)])
    selected_families = list(families_override) if families_override is not None else list(args.families)
    if not selected_families:
        raise ValueError("prep_families requires at least one family to run")
    families_arg = ",".join(selected_families)
    cmd.extend(["--families", families_arg])
    cmd.extend(["--workers", str(max(1, args.cache_workers))])
    hf_workers = getattr(args, "cache_hf_workers", None)
    if hf_workers is not None:
        hf_workers = max(1, hf_workers)
        cmd.extend([
            "--hf-workers",
            str(hf_workers),
        ])
    cmd.extend(["--cache-dir", str(base_cache_dir)])
    cmd.extend(["--output-dir", str(output_dir)])
    strict_flag = "yes" if args.cache_strict else "no"
    cmd.extend(["--strict", strict_flag])
    LOGGER.info(
        "🧩 Running prep_families to refresh cache (workers=%d, hf-workers=%s)",
        args.cache_workers,
        str(hf_workers) if hf_workers is not None else "auto",
    )
    LOGGER.debug("prep_families cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_rebuild_trackc_from_cache(
    *,
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    cache_dir: Optional[Path],
    families: Sequence[str],
) -> None:
    if not REBUILD_TRACKC_SCRIPT.exists():
        raise FileNotFoundError(f"rebuild_trackc_from_cache script missing at {REBUILD_TRACKC_SCRIPT}")
    fam_list = [str(f).strip() for f in families if str(f).strip()]
    if not fam_list:
        raise ValueError("rebuild_trackc_from_cache requires at least one family")
    cmd = [
        sys.executable,
        str(REBUILD_TRACKC_SCRIPT),
        "--symbols",
        symbol,
        "--horizon",
        str(int(horizon)),
        "--start",
        start,
        "--end",
        end,
        "--families",
        ",".join(fam_list),
    ]
    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    LOGGER.info("🧱 Rebuilding unified TrackC parquet for %s (h=%d)", symbol, int(horizon))
    LOGGER.debug("rebuild_trackc_from_cache cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_cache_prepared(args: argparse.Namespace) -> Path:
    unified = find_unified_panel_parquet(args.symbol, args.horizon, args.panel_cache_dir)
    if unified is not None and unified.exists() and not args.force_cache_refresh:
        LOGGER.info("✅ Using unified cached panel %s", unified)
        setattr(args, "_panel_unified_parquet", str(unified))
        return unified.parent

    base_cache_dir, cache_dir, output_dir, manifest_path = resolve_cache_paths(
        args,
        args.symbol,
        args.horizon,
    )
    desired_start, desired_end = resolve_walkforward_window(args)
    manifest = load_manifest(manifest_path)
    manifest_valid = manifest is not None and manifest_is_valid(
        manifest,
        args.symbol,
        args.horizon,
        desired_start,
        desired_end,
        args.prep_mode,
        args.families,
    )
    if args.force_cache_refresh:
        manifest_valid = False
    if not manifest_valid:
        if args.skip_cache_prep:
            raise RuntimeError(
                "Cache manifest missing or invalid but --skip-cache-prep was set"
            )
        run_prep_families(args, args.symbol, args.horizon, base_cache_dir, output_dir)
        manifest = load_manifest(manifest_path)
        manifest_valid = manifest is not None and manifest_is_valid(
            manifest,
            args.symbol,
            args.horizon,
            desired_start,
            desired_end,
            args.prep_mode,
            args.families,
        )
    if not manifest_valid:
        raise RuntimeError(
            f"prep_families manifest invalid for {args.symbol} h{args.horizon}; "
            f"expected coverage through {desired_end}"
        )
    if not cache_dir.exists():
        raise RuntimeError(f"Cache directory {cache_dir} missing after preparation")
    if not any(cache_dir.glob("*.parquet")):
        raise RuntimeError(f"Cache directory {cache_dir} is empty")
    LOGGER.info("✅ Cache ready at %s", cache_dir)
    return cache_dir


def maybe_auto_regen_missing_families(
    args: argparse.Namespace,
    error: MissingFamilyColumnsError,
) -> bool:
    # Cached-only Stage A must not call prep_families.
    if getattr(args, "require_cached_panel", False):
        return False
    if not getattr(args, "auto_regen_missing_families", True):
        return False
    attempted: Set[str] = getattr(args, "_auto_regen_attempts", set())
    pending = [fam for fam in error.families if fam not in attempted]
    if not pending:
        return False
    setattr(args, "_auto_regen_attempts", attempted | set(pending))
    base_cache_dir, _, output_dir, _ = resolve_cache_paths(args, args.symbol, args.horizon)
    reason_bits = []
    for fam in pending:
        detail = error.drop_details.get(fam)
        if detail:
            reason_bits.append(f"{fam}: {detail}")
    if reason_bits:
        LOGGER.warning(
            "♻️ Auto-regenerating families (%s) due to %s",
            ", ".join(pending),
            "; ".join(reason_bits),
        )
    else:
        LOGGER.warning(
            "♻️ Auto-regenerating families (%s) after Stage-A filter drop",
            ", ".join(pending),
        )
    run_prep_families(
        args,
        args.symbol,
        args.horizon,
        base_cache_dir,
        output_dir,
        families_override=pending,
    )
    return True


def clone_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def build_run_grid(args: argparse.Namespace) -> List[Tuple[str, int]]:
    if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
        # In global mode, --symbols is treated as the pooled universe; we run once.
        symbols = [args.symbol]
    else:
        symbols = args.symbols or [args.symbol]
    horizons = args.horizons or [args.horizon]
    combos: List[Tuple[str, int]] = []
    for symbol in symbols:
        for horizon in horizons:
            combos.append((symbol.upper(), int(horizon)))
    return combos


def normalize_datetime_index(index: pd.Index | Sequence[Any]) -> pd.DatetimeIndex:
    """Return a tz-naive DateTimeIndex for consistent joins."""

    if not isinstance(index, pd.DatetimeIndex):
        converted = pd.to_datetime(index)
    else:
        converted = pd.DatetimeIndex(index)
    if getattr(converted, "tz", None) is not None:
        converted = converted.tz_localize(None)
    return pd.DatetimeIndex(converted)


def resolve_walkforward_window(args: argparse.Namespace) -> Tuple[str, str]:
    """Return the effective walk-forward [start, end] bounds as strings."""

    start = args.wf_start or args.start
    end = args.wf_end or args.end
    if not start or not end:
        raise ValueError("walk-forward start/end must be provided")
    return str(start), str(end)


def resolve_universe_symbols(args: argparse.Namespace) -> List[str]:
    """Resolve the effective symbol universe for pooled multi-symbol mode."""

    syms = getattr(args, "symbols", None) or []
    resolved = [str(s).upper() for s in syms if str(s).strip()]
    if not resolved:
        raise ValueError("--universe-mode=global requires --symbols")
    return resolved


def _resolve_cache_dir_for_symbol(args: argparse.Namespace, symbol: str, horizon: int) -> Optional[str]:
    """Return the cache_dir to use for this symbol when loading panels."""

    # Unified parquets live under cache/features; cache_dir is optional.
    if find_unified_panel_parquet(str(symbol), int(horizon), getattr(args, "panel_cache_dir", None)) is not None:
        return getattr(args, "panel_cache_dir", None)

    # In cached-only mode, do not fall back to prep_families shards.
    if getattr(args, "require_cached_panel", False):
        default_root = Path(getattr(args, "panel_cache_dir", None) or (REPO_ROOT / "cache" / "features")).expanduser()
        expected = default_root / f"{str(symbol).upper()}_h{int(horizon)}_trackc.parquet"
        raise FileNotFoundError(
            f"Missing unified Track-C parquet for {str(symbol).upper()} h{int(horizon)} at {expected}. "
            "Refusing to run prep_families in cached-only mode."
        )

    # For prep_families shards, build_panel expects the per-symbol step cache.
    _base_cache, step_cache, _output_dir, _manifest = resolve_cache_paths(args, str(symbol), int(horizon))
    return str(step_cache)


def analyze_duplicate_columns(
    columns: Sequence[str],
    family_mapping: Optional[Dict[str, Sequence[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return metadata about duplicate column names, or None if unique."""

    counter = Counter(columns)
    dup_counts = {col: int(count) for col, count in counter.items() if count > 1}
    if not dup_counts:
        return None

    family_lookup: Dict[str, str] = {}
    if family_mapping:
        for family, cols in family_mapping.items():
            for col in cols:
                family_lookup.setdefault(col, family)

    def resolve_family(column: str) -> str:
        if family_lookup:
            fam = family_lookup.get(column)
            if fam:
                return fam
        prefix = column.split("_", 1)[0]
        return prefix

    family_summary: Dict[str, Dict[str, Any]] = {}
    for col, count in dup_counts.items():
        family = resolve_family(col)
        fam_entry = family_summary.setdefault(
            family,
            {
                "total": 0,
                "columns": [],
            },
        )
        fam_entry["total"] += count
        fam_entry["columns"].append({"name": col, "count": count})
    for fam_entry in family_summary.values():
        fam_entry["columns"].sort(key=lambda item: item["name"])
    report = {
        "duplicate_column_count": len(dup_counts),
        "duplicate_value_count": sum(dup_counts.values()),
        "columns": dict(sorted(dup_counts.items())),
        "families": dict(sorted(family_summary.items())),
    }
    return report


def dump_duplicate_reports(
    reports: List[Dict[str, Any]],
    outdir: Path,
    symbol: str,
    horizon: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "symbol": symbol,
        "horizon": horizon,
        "generated_at": timestamp,
        "stages": reports,
    }
    path = outdir / "duplicate_columns.json"
    path.write_text(json.dumps(payload, indent=2))
    LOGGER.warning("Duplicate column details dumped to %s", path)


def resolve_outdir(
    args: argparse.Namespace,
    symbol: str,
    horizon: int,
    multi_mode: bool,
) -> Path:
    template = args.outdir_template if (multi_mode and args.outdir_template) else args.outdir
    if multi_mode and template == args.outdir and "{symbol}" not in template and "{horizon}" not in template:
        raise ValueError(
            "Batch mode requires --outdir-template or placeholders {symbol}/{horizon} in --outdir"
        )
    path_str = template
    if multi_mode or ("{symbol}" in template or "{horizon}" in template):
        try:
            path_str = template.format(symbol=symbol, horizon=horizon)
        except KeyError as exc:  # pragma: no cover - invalid template
            raise ValueError(f"Invalid outdir template: missing placeholder {exc}") from exc
    return Path(path_str)


def should_skip_run(outdir: Path, reuse: bool, max_age_hours: float) -> bool:
    if not reuse:
        return False
    weights_path = outdir / "family_weights_best.json"
    if not weights_path.exists():
        return False
    age_seconds = (datetime.utcnow() - datetime.fromtimestamp(weights_path.stat().st_mtime)).total_seconds()
    age_hours = age_seconds / 3600.0
    if age_hours <= max_age_hours:
        LOGGER.info("♻️ Reusing existing Stage A weights at %s (age %.1f h)", weights_path, age_hours)
        return True
    return False


def snapshot_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "symbol": args.symbol,
        "horizon": args.horizon,
        "start": args.start,
        "end": args.end,
        "n_trials": args.n_trials,
        "folds": args.folds,
        "fold_workers": args.fold_workers,
        "model": "group_lasso_softmax",
        "objective_weights": list(args.objective_weights),
        "family_sparsity_coef": args.family_sparsity_coef,
        "threshold_mode": args.threshold_mode,
        "upper_threshold": args.upper_threshold,
        "lower_threshold": args.lower_threshold,
        "threshold_percentile": args.threshold_percentile,
        "percentile_long": args.percentile_long,
        "percentile_short": args.percentile_short,
        "selection_min_weight": args.selection_min_weight,
        "selection_min_coverage": args.selection_min_coverage,
        "normalize_family_variance": args.normalize_family_variance,
        "pruner": args.pruner,
        "pruner_max_resource": args.pruner_max_resource,
        "pruner_reduction_factor": args.pruner_reduction_factor,
        "reuse_existing": args.reuse_existing,
        "reuse_max_age_hours": args.reuse_max_age_hours,
        "feature_noise_std": args.feature_noise_std,
        "stagea_lambda": args.stagea_lambda,
        "stagea_lambda_span": args.stagea_lambda_span,
        "group_l1_ratio": args.group_l1_ratio,
        "group_learning_rate": args.group_learning_rate,
        "group_max_iter": args.group_max_iter,
        "group_tol": args.group_tol,
        "max_cols_per_family": args.max_cols_per_family,
        "family_min_coverage": args.family_min_coverage,
        "wf_start": args.wf_start,
        "wf_end": args.wf_end,
        "wf_train_years": args.wf_train_years,
        "wf_step_years": args.wf_step_years,
        "wf_step_days": args.wf_step_days,
        "cache_workers": args.cache_workers,
        "cache_hf_workers": args.cache_hf_workers,
        "cache_strict": args.cache_strict,
        "skip_cache_prep": args.skip_cache_prep,
        "force_cache_refresh": args.force_cache_refresh,
        "prep_mode": args.prep_mode,
        "ray_parallelism": args.ray_parallelism,
        "families_stage_a": list(STAGE_A_FAMILIES),
        "families_stage_b": list(STAGE_B_FAMILIES),
        "families_allowed": list(ALL_FAMILIES),
        "families": list(args.families),
        "panel_cache_dir": args.panel_cache_dir,
        "panel_unified_parquet": getattr(args, "_panel_unified_parquet", None),
        "panel_horizon": args.panel_horizon or args.horizon,
        "require_cached_panel": args.require_cached_panel,
        "require_cached_families": list(args.require_cached_families),
        "stage": "A",
    }


def build_runtime_payload(args: argparse.Namespace) -> Dict[str, Any]:
    w_sharpe, w_rwa, w_stability = args.objective_weights
    return {
        "w_rwa": w_rwa,
        "w_sharpe": w_sharpe,
        "w_stability": w_stability,
        "family_sparsity_coef": args.family_sparsity_coef,
        "normalize_family_variance": args.normalize_family_variance,
        "fold_stability_baseline": 0.0,
    }


def resolve_pruner(args: argparse.Namespace) -> Optional[optuna.pruners.BasePruner]:
    if args.pruner == "none":
        return None
    max_resource = args.pruner_max_resource or args.folds
    reduction = max(2, args.pruner_reduction_factor)
    return pruners.HyperbandPruner(max_resource=max_resource, reduction_factor=reduction)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _ensure_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def load_panel_df(
    symbol: str,
    start: str,
    end: str,
    families: Sequence[str],
    horizon: int,
    cache_dir: Optional[str],
    panel_horizon: Optional[int],
    require_cached_panel: bool,
    required_cached_families: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    LOGGER.info("📦 Loading feature panel for %s (%s → %s)", symbol, start, end)
    effective_horizon = panel_horizon or horizon
    unified = find_unified_panel_parquet(symbol, effective_horizon, cache_dir)
    if unified is not None and unified.exists():
        LOGGER.info("📦 Loading unified parquet panel %s", unified)
        df = pd.read_parquet(unified)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = normalize_datetime_index(df.index)
        df = df.sort_index()
        df = df.loc[pd.to_datetime(start) : pd.to_datetime(end)]
        telemetry: Dict[str, Dict[str, Any]] = {}
        for fam in families:
            prefix = f"{fam}_"
            if any(str(c).startswith(prefix) for c in df.columns):
                telemetry[fam] = {
                    "status": "cached",
                    "source": "cached_unified_parquet",
                    "path": str(unified),
                }
        df.attrs.setdefault("telemetry", telemetry)
        df.attrs.setdefault("families", list(families))
        df.attrs["panel_source"] = "unified_parquet"
        df.attrs["panel_path"] = str(unified)
    else:
        if require_cached_panel:
            expected_root = Path(cache_dir).expanduser() if cache_dir else (REPO_ROOT / "cache" / "features")
            expected = expected_root / f"{symbol.upper()}_h{int(effective_horizon)}_trackc.parquet"
            raise FileNotFoundError(
                "Unified Track-C parquet panel missing for cached-only Stage A. "
                f"Expected {expected}. "
                "If you want to regenerate panels, pass --allow-panel-regen (may trigger slow data generation)."
            )
        df = build_panel(
            symbol=symbol.upper(),
            start=start,
            end=end,
            families=list(families),
            cache_dir=Path(cache_dir) if cache_dir else None,
            horizon=effective_horizon,
            stage="A",
        )
    if df is None or df.empty:
        raise RuntimeError("build_panel returned empty dataframe")
    enforce_families = required_cached_families if require_cached_panel else None
    verify_cached_panel_sources(df, families, enforce_families)
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.ffill().bfill()
    numeric.index = normalize_datetime_index(numeric.index)
    numeric = numeric.sort_index()
    return numeric


def verify_cached_panel_sources(
    panel: pd.DataFrame,
    families: Sequence[str],
    required_families: Optional[Sequence[str]],
) -> None:
    if not required_families:
        return
    # If we're loading from a single consolidated cached parquet, treat it as cached.
    # The column-prefix ↔ family-name mapping is not guaranteed to be 1:1 for all
    # families, so telemetry can be incomplete even though the source is cached.
    if str(panel.attrs.get("panel_source", "")).lower().strip() == "unified_parquet":
        return
    telemetry = panel.attrs.get("telemetry") or {}
    telemetry_lower = {str(name).lower(): info for name, info in telemetry.items()}
    families_lower = {fam.lower(): fam for fam in families}
    targets = [fam for fam in required_families if fam.lower() in families_lower]
    if not targets:
        return
    missing: List[str] = []
    uncached: List[str] = []
    for fam in targets:
        key = fam.lower()
        info = telemetry_lower.get(key)
        if not info:
            missing.append(fam)
            continue
        source = str(info.get("source", "")).lower()
        if "cached" not in source:
            uncached.append(fam)
    if not missing and not uncached:
        return
    details = []
    if missing:
        details.append(f"missing telemetry={','.join(sorted(missing))}")
    if uncached:
        details.append(f"live_sources={','.join(sorted(uncached))}")
    hint = "Run prep_families for cache-only families or pass --allow-live-panel to override."
    raise RuntimeError(
        "Panel cache requirement failed: " + "; ".join(details) + f" {hint}"
    )


def fetch_prices(symbol: str, start: str, end: str) -> pd.Series:
    LOGGER.info("💾 Fetching price history for %s", symbol)
    prices = _fetch_price_data(symbol, start, end)
    if prices is None or prices.empty:
        raise RuntimeError("Failed to fetch price data")
    if "close" not in prices.columns:
        raise RuntimeError("Price data missing 'close' column")
    series = prices["close"].copy()
    series.index = normalize_datetime_index(series.index)
    series = series.sort_index().ffill().bfill()
    return series


def summarize_panel(panel: pd.DataFrame, families: Sequence[str]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for fam in families:
        cols = [c for c in panel.columns if c.startswith(f"{fam}_")]
        if not cols:
            summary[fam] = {"columns": 0, "usable": 0, "nan_ratio": 1.0}
            continue
        fam_df = panel[cols]
        nan_ratio = float(fam_df.isna().mean().mean())
        variances = fam_df.var().fillna(0.0)
        usable = int((variances > 0).sum())
        summary[fam] = {"columns": len(cols), "usable": usable, "nan_ratio": nan_ratio}
    return summary


def _panel_family_columns(panel: pd.DataFrame, family: str) -> List[str]:
    prefix = f"{family}_"
    return [col for col in panel.columns if str(col).startswith(prefix)]


def infer_families_from_panel(panel: pd.DataFrame) -> List[str]:
    """Infer family names from column prefixes in a unified Track-C panel.

    Unified panels may not contain the full Stage-A family set; in cached-only
    mode we treat the available column prefixes as the effective families.
    """

    if panel is None or panel.empty:
        return []
    prefixes: Set[str] = set()
    for col in panel.columns:
        s = str(col)
        if s == "date":
            continue
        if "_" not in s:
            continue
        prefixes.add(s.split("_", 1)[0])
    return sorted(prefixes)


def resolve_effective_families_for_panel(
    panel: pd.DataFrame,
    requested_families: Sequence[str],
    *,
    cached_only: bool,
) -> List[str]:
    """Resolve the family list to use for filtering/lagging/PCA.

    - In cached-only mode, fall back to inferred families when requested ones
      aren't present in the unified parquet.
    - In regen-allowed mode, keep the requested family list unchanged.
    """

    requested = [str(f).strip() for f in (requested_families or []) if str(f).strip()]
    if not cached_only:
        return requested

    inferred = infer_families_from_panel(panel)
    if not requested:
        return inferred

    present = [fam for fam in requested if _panel_family_columns(panel, fam)]
    if present:
        missing = [fam for fam in requested if fam not in present]
        if missing:
            LOGGER.warning(
                "⚠️ Cached-only mode: %d requested families missing in unified panel; using %d present families",
                len(missing),
                len(present),
            )
        return present

    if inferred:
        LOGGER.warning(
            "⚠️ Cached-only mode: none of the requested families exist in unified panel; falling back to inferred families: %s",
            ", ".join(inferred),
        )
    return inferred


def ensure_panel_family_coverage(
    panel: pd.DataFrame,
    families: Sequence[str],
    stage_label: str,
) -> None:
    missing = [fam for fam in families if not _panel_family_columns(panel, fam)]
    if missing:
        raise MissingFamilyColumnsError(stage_label, missing)


class MissingFamilyColumnsError(RuntimeError):
    def __init__(
        self,
        stage_label: str,
        families: Sequence[str],
        drop_details: Optional[Dict[str, str]] = None,
    ) -> None:
        message = (
            f"{stage_label} missing derived columns for families: {', '.join(sorted(families))}. "
            "Inspect Stage-A cache outputs or adjust lag_config thresholds."
        )
        super().__init__(message)
        self.stage_label = stage_label
        self.families = list(families)
        self.drop_details = drop_details or {}


def ensure_mapping_family_coverage(
    mapping: Dict[str, List[str]],
    families: Sequence[str],
    stage_label: str,
    drop_summary: Optional[Dict[str, str]] = None,
) -> None:
    missing = [fam for fam in families if not mapping.get(fam)]
    if missing:
        summary = drop_summary or {}
        raise MissingFamilyColumnsError(stage_label, missing, summary)


def prune_missing_families_or_raise(
    mapping: Mapping[str, Sequence[str]],
    families: Sequence[str],
    *,
    stage_label: str,
    drop_summary: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[str]]:
    """Drop families with zero columns, warn, and return the cleaned mapping.

    This is primarily for pooled/global mode where some families can legitimately
    be filtered out (e.g., constant columns). We still raise if nothing remains.
    """

    cleaned: Dict[str, List[str]] = {}
    requested = [str(f).strip() for f in families if str(f).strip()]
    for fam in requested:
        cols = list(mapping.get(fam, []) or [])
        if cols:
            cleaned[fam] = cols

    missing = [fam for fam in requested if fam not in cleaned]
    if missing:
        detail = drop_summary or {}
        reason_bits = [f"{fam}: {detail[fam]}" for fam in missing if fam in detail]
        if reason_bits:
            LOGGER.warning("⚠️ %s dropped families: %s", stage_label, "; ".join(reason_bits))
        else:
            LOGGER.warning("⚠️ %s dropped families: %s", stage_label, ", ".join(missing))

    if not cleaned:
        raise RuntimeError(f"{stage_label}: no usable feature families remained")
    return cleaned


def is_counter_like_column(name: str) -> bool:
    lower = name.lower()
    if any(pattern in lower for pattern in COUNTER_COLUMN_PATTERNS):
        return True
    if "since" in lower:
        return True
    if lower.endswith(("_idx", "_seq")):
        return True
    return False


def filter_columns(
    panel: pd.DataFrame,
    families: Sequence[str],
    max_nan_ratio: float,
    min_variance: float,
    max_cols_per_family: Optional[int],
    min_family_coverage: float,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, float], Dict[str, str]]:
    kept_columns: List[str] = []
    family_cols: Dict[str, List[str]] = {}
    family_coverage: Dict[str, float] = {}
    family_drop_summary: Dict[str, str] = {}
    column_drop_reasons: Dict[str, List[str]] = defaultdict(list)
    for fam in families:
        prefix = f"{fam}_"
        cols = [c for c in panel.columns if c.startswith(prefix)]
        if not cols:
            continue
        fam_df = panel[cols]
        valid_cols = []
        for col in cols:
            def note(reason: str) -> None:
                column_drop_reasons[fam].append(f"{col}:{reason}")

            series = fam_df[col]
            if is_counter_like_column(col):
                LOGGER.debug("🚫 Dropping counter/index column %s", col)
                note("counter_like")
                continue
            if float(series.isna().mean()) > max_nan_ratio:
                note("high_nan")
                continue
            variance = float(series.var(ddof=0))
            if variance <= min_variance:
                note("low_variance")
                continue
            unique_count = int(series.nunique(dropna=True))
            if unique_count <= 1:
                note("low_unique")
                continue
            top_fraction = float(series.value_counts(dropna=True, normalize=True).max()) if unique_count > 0 else 1.0
            if top_fraction >= 0.995:
                LOGGER.debug("🚫 Dropping near-constant column %s (%.2f%%)", col, top_fraction * 100)
                note("near_constant")
                continue
            if series.is_monotonic_increasing or series.is_monotonic_decreasing:
                LOGGER.debug("🚫 Dropping monotonic column %s", col)
                note("monotonic")
                continue
            valid_cols.append(col)
        if not valid_cols:
            if fam in FORCED_FAMILIES and cols:
                LOGGER.warning("⚠️ Keeping forced family %s with fallback column %s", fam, cols[0])
                valid_cols = [cols[0]]
            else:
                reasons = column_drop_reasons.get(fam, [])
                snippet = "; ".join(reasons[:5])
                if reasons and len(reasons) > 5:
                    snippet += "; …"
                detail = f"all columns filtered"
                if snippet:
                    detail = f"{detail}: {snippet}"
                family_drop_summary[fam] = detail
                continue
        hard_limit = max_cols_per_family
        if hard_limit and len(valid_cols) > hard_limit:
            variances = fam_df[valid_cols].var(ddof=0).fillna(0.0).sort_values(ascending=False)
            LOGGER.warning(
                "⚠️ Hard-capping family %s columns from %d → %d",
                fam,
                len(valid_cols),
                hard_limit,
            )
            valid_cols = list(variances.head(hard_limit).index)
        fam_cov = 1.0 - float(fam_df[valid_cols].isna().mean().mean())
        if fam_cov < min_family_coverage and fam not in FORCED_FAMILIES:
            LOGGER.info("🚫 Dropping family %s for low coverage (%.2f)", fam, fam_cov)
            family_drop_summary[fam] = f"coverage {fam_cov:.2f} < {min_family_coverage:.2f}"
            continue
        if fam_cov < min_family_coverage and fam in FORCED_FAMILIES:
            LOGGER.warning(
                "⚠️ Keeping forced family %s despite coverage %.2f < %.2f",
                fam,
                fam_cov,
                min_family_coverage,
            )
        family_cols[fam] = valid_cols
        family_coverage[fam] = fam_cov
        kept_columns.extend(valid_cols)
    forced_scope = [fam for fam in FORCED_FAMILIES if fam in families]
    missing_forced = [fam for fam in forced_scope if fam not in family_cols]
    if missing_forced:
        raise RuntimeError(
            "Forced families missing feature columns: " + ", ".join(sorted(missing_forced))
        )
    filtered = panel[kept_columns].copy()
    filtered = filtered.replace([np.inf, -np.inf], np.nan)
    filtered = filtered.ffill().bfill()
    filtered = filtered.dropna(how="all")
    return filtered, family_cols, family_coverage, family_drop_summary


def _deduplicate_column_names(columns: Sequence[str], suffix_template: str = "__dup{}") -> Tuple[List[str], int]:
    """Append deterministic suffixes to keep column names unique."""

    seen: Dict[str, int] = {}
    deduped: List[str] = []
    renamed = 0
    for col in columns:
        count = seen.get(col, 0)
        if count == 0:
            deduped.append(col)
        else:
            suffix = count + 1
            candidate = f"{col}{suffix_template.format(suffix)}"
            while candidate in seen:
                suffix += 1
                candidate = f"{col}{suffix_template.format(suffix)}"
            deduped.append(candidate)
            renamed += 1
            seen[candidate] = 1
        seen[col] = count + 1
    return deduped, renamed


def enforce_unique_family_columns(
    frame: pd.DataFrame,
    family_columns: Dict[str, List[str]],
    stage: str,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], int]:
    """Ensure DataFrame columns and mapping remain unique per stage."""

    if frame is None or frame.empty or not frame.columns.duplicated().any():
        return frame, family_columns, 0
    original_cols = list(frame.columns)
    deduped_cols, renamed_count = _deduplicate_column_names(original_cols)
    if renamed_count == 0:
        return frame, family_columns, 0
    column_map: Dict[str, deque[str]] = defaultdict(deque)
    for old, new in zip(original_cols, deduped_cols):
        column_map[old].append(new)
    updated_mapping: Dict[str, List[str]] = {}
    for fam, cols in family_columns.items():
        updated: List[str] = []
        for col in cols:
            queue = column_map[col]
            if queue:
                updated.append(queue.popleft())
            else:
                updated.append(col)
        updated_mapping[fam] = updated
    renamed_frame = frame.copy()
    renamed_frame.columns = deduped_cols
    LOGGER.warning(
        "🩹 Deduplicated %d column names during %s stage via __dup suffix",
        renamed_count,
        stage,
    )
    return renamed_frame, updated_mapping, renamed_count


def apply_family_lags(
    panel: pd.DataFrame,
    family_columns: Dict[str, List[str]],
    disable_lags: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Expand each family's columns with lag_config rules."""

    if disable_lags or not family_columns:
        return panel, family_columns

    lagged_blocks: List[pd.DataFrame] = []
    updated_mapping: Dict[str, List[str]] = {}

    for fam, cols in family_columns.items():
        if not cols:
            continue
        fam_frame = panel[cols]
        fam_with_lags = apply_lags(fam_frame, fam)
        if fam_with_lags is None or fam_with_lags.empty:
            continue
        lagged_blocks.append(fam_with_lags)
        updated_mapping[fam] = list(fam_with_lags.columns)

    if not lagged_blocks:
        return panel, family_columns

    expanded = pd.concat(lagged_blocks, axis=1)
    expanded = expanded.replace([np.inf, -np.inf], np.nan)
    expanded = expanded.ffill().bfill().dropna(how="all")

    for fam, cols in family_columns.items():
        updated_mapping.setdefault(fam, cols)

    return expanded, updated_mapping


def _build_pca_cache_key(
    family: str,
    columns: Sequence[str],
    standardized: pd.DataFrame,
    n_components: int,
) -> str:
    arr = standardized.to_numpy(dtype=np.float32, copy=True)
    hasher = hashlib.sha1()
    hasher.update(family.lower().encode("utf-8"))
    hasher.update("|".join(columns).encode("utf-8"))
    hasher.update(str(n_components).encode("utf-8"))
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def apply_conditional_family_pca(
    panel: pd.DataFrame,
    family_columns: Dict[str, List[str]],
    threshold: int = PCA_FEATURE_THRESHOLD,
    protected_families: Sequence[str] = HF_PCA_PROTECTED_FAMILIES,
    cache_enabled: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str], Dict[str, Dict[str, Any]]]:
    """Reduce dimensionality for high-width families while skipping HF embeddings."""

    if panel is None or panel.empty or not family_columns or threshold <= 0:
        return panel, family_columns, [], {}

    protected = {fam.lower() for fam in protected_families}
    applied: List[str] = []
    reduced_blocks: List[pd.DataFrame] = []
    updated_mapping: Dict[str, List[str]] = {}
    projection_details: Dict[str, Dict[str, Any]] = {}
    cache_root: Optional[Path] = None
    if cache_enabled and cache_dir:
        cache_root = Path(cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)

    for family, columns in family_columns.items():
        if not columns:
            continue
        fam_frame = panel[columns]
        total_cols = len(columns)
        needs_pca = total_cols > threshold and family.lower() not in protected
        if not needs_pca:
            reduced_blocks.append(fam_frame)
            updated_mapping[family] = list(columns)
            continue

        target_components = max(1, PCA_COMPONENTS_PER_FAMILY)
        n_components = min(target_components, total_cols)
        if n_components <= 0:
            reduced_blocks.append(fam_frame)
            updated_mapping[family] = list(columns)
            continue

        fam_values = fam_frame.copy()
        fam_values = fam_values.replace([np.inf, -np.inf], np.nan)
        fam_values = fam_values.ffill().bfill()
        column_means = fam_values.mean()
        fam_values = fam_values.fillna(column_means)
        fam_values = fam_values.fillna(0.0)
        centered = fam_values - column_means.fillna(0.0)
        col_std = fam_values.std(ddof=0).replace(0.0, 1.0)
        standardized = centered / col_std
        cache_path: Optional[Path] = None
        if cache_root is not None:
            cache_key = _build_pca_cache_key(family, columns, standardized, n_components)
            cache_path = cache_root / f"{family.lower()}_{cache_key}.npz"
            if cache_path.exists():
                try:
                    with np.load(str(cache_path), allow_pickle=False) as cached:
                        comps = cached["components"]
                        cached_cols = cached["columns"].astype(str).tolist()
                        loadings = cached["loadings"] if "loadings" in cached else None
                        raw_cols = cached["raw_columns"].astype(str).tolist() if "raw_columns" in cached else None
                    if comps.shape == (len(fam_frame), n_components) and loadings is not None and raw_cols is not None:
                        reduced_df = pd.DataFrame(comps, index=fam_frame.index, columns=cached_cols)
                        reduced_blocks.append(reduced_df)
                        updated_mapping[family] = list(reduced_df.columns)
                        applied.append(family)
                        projection_details[family] = {
                            "raw_columns": raw_cols,
                            "component_columns": list(reduced_df.columns),
                            "loadings": loadings,
                        }
                        LOGGER.info(
                            "💾 Loaded PCA cache for %s: %d raw+lag cols → %d components",
                            family,
                            total_cols,
                            n_components,
                        )
                        continue
                    LOGGER.warning(
                        "⚠️ PCA cache mismatch for %s (expected %s, found %s) — recomputing",
                        family,
                        (len(fam_frame), n_components),
                        comps.shape,
                    )
                except Exception as exc:
                    LOGGER.warning("⚠️ Failed to load PCA cache for %s: %s", family, exc)

        pca_seed = (hash((family.lower(), len(columns))) & 0xFFFFFFFF) or 1
        pca = PCA(n_components=n_components, random_state=pca_seed)
        try:
            transformed = pca.fit_transform(standardized.values)
        except Exception as exc:
            LOGGER.warning(
                "⚠️ PCA failed for family %s (%d cols) — keeping originals: %s",
                family,
                len(columns),
                exc,
            )
            reduced_blocks.append(fam_frame)
            updated_mapping[family] = list(columns)
            continue

        component_cols = [f"{family.lower()}_pca_{idx+1}" for idx in range(n_components)]
        reduced_df = pd.DataFrame(transformed, index=fam_frame.index, columns=component_cols)
        projection_details[family] = {
            "raw_columns": list(columns),
            "component_columns": list(component_cols),
            "loadings": pca.components_.astype(np.float32, copy=True),
        }
        if cache_path is not None:
            try:
                np.savez_compressed(
                    str(cache_path),
                    components=reduced_df.to_numpy(dtype=np.float32, copy=True),
                    columns=np.asarray(component_cols, dtype="U64"),
                    loadings=pca.components_.astype(np.float32, copy=False),
                    raw_columns=np.asarray(columns, dtype="U64"),
                )
                LOGGER.debug("💾 Cached PCA projection for %s at %s", family, cache_path)
            except Exception as exc:
                LOGGER.warning("⚠️ Failed to cache PCA result for %s: %s", family, exc)
        reduced_blocks.append(reduced_df)
        updated_mapping[family] = list(reduced_df.columns)
        applied.append(family)
        LOGGER.info(
            "📉 Applied PCA to %s: %d raw+lag cols → %d components",
            family,
            total_cols,
            n_components,
        )

    if not applied:
        return panel, family_columns, [], {}

    combined = pd.concat(reduced_blocks, axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.ffill().bfill()
    return combined, updated_mapping, applied, projection_details


def build_feature_metadata(
    family_columns: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for family, columns in family_columns.items():
        prefix = f"{family.lower()}_"
        for column in columns:
            base = column
            lower = column.lower()
            if lower.startswith(prefix):
                base = column[len(prefix):]
            lag_value: Optional[int] = None
            base_clean = base
            if "_lag" in base:
                head, tail = base.rsplit("_lag", 1)
                if tail.isdigit():
                    lag_value = int(tail)
                    base_clean = head
            metadata[column] = {
                "family": family,
                "column": column,
                "base_feature": base_clean,
                "lag": lag_value,
                "is_lag": lag_value is not None,
            }
    return metadata


def build_family_component_projections(
    final_family_columns: Mapping[str, Sequence[str]],
    raw_family_columns: Mapping[str, Sequence[str]],
    pca_projection_details: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    projections: Dict[str, Dict[str, Any]] = {}
    for family, final_cols in final_family_columns.items():
        raw_cols = list(raw_family_columns.get(family, final_cols))
        detail = pca_projection_details.get(family)
        component_weights: Dict[str, np.ndarray] = {}
        if detail:
            loadings = np.asarray(detail.get("loadings", []), dtype=np.float32)
            component_names = detail.get("component_columns", final_cols)
            raw_reference = detail.get("raw_columns", raw_cols)
            if loadings.size == 0 or loadings.shape[0] != len(component_names):
                LOGGER.warning(
                    "⚠️ PCA detail mismatch for %s — expected %d rows, got %s",
                    family,
                    len(component_names),
                    loadings.shape,
                )
                loadings = np.eye(len(final_cols), len(raw_cols), dtype=np.float32)
                component_names = list(final_cols)
            if len(raw_reference) != loadings.shape[1]:
                LOGGER.warning(
                    "⚠️ PCA detail raw column mismatch for %s — expected %d cols, got %d",
                    family,
                    loadings.shape[1],
                    len(raw_reference),
                )
                raw_reference = raw_cols
            raw_cols = list(raw_reference)
            for idx, comp_name in enumerate(component_names):
                weights = np.abs(loadings[idx]).astype(np.float32, copy=False)
                denom = float(weights.sum())
                if denom <= 0:
                    weights = np.zeros_like(weights)
                else:
                    weights = weights / denom
                component_weights[comp_name] = weights
        else:
            for idx, comp_name in enumerate(final_cols):
                weights = np.zeros(len(raw_cols), dtype=np.float32)
                if idx < len(raw_cols):
                    weights[idx] = 1.0
                component_weights[comp_name] = weights
        projections[family] = {
            "raw_columns": raw_cols,
            "component_weights": component_weights,
        }
    return projections


def summarize_top_features_by_family(
    feature_metadata: Mapping[str, Dict[str, Any]],
    average_feature_scores: Mapping[str, float],
    normalized_feature_scores: Mapping[str, float],
    limit: int = TOP_FEATURES_PER_FAMILY,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group per-feature scores by family and keep the strongest entries."""

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    score_columns = set(average_feature_scores.keys()) | set(normalized_feature_scores.keys())
    for column in score_columns:
        meta = feature_metadata.get(column, {})
        family = meta.get("family")
        if not family:
            if "_" in column:
                family = column.split("_", 1)[0]
            else:
                continue
        entry = {
            "column": column,
            "base_feature": meta.get("base_feature", column),
            "lag": meta.get("lag"),
            "is_lag": bool(meta.get("is_lag")),
            "score": float(average_feature_scores.get(column, 0.0)),
            "normalized_score": float(normalized_feature_scores.get(column, 0.0)),
        }
        grouped[family].append(entry)

    summary: Dict[str, List[Dict[str, Any]]] = {}
    for family, entries in grouped.items():
        entries.sort(key=lambda item: (item["score"], item["normalized_score"]), reverse=True)
        summary[family] = entries if limit <= 0 else entries[:limit]
    return summary


def compute_forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    aligned = prices.sort_index()
    forward = aligned.shift(-horizon) / aligned - 1.0
    return forward


def _purged_time_series_splits(
    *,
    n_samples: int,
    n_splits: int,
    val_size: int,
    gap: int,
    min_train: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits with a purge gap.

    Fold i uses:
      train = [0, val_start-gap)
      val   = [val_start, val_end)
    """

    n_splits = int(max(1, n_splits))
    val_size = int(max(1, val_size))
    gap = int(max(0, gap))
    min_train = int(max(1, min_train))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    end = int(n_samples)
    start = int(min_train + gap)
    while start + val_size <= end and len(splits) < n_splits:
        val_start = int(start)
        val_end = int(start + val_size)
        train_end = int(max(0, val_start - gap))
        if train_end < min_train:
            start += val_size
            continue
        train_idx = np.arange(train_end, dtype=int)
        val_idx = np.arange(val_start, val_end, dtype=int)
        splits.append((train_idx, val_idx))
        start += val_size

    if not splits:
        raise ValueError(
            f"Not enough samples for purged CV: n={n_samples} min_train={min_train} gap={gap} val_size={val_size}"
        )
    return splits


def _infer_family_for_trackc_column(col: str, families: Sequence[str]) -> Optional[str]:
    s = str(col)
    if "_enc_" in s:
        return s.split("_enc_")[0]
    for fam in families:
        fam = str(fam)
        if s.startswith(fam + "_"):
            return fam
    return None


def _build_trackc_family_mapping(
    *,
    columns: Sequence[str],
    families: Sequence[str],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in columns:
        fam = _infer_family_for_trackc_column(str(c), families)
        if fam:
            out[str(c)] = str(fam)
    return out


def _coef_norms_by_family(
    *,
    coef: np.ndarray,
    columns: Sequence[str],
    col_to_family: Mapping[str, str],
    families: Sequence[str],
) -> Dict[str, float]:
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    if len(columns) != int(coef.shape[0]):
        raise ValueError(f"coef/columns mismatch: coef={coef.shape} cols={len(columns)}")

    acc: Dict[str, float] = {str(f): 0.0 for f in families}
    for j, c in enumerate(columns):
        fam = col_to_family.get(str(c))
        if not fam:
            continue
        acc[str(fam)] = float(acc.get(str(fam), 0.0) + float(coef[j]) ** 2)
    for fam, v in list(acc.items()):
        acc[fam] = float(np.sqrt(max(0.0, float(v))))
    return acc


def _shrink_clip_renorm(
    *,
    raw: Mapping[str, float],
    families: Sequence[str],
    shrink_to_uniform: float,
    clip_min: float,
    clip_max: float,
) -> Dict[str, float]:
    fams = [str(f) for f in families]
    vals = np.array([float(max(0.0, raw.get(f, 0.0))) for f in fams], dtype=np.float64)
    if not np.isfinite(vals).all():
        vals = np.where(np.isfinite(vals), vals, 0.0)

    if float(vals.sum()) <= 0:
        vals = np.ones_like(vals)

    vals = vals / float(vals.sum())

    shrink = float(np.clip(shrink_to_uniform, 0.0, 1.0))
    if shrink > 0:
        uni = np.ones_like(vals) / float(len(vals))
        vals = (1.0 - shrink) * vals + shrink * uni

    vals = np.clip(vals, float(max(0.0, clip_min)), float(max(0.0, clip_max)))
    s = float(vals.sum())
    if s <= 0 or not np.isfinite(s):
        vals = np.ones_like(vals) / float(len(vals))
    else:
        vals = vals / s

    return {f: float(vals[i]) for i, f in enumerate(fams)}


def _normalize_clip_weights(
    *,
    raw: Mapping[str, float],
    families: Sequence[str],
    shrink_to_uniform: float,
    clip_min: float,
    clip_max: float,
    normalization: str,
) -> Dict[str, float]:
    """Convert raw norms into governance weights.

    - If normalization=='mean1': scale weights so mean(weight)=1.
    - If normalization=='sum1': scale weights so sum(weight)=1 (legacy).
    Then shrink toward uniform, clip, and renormalize in the same space.
    """

    fams = [str(f) for f in families]
    vals = np.array([float(max(0.0, raw.get(f, 0.0))) for f in fams], dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    if float(vals.sum()) <= 0:
        vals = np.ones_like(vals)

    norm = str(normalization or "mean1").lower().strip()
    if norm == "sum1":
        vals = vals / float(max(1e-12, vals.sum()))
    else:
        mean = float(np.mean(vals))
        if not np.isfinite(mean) or mean <= 0:
            mean = 1.0
        vals = vals / mean

    shrink = float(np.clip(shrink_to_uniform, 0.0, 1.0))
    if shrink > 0:
        if norm == "sum1":
            uni = np.ones_like(vals) / float(len(vals))
        else:
            uni = np.ones_like(vals)
        vals = (1.0 - shrink) * vals + shrink * uni

    vals = np.clip(vals, float(max(0.0, clip_min)), float(max(clip_min, clip_max)))

    # Renormalize in the same space.
    if norm == "sum1":
        s = float(vals.sum())
        if not np.isfinite(s) or s <= 0:
            vals = np.ones_like(vals) / float(len(vals))
        else:
            vals = vals / s
    else:
        mean = float(np.mean(vals))
        if not np.isfinite(mean) or mean <= 0:
            vals = np.ones_like(vals)
        else:
            vals = vals / mean

    return {f: float(vals[i]) for i, f in enumerate(fams)}


def _resolve_proxy_universe(args: argparse.Namespace) -> List[str]:
    if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
        syms = getattr(args, "symbols", None)
        if not syms:
            raise ValueError("--universe-mode=global requires --symbols")
        return [str(s).upper() for s in syms]
    return [str(getattr(args, "symbol", "AAPL")).upper()]


def _load_unified_trackc_features(
    *,
    symbol: str,
    horizon: int,
    panel_cache_dir: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    unified = find_unified_panel_parquet(symbol, horizon, panel_cache_dir)
    if unified is None:
        raise FileNotFoundError(f"Missing unified Track-C parquet for {symbol} h{horizon}")
    df = pd.read_parquet(unified)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date", drop=True)
        except Exception:
            pass
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index().loc[str(start) : str(end)]
    df = df.select_dtypes(include=["number", "bool"]).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _ordered_union_columns(dfs: Sequence[pd.DataFrame]) -> List[str]:
    cols: List[str] = []
    seen: Set[str] = set()
    for df in dfs:
        for c in list(df.columns):
            s = str(c)
            if s not in seen:
                cols.append(s)
                seen.add(s)
    return cols


def run_stage_a_proxy(args: argparse.Namespace) -> None:
    """Learn Stage-A family weights using a cheap purged-CV linear proxy."""

    def _has_unified_for_universe() -> bool:
        universe_syms = _resolve_proxy_universe(args)
        return all(
            find_unified_panel_parquet(str(sym), int(args.horizon), getattr(args, "panel_cache_dir", None)) is not None
            for sym in universe_syms
        )

    universe_probe = _resolve_proxy_universe(args)
    has_unified = _has_unified_for_universe()
    if not has_unified:
        missing = [
            str(sym)
            for sym in universe_probe
            if find_unified_panel_parquet(str(sym), int(args.horizon), getattr(args, "panel_cache_dir", None)) is None
        ]
        raise FileNotFoundError(
            "Stage-A proxy requires unified per-symbol Track-C panels for all symbols (cached-only). "
            f"Missing unified panels for: {', '.join(missing)}"
        )

    # Unified per-symbol parquets are already available under cache/features.
    if getattr(args, "panel_cache_dir", None):
        args.panel_cache_dir = str(Path(str(args.panel_cache_dir)).expanduser())
    else:
        args.panel_cache_dir = str((REPO_ROOT / "cache" / "features").resolve())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    universe = _resolve_proxy_universe(args)
    families = list(resolve_families(args.families))

    # Load per-symbol features and labels (forward returns) and pool.
    feat_by_sym: Dict[str, pd.DataFrame] = {}
    y_by_sym: Dict[str, pd.Series] = {}
    for sym in universe:
        features_df = _load_unified_trackc_features(
            symbol=sym,
            horizon=int(args.horizon),
            panel_cache_dir=str(args.panel_cache_dir),
            start=str(args.start),
            end=str(args.end),
        )
        prices = fetch_prices(sym, str(args.start), str(args.end))
        forward_returns = compute_forward_returns(prices, int(args.horizon))
        aligned_idx = features_df.index.intersection(forward_returns.index)
        X_df = features_df.loc[aligned_idx]
        y = forward_returns.loc[aligned_idx].astype(float).replace([np.inf, -np.inf], np.nan)
        mask = np.isfinite(y.to_numpy())
        X_df = X_df.loc[mask]
        y = y.loc[mask]
        if X_df.empty:
            raise ValueError(f"No numeric features available for proxy Stage-A: {sym}")
        feat_by_sym[sym] = X_df
        y_by_sym[sym] = y

    union_cols = _ordered_union_columns(list(feat_by_sym.values()))
    col_to_family = _build_trackc_family_mapping(columns=union_cols, families=families)

    # Build pooled (time-sorted) matrix for alpha selection.
    pooled_rows: List[pd.DataFrame] = []
    for sym in universe:
        X_df = feat_by_sym[sym].reindex(columns=union_cols).fillna(0.0)
        y = y_by_sym[sym]
        tmp = X_df.copy()
        tmp["__target__"] = y
        tmp["__sym__"] = sym
        pooled_rows.append(tmp)
    pooled = pd.concat(pooled_rows, axis=0).sort_index()
    pooled_y = pooled["__target__"].astype(float)
    pooled_X_df = pooled.drop(columns=["__target__", "__sym__"], errors="ignore")
    pooled_X = pooled_X_df.to_numpy(dtype=np.float64)
    pooled_y_arr = pooled_y.to_numpy(dtype=np.float64)

    gap = int(args.proxy_gap) if args.proxy_gap is not None else int(args.horizon)
    splits = _purged_time_series_splits(
        n_samples=len(pooled_X_df),
        n_splits=int(args.proxy_folds),
        val_size=int(args.proxy_val_size),
        gap=int(gap),
        min_train=int(args.proxy_min_train),
    )

    method = str(getattr(args, "selector_method", "ridge_proxy")).lower().strip()
    alpha_grid = [float(a) for a in (args.proxy_alpha_grid or []) if float(a) > 0]
    if not alpha_grid:
        alpha_grid = [1e-2]

    def score_alpha(alpha: float) -> float:
        scores: List[float] = []
        for train_idx, val_idx in splits:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr = scaler.fit_transform(pooled_X[train_idx])
            X_va = scaler.transform(pooled_X[val_idx])
            y_tr = pooled_y_arr[train_idx]
            y_va = pooled_y_arr[val_idx]
            if method == "elasticnet_proxy":
                model = ElasticNet(alpha=float(alpha), l1_ratio=float(args.proxy_l1_ratio), max_iter=5000)
            else:
                model = Ridge(alpha=float(alpha))
            model.fit(X_tr, y_tr)
            pred = np.asarray(model.predict(X_va), dtype=np.float64)
            if pred.size < 5:
                continue
            denom = float(np.std(pred) * np.std(y_va) + 1e-12)
            ic = float(np.mean((pred - pred.mean()) * (y_va - y_va.mean())) / denom)
            if np.isfinite(ic):
                scores.append(ic)
        return float(np.mean(scores)) if scores else -1e9

    alpha_scores = {float(a): float(score_alpha(float(a))) for a in alpha_grid}
    best_alpha = float(max(alpha_scores.items(), key=lambda kv: kv[1])[0])
    LOGGER.info("🧪 Proxy alpha selection: best_alpha=%g scores=%s", best_alpha, {k: round(v, 4) for k, v in alpha_scores.items()})

    # Fold IC diagnostics at best_alpha.
    fold_scores: List[float] = []
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler.fit_transform(pooled_X[train_idx])
        X_va = scaler.transform(pooled_X[val_idx])
        y_tr = pooled_y_arr[train_idx]
        y_va = pooled_y_arr[val_idx]

        if method == "elasticnet_proxy":
            model = ElasticNet(alpha=float(best_alpha), l1_ratio=float(args.proxy_l1_ratio), max_iter=5000)
        else:
            model = Ridge(alpha=float(best_alpha))
        model.fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_va), dtype=np.float64)
        denom = float(np.std(pred) * np.std(y_va) + 1e-12)
        ic = float(np.mean((pred - pred.mean()) * (y_va - y_va.mean())) / denom)
        fold_scores.append(ic)
        LOGGER.info("📉 Fold %d: train=%d val=%d ic=%.4f", fold_id, len(train_idx), len(val_idx), ic)

    # Optional time-adaptive schedule.
    schedule_entries: List[Dict[str, Any]] = []
    do_schedule = bool(getattr(args, "proxy_time_adaptive", False))
    normalization = str(getattr(args, "proxy_weight_normalization", "mean1"))
    clip_min = float(getattr(args, "proxy_clip_min", 0.25))
    clip_max = float(getattr(args, "proxy_clip_max", 4.0))
    shrink = float(getattr(args, "proxy_shrink_to_uniform", 0.10))

    if do_schedule:
        master_idx = pd.DatetimeIndex(sorted({ts for sym in universe for ts in feat_by_sym[sym].index}))
        t0 = pd.to_datetime(getattr(args, "wf_start", None) or getattr(args, "start", None) or str(master_idx.min()))
        t1 = pd.to_datetime(getattr(args, "wf_end", None) or getattr(args, "end", None) or str(master_idx.max()))
        master_idx = master_idx[(master_idx >= t0) & (master_idx <= t1)]
        if master_idx.empty:
            raise ValueError("Empty master index for proxy schedule")

        lookback = int(max(10, int(getattr(args, "proxy_lookback", 1008))))
        update_every = int(max(1, int(getattr(args, "proxy_update_every", 21))))
        smooth_alpha = float(np.clip(float(getattr(args, "proxy_smooth_alpha", 0.2)), 0.0, 1.0))

        start_pos = int(lookback + int(args.horizon))
        if start_pos >= len(master_idx):
            raise ValueError("Not enough history for schedule: increase --end or reduce --proxy-lookback")

        prev_w: Dict[str, float] = {str(f): 1.0 for f in families}
        boundary_positions = list(range(start_pos, len(master_idx), update_every))
        for pos in boundary_positions:
            asof = master_idx[pos]
            # Use only matured history: [T-L, T-H]
            end_pos = int(max(0, pos - int(args.horizon)))
            start_pos_win = int(max(0, end_pos - lookback))
            start_date = master_idx[start_pos_win]
            end_date = master_idx[end_pos - 1] if end_pos - 1 >= 0 else master_idx[0]

            X_blocks: List[np.ndarray] = []
            y_blocks: List[np.ndarray] = []
            for sym in universe:
                X_df = feat_by_sym[sym]
                y = y_by_sym[sym]
                idx = X_df.index
                win_mask = (idx >= start_date) & (idx <= end_date)
                if not bool(np.any(win_mask)):
                    continue
                Xw = X_df.loc[win_mask].reindex(columns=union_cols).fillna(0.0)
                yw = y.loc[Xw.index].astype(float)
                if len(Xw) < int(getattr(args, "proxy_min_train", 756)):
                    continue
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xw_std = scaler.fit_transform(Xw.to_numpy(dtype=np.float64))
                X_blocks.append(Xw_std)
                y_blocks.append(yw.to_numpy(dtype=np.float64))

            if not X_blocks:
                continue
            X_fit = np.vstack(X_blocks)
            y_fit = np.concatenate(y_blocks)

            if method == "elasticnet_proxy":
                model = ElasticNet(alpha=float(best_alpha), l1_ratio=float(args.proxy_l1_ratio), max_iter=5000)
            else:
                model = Ridge(alpha=float(best_alpha))
            model.fit(X_fit, y_fit)
            coef = np.asarray(getattr(model, "coef_", np.zeros((X_fit.shape[1],), dtype=np.float64)), dtype=np.float64)
            raw_norms = _coef_norms_by_family(coef=coef, columns=union_cols, col_to_family=col_to_family, families=families)
            w_new = _normalize_clip_weights(
                raw=raw_norms,
                families=families,
                shrink_to_uniform=shrink,
                clip_min=clip_min,
                clip_max=clip_max,
                normalization=normalization,
            )

            w_smooth = {f: (1.0 - smooth_alpha) * float(prev_w.get(f, 1.0)) + smooth_alpha * float(w_new.get(f, 1.0)) for f in families}
            w_smooth = _normalize_clip_weights(
                raw=w_smooth,
                families=families,
                shrink_to_uniform=0.0,
                clip_min=clip_min,
                clip_max=clip_max,
                normalization=normalization,
            )
            prev_w = dict(w_smooth)

            schedule_entries.append(
                {
                    "asof": str(pd.to_datetime(asof).date()),
                    "weights": {k: float(v) for k, v in w_smooth.items()},
                    "window": {"start": str(pd.to_datetime(start_date).date()), "end": str(pd.to_datetime(end_date).date())},
                    "n_samples": int(len(y_fit)),
                }
            )

    # Static weights (fallback or summary): fit once on pooled data with best_alpha.
    pooled_scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = pooled_scaler.fit_transform(pooled_X)
    if method == "elasticnet_proxy":
        full_model = ElasticNet(alpha=float(best_alpha), l1_ratio=float(args.proxy_l1_ratio), max_iter=5000)
    else:
        full_model = Ridge(alpha=float(best_alpha))
    full_model.fit(X_tr, pooled_y_arr)
    full_coef = np.asarray(getattr(full_model, "coef_", np.zeros((X_tr.shape[1],), dtype=np.float64)), dtype=np.float64)
    avg_norms = _coef_norms_by_family(coef=full_coef, columns=union_cols, col_to_family=col_to_family, families=families)
    weights = _normalize_clip_weights(
        raw=avg_norms,
        families=families,
        shrink_to_uniform=shrink,
        clip_min=clip_min,
        clip_max=clip_max,
        normalization=normalization,
    )

    payload: Dict[str, Any] = {
        "symbol": str(args.symbol).upper(),
        "scope": "global" if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global" else "symbol",
        "symbols": list(universe),
        "horizon": int(args.horizon),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": str(getattr(args, "selector_method", "ridge_proxy")),
        "proxy": {
            "folds": int(len(splits)),
            "gap": int(gap),
            "val_size": int(args.proxy_val_size),
            "min_train": int(args.proxy_min_train),
            "alpha_grid": [float(a) for a in alpha_grid],
            "alpha_scores": {str(k): float(v) for k, v in alpha_scores.items()},
            "best_alpha": float(best_alpha),
            "l1_ratio": float(args.proxy_l1_ratio),
            "fold_ic": [float(x) for x in fold_scores],
            "weight_normalization": str(normalization),
            "clip": {"min": float(clip_min), "max": float(clip_max)},
            "schedule": {
                "enabled": bool(do_schedule),
                "update_every": int(getattr(args, "proxy_update_every", 21)),
                "lookback": int(getattr(args, "proxy_lookback", 1008)),
                "smooth_alpha": float(getattr(args, "proxy_smooth_alpha", 0.2)),
            },
        },
        "family_normalized_scores": weights,
        "family_weights": weights,
        "family_raw_norms": avg_norms,
    }

    if schedule_entries:
        payload["schedule"] = schedule_entries
        # Prefer last schedule snapshot as the headline weights.
        try:
            last_w = schedule_entries[-1].get("weights")
            if isinstance(last_w, dict) and last_w:
                payload["family_normalized_scores"] = {str(k): float(v) for k, v in last_w.items()}
                payload["family_weights"] = {str(k): float(v) for k, v in last_w.items()}
        except Exception:
            pass

    weights_path = outdir / "family_weights_best.json"
    weights_path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("💾 Saved proxy Stage-A weights → %s", weights_path)


def _fallback_threshold_for_horizon(horizon: int) -> float:
    if horizon <= 5:
        return 0.005
    if horizon <= 21:
        return 0.01
    if horizon <= 63:
        return 0.02
    if horizon <= 126:
        return 0.03
    return 0.04


def resolve_thresholds(
    forward_returns: pd.Series,
    mode: str,
    lower: float,
    upper: float,
    percentile: Optional[float],
    horizon: int,
    percentile_long: Optional[float] = None,
    percentile_short: Optional[float] = None,
) -> Tuple[float, float]:
    if mode == "fixed":
        if lower >= 0 or upper <= 0:
            raise ValueError("Fixed thresholds must satisfy lower < 0 < upper")
        return lower, upper
    clean = forward_returns.replace([np.inf, -np.inf], np.nan).dropna()
    abs_returns = clean.abs().values
    if abs_returns.size == 0:
        raise ValueError("Not enough returns to compute percentile thresholds")
    if mode == "percentile":
        if percentile is not None:
            pct = np.clip(percentile, 0.0, 100.0)
            thresh = float(np.percentile(abs_returns, pct))
            if not np.isfinite(thresh) or thresh <= 0:
                raise ValueError("Percentile threshold must be positive")
            return -thresh, thresh

        long_q = np.clip(percentile_long if percentile_long is not None else 0.6, 0.0, 1.0)
        short_q = np.clip(percentile_short if percentile_short is not None else 0.4, 0.0, 1.0)
        if not long_q > short_q:
            raise ValueError("percentile-long must be greater than percentile-short")
        upper_val = float(np.percentile(clean.values, long_q * 100.0))
        lower_val = float(np.percentile(clean.values, short_q * 100.0))
        if not np.isfinite(upper_val):
            raise ValueError("Upper percentile produced invalid threshold")
        if not np.isfinite(lower_val):
            lower_val = -abs(upper_val)
        if lower_val >= 0:
            lower_val = -abs(upper_val)
        if upper_val <= 0:
            upper_val = abs(lower_val)
        return float(lower_val), float(upper_val)
    if mode == "adaptive":
        fallback = _fallback_threshold_for_horizon(horizon)
        vol = float(np.nanstd(clean.values))
        if not np.isfinite(vol) or vol <= 0:
            thresh = fallback
        else:
            thresh = max(fallback, vol * 0.8)
        return -thresh, thresh
    raise ValueError(f"Unsupported threshold mode: {mode}")


def build_labels(
    forward_returns: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.Series:
    labels = pd.Series(index=forward_returns.index, dtype="Int64")
    labels.loc[forward_returns <= lower_threshold] = -1
    mid_mask = forward_returns.between(lower_threshold, upper_threshold)
    labels.loc[mid_mask] = 0
    labels.loc[forward_returns >= upper_threshold] = 1
    return labels.dropna().astype(int)


def trim_datasets(
    features: pd.DataFrame,
    forward_returns: pd.Series,
    labels: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    idx = features.index.intersection(forward_returns.index).intersection(labels.index)
    X = features.loc[idx]
    y = labels.loc[idx]
    r = forward_returns.loc[idx]
    mask = (~X.isna().any(axis=1)) & np.isfinite(r)
    X = X.loc[mask]
    y = y.loc[mask]
    r = r.loc[mask]
    return X, r, y


def validate_class_counts(labels: pd.Series, min_count: int) -> None:
    counts = labels.value_counts()
    missing = [int(cls) for cls in (-1, 0, 1) if counts.get(cls, 0) < min_count]
    if missing:
        raise RuntimeError(
            f"Insufficient samples for classes {missing}; counts={counts.to_dict()}"
        )


def make_dataset_global_pooled(args: argparse.Namespace) -> Dataset:
    """Build a pooled multi-symbol dataset for legacy Optuna.

    Each trial is evaluated on a dataset that includes *all* symbols in --symbols.
    We stack rows using a MultiIndex (date, symbol) and keep fold sizing based on
    unique dates later in the pipeline.
    """

    syms = resolve_universe_symbols(args)
    panel_start, panel_end = resolve_walkforward_window(args)
    LOGGER.info(
        "🌍 Building pooled Stage-A dataset (%d symbols) %s→%s",
        len(syms),
        panel_start,
        panel_end,
    )

    # Build per-symbol datasets using the same transformation pipeline.
    per_sym_features: Dict[str, pd.DataFrame] = {}
    per_sym_labels: Dict[str, pd.Series] = {}
    per_sym_returns: Dict[str, pd.Series] = {}
    per_sym_stats: Dict[str, Dict[str, Any]] = {}

    family_columns_union: Dict[str, List[str]] = defaultdict(list)
    families_union: Set[str] = set()
    coverage_by_sym: Dict[str, Dict[str, float]] = {}

    attempted_by_sym: Dict[str, Set[str]] = getattr(args, "_auto_regen_attempts_by_symbol", {})
    max_regen_attempts = max(0, int(getattr(args, "auto_regen_max_attempts", 0) or 0))

    for sym in syms:
        regen_attempts = 0
        while True:
            try:
                cache_dir = _resolve_cache_dir_for_symbol(args, sym, int(args.horizon))
                panel = load_panel_df(
                    symbol=sym,
                    start=panel_start,
                    end=panel_end,
                    families=args.families,
                    horizon=args.horizon,
                    cache_dir=cache_dir,
                    panel_horizon=args.panel_horizon,
                    require_cached_panel=args.require_cached_panel,
                    required_cached_families=args.require_cached_families,
                )

                effective_families = resolve_effective_families_for_panel(
                    panel,
                    args.families,
                    cached_only=bool(getattr(args, "require_cached_panel", False)),
                )
                if not effective_families:
                    raise RuntimeError(f"No usable feature families found in panel for {sym}")

                filtered_panel, family_columns, family_coverage, family_drop_summary = filter_columns(
                    panel,
                    families=effective_families,
                    max_nan_ratio=args.max_nan_ratio,
                    min_variance=args.min_variance,
                    max_cols_per_family=args.max_cols_per_family,
                    min_family_coverage=args.family_min_coverage,
                )
                family_columns = prune_missing_families_or_raise(
                    family_columns,
                    effective_families,
                    stage_label=f"Post-filter feature set ({sym})",
                    drop_summary=family_drop_summary,
                )
                filtered_panel, family_columns, _ = enforce_unique_family_columns(
                    filtered_panel,
                    family_columns,
                    stage="filtered",
                )

                lagged_panel, lagged_mapping = apply_family_lags(
                    filtered_panel,
                    family_columns,
                    args.disable_lags,
                )
                if lagged_panel is filtered_panel:
                    lagged_panel = filtered_panel
                    lagged_mapping = family_columns
                lagged_panel, lagged_mapping, _ = enforce_unique_family_columns(
                    lagged_panel,
                    lagged_mapping,
                    stage="lagged",
                )

                lagged_panel, lagged_mapping, _pca_fams, _pca_details = apply_conditional_family_pca(
                    lagged_panel,
                    lagged_mapping,
                    threshold=PCA_FEATURE_THRESHOLD,
                    protected_families=HF_PCA_PROTECTED_FAMILIES,
                    cache_enabled=args.pca_cache,
                    cache_dir=args.pca_cache_dir,
                )

                prices = fetch_prices(sym, panel_start, panel_end)
                forward_returns = compute_forward_returns(prices, args.horizon)
                lower_thresh, upper_thresh = resolve_thresholds(
                    forward_returns,
                    args.threshold_mode,
                    args.lower_threshold,
                    args.upper_threshold,
                    args.threshold_percentile,
                    args.horizon,
                    args.percentile_long,
                    args.percentile_short,
                )
                labels = build_labels(forward_returns, lower_thresh, upper_thresh)
                features, forward_returns, labels = trim_datasets(lagged_panel, forward_returns, labels)
                validate_class_counts(labels, args.min_samples_per_class)
                break
            except MissingFamilyColumnsError as exc:
                if regen_attempts >= max_regen_attempts:
                    raise
                if not getattr(args, "auto_regen_missing_families", True):
                    raise
                attempted = attempted_by_sym.get(sym, set())
                pending = [fam for fam in exc.families if fam not in attempted]
                if not pending:
                    raise
                attempted_by_sym[sym] = attempted | set(pending)
                setattr(args, "_auto_regen_attempts_by_symbol", attempted_by_sym)
                base_cache_dir, _, output_dir, _ = resolve_cache_paths(args, sym, args.horizon)
                LOGGER.warning(
                    "♻️ Auto-regenerating families for %s (%s)",
                    sym,
                    ", ".join(pending),
                )
                run_prep_families(
                    args,
                    sym,
                    args.horizon,
                    base_cache_dir,
                    output_dir,
                    families_override=pending,
                )
                # If we're sourcing from a unified TrackC parquet, refresh it as well.
                try:
                    effective_horizon = int(getattr(args, "panel_horizon", None) or args.horizon)
                    unified = find_unified_panel_parquet(sym, effective_horizon, str(base_cache_dir))
                    if unified is not None and unified.exists():
                        run_rebuild_trackc_from_cache(
                            symbol=sym,
                            horizon=effective_horizon,
                            start=panel_start,
                            end=panel_end,
                            cache_dir=base_cache_dir,
                            families=list(args.families),
                        )
                except Exception as rebuild_exc:
                    LOGGER.warning("⚠️ TrackC rebuild skipped/failed for %s: %s", sym, rebuild_exc)
                regen_attempts += 1

        per_sym_features[sym] = features
        per_sym_labels[sym] = labels
        per_sym_returns[sym] = forward_returns
        per_sym_stats[sym] = {
            "rows": int(len(features)),
            "cols": int(features.shape[1]),
            "cache_dir": str(cache_dir) if cache_dir else None,
        }

        families_union.update(list(lagged_mapping.keys()))
        coverage_by_sym[sym] = {fam: float(family_coverage.get(fam, 0.0)) for fam in lagged_mapping.keys()}
        for fam, cols in lagged_mapping.items():
            for c in cols:
                if c not in family_columns_union[fam]:
                    family_columns_union[fam].append(c)

        LOGGER.info("✅ %s: %d rows, %d cols", sym, int(len(features)), int(features.shape[1]))

    union_cols = _ordered_union_columns(list(per_sym_features.values()))

    X_blocks: List[pd.DataFrame] = []
    y_blocks: List[pd.Series] = []
    r_blocks: List[pd.Series] = []
    for sym in syms:
        X = per_sym_features[sym].reindex(columns=union_cols, fill_value=0.0)
        idx = pd.MultiIndex.from_arrays(
            [normalize_datetime_index(X.index), np.asarray([sym] * len(X), dtype=object)],
            names=["date", "symbol"],
        )
        X = X.copy()
        X.index = idx
        y = pd.Series(per_sym_labels[sym].to_numpy(), index=idx, name="label")
        r = pd.Series(per_sym_returns[sym].to_numpy(), index=idx, name="forward_return")
        X_blocks.append(X)
        y_blocks.append(y)
        r_blocks.append(r)

    features = pd.concat(X_blocks, axis=0).sort_index(level=[0, 1])
    labels = pd.concat(y_blocks, axis=0).sort_index(level=[0, 1]).astype(int)
    forward_returns = pd.concat(r_blocks, axis=0).sort_index(level=[0, 1]).astype(float)

    pooled_coverage: Dict[str, float] = {}
    for fam in sorted(families_union):
        pooled_coverage[fam] = float(np.mean([coverage_by_sym.get(sym, {}).get(fam, 0.0) for sym in syms]))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "global_universe.json").write_text(
        json.dumps(
            {
                "symbols": syms,
                "panel_start": panel_start,
                "panel_end": panel_end,
                "union_cols": int(len(union_cols)),
                "rows": int(len(features)),
                "per_symbol": per_sym_stats,
            },
            indent=2,
        )
    )

    feature_metadata = build_feature_metadata(family_columns_union)
    return Dataset(
        features=features,
        labels=labels,
        forward_returns=forward_returns,
        family_columns=dict(family_columns_union),
        families=sorted(list(families_union)),
        family_coverage=pooled_coverage,
        feature_metadata=feature_metadata,
        family_raw_columns={},
        family_component_projections={},
    )


def make_dataset(args: argparse.Namespace) -> Dataset:
    if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
        return make_dataset_global_pooled(args)
    # Use walk-forward dates for panel loading to match cache preparation
    panel_start, panel_end = resolve_walkforward_window(args)
    panel = load_panel_df(
        symbol=args.symbol,
        start=panel_start,
        end=panel_end,
        families=args.families,
        horizon=args.horizon,
        cache_dir=args.panel_cache_dir,
        panel_horizon=args.panel_horizon,
        require_cached_panel=args.require_cached_panel,
        required_cached_families=args.require_cached_families,
    )

    effective_families = resolve_effective_families_for_panel(
        panel,
        args.families,
        cached_only=bool(getattr(args, "require_cached_panel", False)),
    )
    if not effective_families:
        raise RuntimeError("No usable feature families found in panel")

    summary = summarize_panel(panel, effective_families)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "panel_summary.json").write_text(json.dumps(summary, indent=2))

    filtered_panel, family_columns, family_coverage, family_drop_summary = filter_columns(
        panel,
        families=effective_families,
        max_nan_ratio=args.max_nan_ratio,
        min_variance=args.min_variance,
        max_cols_per_family=args.max_cols_per_family,
        min_family_coverage=args.family_min_coverage,
    )
    ensure_mapping_family_coverage(
        family_columns,
        effective_families,
        "Post-filter feature set",
        drop_summary=family_drop_summary,
    )
    LOGGER.info("📐 Filtered panel shape: %s", filtered_panel.shape)
    filtered_panel, family_columns, _ = enforce_unique_family_columns(
        filtered_panel,
        family_columns,
        stage="filtered",
    )

    lagged_panel, lagged_mapping = apply_family_lags(
        filtered_panel,
        family_columns,
        args.disable_lags,
    )
    ensure_mapping_family_coverage(lagged_mapping, effective_families, "Lag-expanded feature set")
    if lagged_panel is not filtered_panel:
        LOGGER.info("⏱️ Applied lag expansion: %s → %s columns", filtered_panel.shape[1], lagged_panel.shape[1])
    else:
        lagged_panel = filtered_panel
        lagged_mapping = family_columns
    lagged_panel, lagged_mapping, _ = enforce_unique_family_columns(
        lagged_panel,
        lagged_mapping,
        stage="lagged",
    )

    stage_frames: List[Tuple[str, pd.DataFrame, Dict[str, List[str]]]] = [
        ("filtered", filtered_panel, family_columns),
        ("lagged", lagged_panel, lagged_mapping),
    ]

    raw_family_columns = {fam: list(cols) for fam, cols in lagged_mapping.items()}

    lagged_panel, lagged_mapping, pca_families, pca_projection_details = apply_conditional_family_pca(
        lagged_panel,
        lagged_mapping,
        threshold=PCA_FEATURE_THRESHOLD,
        protected_families=HF_PCA_PROTECTED_FAMILIES,
        cache_enabled=args.pca_cache,
        cache_dir=args.pca_cache_dir,
    )
    if pca_families:
        stage_frames.append(("pca", lagged_panel, lagged_mapping))
        preview = pca_families[:5]
        if len(pca_families) > 5:
            preview.append("…")
        LOGGER.info(
            "📉 Conditional PCA applied to %d families (> %d cols): %s",
            len(pca_families),
            PCA_FEATURE_THRESHOLD,
            preview,
        )

    family_component_projections = build_family_component_projections(
        lagged_mapping,
        raw_family_columns,
        pca_projection_details,
    )

    duplicate_reports: List[Dict[str, Any]] = []
    for stage_name, frame, mapping in stage_frames:
        report = analyze_duplicate_columns(frame.columns, mapping)
        if not report:
            continue
        stage_report = {"stage": stage_name, **report}
        duplicate_reports.append(stage_report)
        dup_names = list(report["columns"].keys())
        preview = dup_names[:10]
        if len(dup_names) > 10:
            preview.append("…")
        LOGGER.warning(
            "⚠️ Duplicate columns detected after %s stage (%d unique): %s",
            stage_name,
            report["duplicate_column_count"],
            preview,
        )
        for fam, info in stage_report["families"].items():
            fam_dup_names = [entry["name"] for entry in info["columns"][:5]]
            if len(info["columns"]) > 5:
                fam_dup_names.append("…")
            LOGGER.warning(
                "   ↳ Family %s duplicates (%d values): %s",
                fam,
                info["total"],
                fam_dup_names,
            )
    if duplicate_reports:
        dump_duplicate_reports(duplicate_reports, outdir, args.symbol, args.horizon)

    prices = fetch_prices(args.symbol, panel_start, panel_end)
    forward_returns = compute_forward_returns(prices, args.horizon)
    lower_thresh, upper_thresh = resolve_thresholds(
        forward_returns,
        args.threshold_mode,
        args.lower_threshold,
        args.upper_threshold,
        args.threshold_percentile,
        args.horizon,
        args.percentile_long,
        args.percentile_short,
    )
    LOGGER.info("🎯 Thresholds resolved: lower=%.4f upper=%.4f", lower_thresh, upper_thresh)
    labels = build_labels(forward_returns, lower_thresh, upper_thresh)
    features, forward_returns, labels = trim_datasets(lagged_panel, forward_returns, labels)
    validate_class_counts(labels, args.min_samples_per_class)

    resolved_families = list(lagged_mapping.keys())
    coverage_filtered = {fam: family_coverage.get(fam, 0.0) for fam in resolved_families}
    feature_metadata = build_feature_metadata(lagged_mapping)

    return Dataset(
        features=features,
        labels=labels,
        forward_returns=forward_returns,
        family_columns=lagged_mapping,
        families=resolved_families,
        family_coverage=coverage_filtered,
        feature_metadata=feature_metadata,
        family_raw_columns=raw_family_columns,
        family_component_projections=family_component_projections,
    )


class RollingWindowSplitter:
    """Time-ordered splitter with explicit train/validation window sizing."""

    def __init__(self, min_train: int, val_size: int, max_splits: int) -> None:
        self.min_train = min_train
        self.val_size = val_size
        self.max_splits = max_splits

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        start = self.min_train
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        while start + self.val_size <= n_samples:
            train_idx = np.arange(start)
            val_idx = np.arange(start, start + self.val_size)
            splits.append((train_idx, val_idx))
            start += self.val_size
        if not splits:
            raise ValueError("Not enough samples to generate any folds")
        if self.max_splits and len(splits) > self.max_splits:
            splits = splits[-self.max_splits:]
        for train_idx, val_idx in splits:
            yield train_idx, val_idx


def make_splitter(args: argparse.Namespace, n_samples: int) -> RollingWindowSplitter:
    if args.fold_size <= 0:
        raise ValueError("fold_size must be positive")
    if args.min_train_window <= 0:
        raise ValueError("min_train_window must be positive")
    if args.min_train_window + args.fold_size > n_samples:
        raise ValueError("Dataset too small for requested train/validation windows")
    return RollingWindowSplitter(
        min_train=args.min_train_window,
        val_size=args.fold_size,
        max_splits=args.folds,
    )


def build_fold_specs(
    splitter: RollingWindowSplitter,
    features: pd.DataFrame,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Materialize splitter indices for reuse by Optuna workers."""

    fold_specs: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(features)):
        if train_idx.size == 0 or val_idx.size == 0:
            LOGGER.warning("Skipping empty fold %d (train=%d val=%d)", fold_id, train_idx.size, val_idx.size)
            continue
        fold_specs.append((train_idx.copy(), val_idx.copy()))
        LOGGER.debug(
            "Fold %d: train %d samples (%s → %s), val %d samples (%s → %s)",
            fold_id,
            train_idx.size,
            features.index[train_idx[0]] if train_idx.size else "-",
            features.index[train_idx[-1]] if train_idx.size else "-",
            val_idx.size,
            features.index[val_idx[0]] if val_idx.size else "-",
            features.index[val_idx[-1]] if val_idx.size else "-",
        )
    return fold_specs


def build_fold_specs_by_date(
    features: pd.DataFrame,
    *,
    min_train_days: int,
    val_days: int,
    max_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build fold specs based on unique dates rather than row counts.

    This is required for pooled multi-symbol datasets where each trading date
    contributes multiple (date, symbol) rows.
    """

    if min_train_days <= 0 or val_days <= 0:
        raise ValueError("min_train_days and val_days must be positive")

    if isinstance(features.index, pd.MultiIndex):
        dates = normalize_datetime_index(features.index.get_level_values(0))
    else:
        dates = normalize_datetime_index(features.index)
    unique_dates = pd.DatetimeIndex(pd.unique(dates)).sort_values()
    if len(unique_dates) < int(min_train_days) + int(val_days):
        raise ValueError("Dataset too small for requested train/validation windows")

    date_to_rows: Dict[pd.Timestamp, List[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        date_to_rows[pd.Timestamp(d)].append(i)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = int(min_train_days)
    while start + int(val_days) <= len(unique_dates):
        train_dates = unique_dates[:start]
        val_dates = unique_dates[start : start + int(val_days)]
        train_idx = np.asarray([j for d in train_dates for j in date_to_rows[pd.Timestamp(d)]], dtype=int)
        val_idx = np.asarray([j for d in val_dates for j in date_to_rows[pd.Timestamp(d)]], dtype=int)
        splits.append((train_idx, val_idx))
        start += int(val_days)
    if not splits:
        raise ValueError("Not enough samples to generate any folds")
    if max_splits and len(splits) > int(max_splits):
        splits = splits[-int(max_splits) :]
    return splits


def inject_noise(rng: np.random.Generator, X: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    target_dtype = X.dtype
    scale_arr = np.asarray(scale, dtype=target_dtype)
    if np.all(scale_arr <= 0):
        return X
    noise = rng.normal(0.0, 1.0, size=X.shape).astype(target_dtype, copy=False) * scale_arr
    return X + noise


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    forward_returns: np.ndarray,
) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    weight = np.abs(forward_returns)
    rwa = float((weight * (y_true == y_pred)).sum() / (weight.sum() + 1e-9))
    positions = np.where(y_pred == 1, 1.0, np.where(y_pred == -1, -1.0, 0.0))
    strat_returns = positions * forward_returns
    sharpe = float(
        (np.mean(strat_returns) / (np.std(strat_returns) + 1e-9)) * math.sqrt(252)
    )
    coverage = float(np.mean(np.abs(positions) > 0))
    long_mask = np.isin(y_true, (0, 1))
    short_mask = np.isin(y_true, (0, -1))

    def masked_acc(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(accuracy_score(y_true[mask], y_pred[mask]))

    acc_long = masked_acc(long_mask)
    acc_short = masked_acc(short_mask)
    return {
        "acc": acc,
        "rwa": rwa,
        "sharpe": sharpe,
        "coverage": coverage,
        "acc_long": acc_long,
        "acc_short": acc_short,
    }


def compute_sharpe_from_predictions(y_pred: np.ndarray, forward_returns: np.ndarray) -> float:
    positions = np.where(y_pred == 1, 1.0, np.where(y_pred == -1, -1.0, 0.0))
    strat_returns = positions * forward_returns
    return float((np.mean(strat_returns) / (np.std(strat_returns) + 1e-9)) * math.sqrt(252))


def compute_feature_sharpe_deltas(
    model: GroupLassoSoftmax,
    X_val_scaled: np.ndarray,
    baseline_logits: np.ndarray,
    forward_returns: np.ndarray,
    columns: Sequence[str],
    baseline_sharpe: float,
) -> Dict[str, float]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Model classes_ missing for Sharpe delta computation")
    classes = np.asarray(classes)
    deltas: Dict[str, float] = {}
    for idx, column in enumerate(columns):
        coef_column = model.coef_[:, idx]
        if not np.any(coef_column):
            deltas[column] = 0.0
            continue
        feature_effect = np.outer(X_val_scaled[:, idx], coef_column)
        logits_minus = baseline_logits - feature_effect
        pred_indices = np.argmax(logits_minus, axis=1)
        preds_minus = classes[pred_indices]
        sharpe_minus = compute_sharpe_from_predictions(preds_minus, forward_returns)
        deltas[column] = float(baseline_sharpe - sharpe_minus)
    return deltas


def build_family_feature_payload(
    families: Sequence[str],
    family_columns: Mapping[str, Sequence[str]],
    component_importances: Mapping[str, float],
    component_deltas: Mapping[str, float],
    family_component_projections: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for family in families:
        projection = family_component_projections.get(family)
        if projection:
            raw_cols = projection.get("raw_columns", [])
            weights_map: Mapping[str, np.ndarray] = projection.get("component_weights", {})
        else:
            fallback_cols = list(family_columns.get(family, []))
            raw_cols = fallback_cols
            weights_map = {}
            for idx, col in enumerate(fallback_cols):
                eye = np.zeros(len(fallback_cols), dtype=np.float32)
                eye[idx] = 1.0
                weights_map[col] = eye
        if not raw_cols:
            continue
        raw_importance = np.zeros(len(raw_cols), dtype=np.float32)
        raw_delta = np.zeros(len(raw_cols), dtype=np.float32)
        for column in family_columns.get(family, []):
            weights = weights_map.get(column)
            if weights is None or weights.size != len(raw_cols):
                continue
            importance = float(component_importances.get(column, 0.0))
            delta = float(component_deltas.get(column, 0.0))
            raw_importance += importance * weights
            raw_delta += delta * weights
        payload[family] = {
            "raw_columns": list(raw_cols),
            "importance": {
                raw_cols[idx]: float(raw_importance[idx]) for idx in range(len(raw_cols))
            },
            "delta": {raw_cols[idx]: float(raw_delta[idx]) for idx in range(len(raw_cols))},
        }
    return payload


def aggregate_family_feature_scores(
    families: Sequence[str],
    fold_feature_payloads: Sequence[Mapping[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    family_details: Dict[str, Any] = {}
    family_scores: Dict[str, float] = {}
    dropped: List[str] = []
    hard_dropped: List[str] = []
    negative_families: List[str] = []
    for family in families:
        feature_values: Dict[str, List[float]] = defaultdict(list)
        delta_values: Dict[str, List[float]] = defaultdict(list)
        for fold_payload in fold_feature_payloads:
            fam_payload = fold_payload.get(family)
            if not fam_payload:
                continue
            for feature, value in fam_payload.get("importance", {}).items():
                feature_values[feature].append(float(value))
            for feature, value in fam_payload.get("delta", {}).items():
                delta_values[feature].append(float(value))
        if not feature_values:
            family_details[family] = {
                "features": [],
                "family_mean": 0.0,
                "family_stability": 0.0,
                "useful_fraction": 0.0,
                "family_delta": 0.0,
            }
            family_scores[family] = -1.0
            hard_dropped.append(family)
            negative_families.append(family)
            continue
        feature_stats: List[Dict[str, Any]] = []
        useful_count = 0
        mean_values: List[float] = []
        stability_values: List[float] = []
        delta_means: List[float] = []
        for feature, values in feature_values.items():
            arr = np.array(values, dtype=float)
            mean_imp = float(np.mean(arr)) if arr.size else 0.0
            std_imp = float(np.std(arr)) if arr.size else 0.0
            stability = 1.0 - (std_imp / (abs(mean_imp) + FEATURE_CONTRIB_EPS))
            stability = float(np.clip(stability, 0.0, 1.0))
            delta_arr = np.array(delta_values.get(feature, [0.0]), dtype=float)
            mean_delta = float(np.mean(delta_arr)) if delta_arr.size else 0.0
            useful = mean_imp > FEATURE_USEFUL_THRESHOLD
            if useful:
                useful_count += 1
            mean_values.append(mean_imp)
            stability_values.append(stability)
            delta_means.append(mean_delta)
            feature_stats.append(
                {
                    "feature": feature,
                    "mean_importance": mean_imp,
                    "std_importance": std_imp,
                    "stability": stability,
                    "mean_delta": mean_delta,
                    "useful": useful,
                }
            )
        total_features = len(feature_stats)
        family_mean = float(np.mean(mean_values)) if mean_values else 0.0
        family_stability = float(np.mean(stability_values)) if stability_values else 0.0
        useful_fraction = useful_count / total_features if total_features else 0.0
        family_delta = float(np.mean(delta_means)) if delta_means else 0.0
        score = (
            FAMILY_SCORE_WEIGHTS["mean"] * family_mean
            + FAMILY_SCORE_WEIGHTS["stability"] * family_stability
            + FAMILY_SCORE_WEIGHTS["useful"] * useful_fraction
            + FAMILY_SCORE_WEIGHTS["delta"] * family_delta
        )
        score = float(np.clip(score, -1.0, 1.0))
        if score < 0:
            negative_families.append(family)
        if score < FAMILY_SCORE_LOWER_THRESHOLD:
            dropped.append(family)
        if score < FAMILY_SCORE_HARD_CUT_THRESHOLD:
            hard_dropped.append(family)
        family_scores[family] = score
        family_details[family] = {
            "features": feature_stats,
            "family_mean": family_mean,
            "family_stability": family_stability,
            "useful_fraction": useful_fraction,
            "family_delta": family_delta,
            "score": score,
        }
    return {
        "family_scores": family_scores,
        "family_details": family_details,
        "dropped_families": dropped,
        "hard_drops": hard_dropped,
        "negative_families": negative_families,
    }


def aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not fold_metrics:
        raise ValueError("No fold metrics to aggregate")
    keys = fold_metrics[0].keys()
    aggregated = {}
    for k in keys:
        values = np.array([fm[k] for fm in fold_metrics], dtype=float)
        aggregated[k] = float(np.nanmean(values))
    return aggregated


def make_noise_seed(base_seed: int, fold_idx: int) -> int:
    seed = (hash((base_seed, fold_idx)) & 0xFFFFFFFF) or 1
    return seed


def train_and_eval_fold(
    fold_id: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset_arrays: Dict[str, Any],
    params: Dict[str, Any],
    noise_seed: int,
) -> Optional[Dict[str, Any]]:
    X = dataset_arrays["features"]
    y = dataset_arrays["labels"]
    r = dataset_arrays["returns"]
    columns = dataset_arrays["columns"]
    family_columns = dataset_arrays["family_columns"]
    families = dataset_arrays["families"]

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    r_val = r[val_idx]

    if len(np.unique(y_train)) < 2:
        return None

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler_start = perf_counter()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32, copy=False)
    X_val_scaled = scaler.transform(X_val).astype(np.float32, copy=False)
    scaler_time = perf_counter() - scaler_start

    std_per_feature = np.std(X_train_scaled, axis=0, dtype=np.float32)
    noise_scale = params.get("feature_noise_std", 0.0) * (std_per_feature + 1e-6)
    rng = np.random.default_rng(noise_seed)
    X_train_noisy = inject_noise(rng, X_train_scaled, noise_scale).astype(np.float32, copy=False)

    group_ids = dataset_arrays.get("group_ids")
    if group_ids is None:
        raise ValueError("Group ids missing from dataset arrays")
    lam = float(params.get("lambda", 0.0))
    l1_ratio = float(params.get("group_l1_ratio", 0.0))
    model = GroupLassoSoftmax(
        groups=group_ids,
        group_reg=lam,
        l1_reg=lam * max(l1_ratio, 0.0),
        learning_rate=float(params.get("group_learning_rate", 0.1)),
        max_iter=int(params.get("group_max_iter", 200)),
        tol=float(params.get("group_tol", 1e-4)),
    )
    solver_start = perf_counter()
    model.fit(X_train_noisy, y_train)
    solver_time = perf_counter() - solver_start
    model.scaler_ = scaler  # type: ignore[attr-defined]
    if not getattr(model, "converged_", True):
        LOGGER.warning(
            "⚠️ GroupLasso solver hit max_iter=%d in fold %d (lambda=%.6f)",
            model.max_iter,
            fold_id,
            lam,
        )
    LOGGER.info(
        "⏱️ Fold %d timings: scaler %.2fs, solver %.2fs",
        fold_id,
        scaler_time,
        solver_time,
    )
    preds = model.predict(X_val_scaled)
    metrics = score_predictions(y_val, preds, r_val)
    baseline_logits = model.decision_function(X_val_scaled)
    feature_deltas = compute_feature_sharpe_deltas(
        model,
        X_val_scaled,
        baseline_logits,
        r_val,
        columns,
        metrics["sharpe"],
    )

    fam_scores = compute_family_group_norms(
        coef_matrix=model.coef_,
        family_column_indices=dataset_arrays["family_column_indices"],
    )
    if dataset_arrays.get("normalize_family_variance"):
        fam_scores = normalize_scores_by_family_variance(
            fam_scores,
            family_columns,
            dataset_arrays.get("column_variances", {}),
        )
    normalized_scores = normalize_importances(fam_scores)
    feature_scores = compute_feature_group_norms(model.coef_, columns)
    feature_normalized = normalize_importances(feature_scores)
    family_feature_payload = build_family_feature_payload(
        families,
        family_columns,
        feature_scores,
        feature_deltas,
        dataset_arrays.get("family_component_projections", {}),
    )
    active = sum(1 for score in fam_scores.values() if score > 1e-6)
    active_ratio = active / max(len(families), 1)
    return {
        "metrics": metrics,
        "active_ratio": active_ratio,
        "family_scores": fam_scores,
        "normalized_scores": normalized_scores,
        "feature_scores": feature_scores,
        "feature_normalized_scores": feature_normalized,
        "coefficients": model.coef_.tolist(),
        "fold_id": fold_id,
        "family_feature_payload": family_feature_payload,
    }


def evaluate_trial_core(
    dataset_arrays: Dict[str, Any],
    fold_specs: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    trial_seed: int,
    fold_workers: int,
    trial_obj: Optional[optuna.trial.Trial] = None,
) -> Dict[str, Any]:
    if not fold_specs:
        return {"status": "pruned", "reason": "No fold specifications"}

    fold_jobs = []
    for fold_id, (train_idx, val_idx) in enumerate(fold_specs):
        seed = make_noise_seed(trial_seed, fold_id)
        fold_jobs.append((fold_id, train_idx, val_idx, seed))

    results: List[Optional[Dict[str, Any]]] = []
    if fold_workers and fold_workers > 1:
        parallel = Parallel(n_jobs=fold_workers, prefer="threads")
        results = parallel(
            delayed(train_and_eval_fold)(fold_id, train_idx, val_idx, dataset_arrays, params, seed)
            for fold_id, train_idx, val_idx, seed in fold_jobs
        )
    else:
        for fold_id, train_idx, val_idx, seed in fold_jobs:
            results.append(
                train_and_eval_fold(fold_id, train_idx, val_idx, dataset_arrays, params, seed)
            )

    valid_results = [res for res in results if res]
    if not valid_results:
        return {"status": "pruned", "reason": "No valid folds"}

    fold_metrics = [res["metrics"] for res in valid_results]
    aggregated = aggregate_metrics(fold_metrics)
    active_ratios = [res["active_ratio"] for res in valid_results]
    fold_normalized_scores = [res["normalized_scores"] for res in valid_results]
    fold_family_scores = [res["family_scores"] for res in valid_results]
    fold_feature_scores = [res["feature_scores"] for res in valid_results]
    fold_feature_normalized = [res["feature_normalized_scores"] for res in valid_results]
    stability = compute_fold_stability(fold_normalized_scores)
    stability_score = compute_stability_score(stability)
    avg_family_scores = {
        fam: float(np.nanmean([scores.get(fam, 0.0) for scores in fold_family_scores]))
        for fam in dataset_arrays["families"]
    }
    normalized_importances = {
        fam: float(np.nanmean([scores.get(fam, 0.0) for scores in fold_normalized_scores]))
        for fam in dataset_arrays["families"]
    }
    columns = dataset_arrays["columns"]
    avg_feature_scores = {
        col: float(np.nanmean([scores.get(col, 0.0) for scores in fold_feature_scores]))
        for col in columns
    }
    normalized_feature_scores = {
        col: float(np.nanmean([scores.get(col, 0.0) for scores in fold_feature_normalized]))
        for col in columns
    }
    fold_feature_payloads = [res.get("family_feature_payload", {}) for res in valid_results]
    feature_summary = aggregate_family_feature_scores(dataset_arrays["families"], fold_feature_payloads)
    family_scores = feature_summary["family_scores"]
    family_feature_details = feature_summary["family_details"]
    dropped_families = feature_summary["dropped_families"]
    hard_drops = feature_summary["hard_drops"]
    negative_families = feature_summary["negative_families"]
    for fam in dropped_families:
        normalized_importances[fam] = 0.0
    for fam in hard_drops:
        avg_family_scores[fam] = 0.0
    mean_family_score = float(np.mean(list(family_scores.values()))) if family_scores else 0.0
    drop_penalty = FAMILY_NEGATIVE_PENALTY_WEIGHT * len(negative_families)
    sparsity_penalty = runtime_cfg["family_sparsity_coef"] * float(np.mean(active_ratios)) if active_ratios else 0.0
    composite = (
        runtime_cfg["w_rwa"] * aggregated["rwa"]
        + runtime_cfg["w_sharpe"] * aggregated["sharpe"]
        + runtime_cfg["w_stability"] * stability_score
        + FAMILY_SCORE_OBJECTIVE_WEIGHT * mean_family_score
        - drop_penalty
        - sparsity_penalty
    )
    return {
        "status": "ok",
        "value": composite,
        "metrics": aggregated,
        "sparsity_penalty": sparsity_penalty,
        "normalized_scores": normalized_importances,
        "average_family_scores": avg_family_scores,
        "fold_normalized_scores": fold_normalized_scores,
        "fold_family_scores": fold_family_scores,
        "fold_feature_scores": fold_feature_scores,
        "fold_feature_normalized_scores": fold_feature_normalized,
        "average_feature_scores": avg_feature_scores,
        "normalized_feature_scores": normalized_feature_scores,
        "fold_coefficients": [res["coefficients"] for res in valid_results],
        "fold_ids": [res["fold_id"] for res in valid_results],
        "stability": stability,
        "stability_score": stability_score,
        "family_scores": family_scores,
        "family_feature_details": family_feature_details,
        "dropped_families": dropped_families,
        "hard_drops": hard_drops,
        "negative_families": negative_families,
        "mean_family_score": mean_family_score,
        "family_drop_penalty": drop_penalty,
    }


def sample_trial_params(trial: optuna.trial.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "feature_noise_std": args.feature_noise_std,
        "group_learning_rate": args.group_learning_rate,
        "group_max_iter": args.group_max_iter,
        "group_tol": args.group_tol,
        "group_l1_ratio": args.group_l1_ratio,
    }
    lambda_base = args.stagea_lambda if args.stagea_lambda and args.stagea_lambda > 0 else None
    span = args.stagea_lambda_span if args.stagea_lambda_span and args.stagea_lambda_span > 0 else None
    if lambda_base is not None and span is not None:
        if span <= 1.0 or args.n_trials <= 1:
            lam_value = max(lambda_base, 1e-6)
        else:
            low = max(lambda_base / span, 1e-6)
            high = max(lambda_base * span, low + 1e-6)
            lam_value = trial.suggest_float("lambda", low, high, log=True)
        params["lambda"] = lam_value
    else:
        lam_value = trial.suggest_float("lambda", 5e-4, 2.5e-3, log=True)
        params["lambda"] = lam_value
    return params

@dataclass
class RayTrialState:
    trial: optuna.trial.Trial
    params: Dict[str, Any]
    trial_seed: int
    next_fold: int = 0
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    pruned: bool = False


def run_parallel_trials_with_ray(
    study: optuna.study.Study,
    dataset_arrays: Dict[str, Any],
    fold_specs: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    runtime_payload: Dict[str, Any],
) -> None:
    if ray is None or _ray_fold_runner is None:
        raise RuntimeError("Ray runtime unavailable")
    if not ray.is_initialized():
        ray.init(num_cpus=args.ray_parallelism, ignore_reinit_error=True)

    LOGGER.info(
        "🚀 Starting Optuna study (%d trials, ray=%d, folds=%d)",
        args.n_trials,
        args.ray_parallelism,
        len(fold_specs),
    )

    dataset_ref = ray.put(dataset_arrays)
    max_workers = max(1, min(args.ray_parallelism or 1, args.n_trials))
    launched = 0
    pending: Dict["ray.ObjectRef", Tuple[RayTrialState, int]] = {}
    ready_queue: deque[RayTrialState] = deque()

    def schedule_trial() -> None:
        nonlocal launched
        if launched >= args.n_trials:
            return
        trial = study.ask()
        # Attach pooled universe metadata for post-hoc verification (Ray path).
        try:
            if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
                trial.set_user_attr("universe_mode", "global")
                trial.set_user_attr("symbols", tuple(resolve_universe_symbols(args)))
            else:
                trial.set_user_attr("universe_mode", "grid")
        except Exception:
            pass
        try:
            trial.set_user_attr("ray_parallelism", int(getattr(args, "ray_parallelism", 0) or 0))
            trial.set_user_attr("n_jobs", int(getattr(args, "n_jobs", 1) or 1))
        except Exception:
            pass
        params = sample_trial_params(trial, args)
        trial_seed = make_noise_seed(args.seed, trial.number + 1)
        params = dict(params)
        params.setdefault("seed", trial_seed)
        state = RayTrialState(trial=trial, params=params, trial_seed=trial_seed)
        ready_queue.append(state)
        launched += 1

    def launch_fold(state: RayTrialState) -> None:
        fold_id = state.next_fold
        train_idx, val_idx = fold_specs[fold_id]
        seed = make_noise_seed(state.trial_seed, fold_id)
        future = _ray_fold_runner.remote(dataset_ref, train_idx, val_idx, state.params, fold_id, seed)
        pending[future] = (state, fold_id)
        state.next_fold += 1

    while pending or ready_queue or launched < args.n_trials:
        while len(pending) < max_workers:
            if ready_queue:
                candidate = ready_queue.popleft()
                if candidate.pruned or candidate.next_fold >= len(fold_specs):
                    continue
                launch_fold(candidate)
            elif launched < args.n_trials:
                schedule_trial()
            else:
                break
        if not pending:
            if launched >= args.n_trials and not ready_queue:
                break
            continue
        futures = list(pending.keys())
        ready, _ = ray.wait(futures, num_returns=1)
        finished = ready[0]
        state, fold_id = pending.pop(finished)
        try:
            result = ray.get(finished)
        except Exception as exc:  # pragma: no cover - runtime failure path
            LOGGER.exception("Ray fold worker failed for trial %s fold %d", state.trial.number, fold_id)
            state.trial.set_user_attr("exception", str(exc))
            study.tell(state.trial, state=TrialState.FAIL)
            schedule_trial()
            continue

        if not result:
            state.trial.set_user_attr("pruned_reason", f"Fold {fold_id} returned no data")
            study.tell(state.trial, state=TrialState.PRUNED)
            schedule_trial()
            continue

        state.fold_results.append(result)
        fold_value = compute_fold_value(result["metrics"], result["active_ratio"], runtime_payload)
        state.trial.report(fold_value, step=fold_id)
        if state.trial.should_prune():
            state.pruned = True
            state.trial.set_user_attr("pruned_reason", f"Pruned at fold {fold_id}")
            study.tell(state.trial, state=TrialState.PRUNED)
            schedule_trial()
            continue

        if state.next_fold < len(fold_specs):
            ready_queue.append(state)
            continue

        summary = summarize_trial_results(state.fold_results, runtime_payload, dataset_arrays)
        trial = state.trial
        trial.set_user_attr("metrics", summary["metrics"])
        trial.set_user_attr("sparsity_penalty", summary["sparsity_penalty"])
        trial.set_user_attr("normalized_scores", summary["normalized_scores"])
        trial.set_user_attr("average_family_scores", summary["average_family_scores"])
        trial.set_user_attr("fold_normalized_scores", summary["fold_normalized_scores"])
        trial.set_user_attr("fold_family_scores", summary["fold_family_scores"])
        trial.set_user_attr("fold_feature_scores", summary["fold_feature_scores"])
        trial.set_user_attr(
            "fold_feature_normalized_scores",
            summary["fold_feature_normalized_scores"],
        )
        trial.set_user_attr("average_feature_scores", summary["average_feature_scores"])
        trial.set_user_attr("normalized_feature_scores", summary["normalized_feature_scores"])
        trial.set_user_attr("fold_coefficients", summary["fold_coefficients"])
        trial.set_user_attr("fold_ids", summary["fold_ids"])
        trial.set_user_attr("stability", summary["stability"])
        trial.set_user_attr("stability_score", summary.get("stability_score"))
        trial.set_user_attr("family_scores", summary.get("family_scores"))
        trial.set_user_attr("family_feature_details", summary.get("family_feature_details"))
        trial.set_user_attr("dropped_families", summary.get("dropped_families"))
        trial.set_user_attr("hard_drops", summary.get("hard_drops"))
        trial.set_user_attr("mean_family_score", summary.get("mean_family_score"))
        study.tell(trial, summary["value"])
        schedule_trial()


def compute_family_group_norms(
    coef_matrix: np.ndarray,
    family_column_indices: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    fam_scores: Dict[str, float] = {}
    for fam, indices in family_column_indices.items():
        if not indices:
            continue
        block = coef_matrix[:, indices]
        fam_scores[fam] = float(np.linalg.norm(block))
    return fam_scores


def compute_feature_group_norms(
    coef_matrix: np.ndarray,
    column_names: Sequence[str],
) -> Dict[str, float]:
    norms = np.linalg.norm(coef_matrix, axis=0)
    return {column_names[idx]: float(norms[idx]) for idx in range(len(column_names))}


def compute_fold_value(
    metrics: Dict[str, float],
    active_ratio: float,
    runtime_cfg: Dict[str, Any],
) -> float:
    sparsity_penalty = runtime_cfg["family_sparsity_coef"] * active_ratio
    return (
        runtime_cfg["w_rwa"] * metrics["rwa"]
        + runtime_cfg["w_sharpe"] * metrics["sharpe"]
        + runtime_cfg["w_stability"] * runtime_cfg.get("fold_stability_baseline", 0.0)
        - sparsity_penalty
    )


def normalize_scores_by_family_variance(
    fam_scores: Dict[str, float],
    family_columns: Dict[str, List[str]],
    variance_lookup: Mapping[str, float],
) -> Dict[str, float]:
    if not fam_scores or not variance_lookup:
        return dict(fam_scores)
    adjusted: Dict[str, float] = {}
    eps = 1e-12
    for family, score in fam_scores.items():
        cols = family_columns.get(family, [])
        if not cols:
            adjusted[family] = score
            continue
        values = [float(variance_lookup.get(col, 0.0)) for col in cols]
        positive = [val for val in values if val > 0]
        if not positive:
            adjusted[family] = score
            continue
        mean_var = float(np.mean(positive))
        adjusted[family] = score / max(mean_var, eps)
    return adjusted


def softmax(values: Dict[str, float], temperature: float) -> Dict[str, float]:
    temp = max(temperature, 1e-6)
    arr = np.array(list(values.values()), dtype=float) / temp
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = exp.sum()
    if denom <= 0:
        n = len(values)
        return {k: 1.0 / n for k in values.keys()} if n else {}
    weights = exp / denom
    return {k: float(w) for k, w in zip(values.keys(), weights)}


def compute_entropy(weights: Dict[str, float]) -> float:
    arr = np.array(list(weights.values()))
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def summarize_trial_results(
    valid_results: Sequence[Dict[str, Any]],
    runtime_cfg: Dict[str, Any],
    dataset_arrays: Dict[str, Any],
) -> Dict[str, Any]:
    fold_metrics = [res["metrics"] for res in valid_results]
    aggregated = aggregate_metrics(fold_metrics)
    active_ratios = [res["active_ratio"] for res in valid_results]
    fold_normalized_scores = [res["normalized_scores"] for res in valid_results]
    fold_family_scores = [res["family_scores"] for res in valid_results]
    fold_feature_scores = [res["feature_scores"] for res in valid_results]
    fold_feature_normalized = [res["feature_normalized_scores"] for res in valid_results]
    stability = compute_fold_stability(fold_normalized_scores)
    stability_score = compute_stability_score(stability)
    families = dataset_arrays["families"]
    avg_family_scores = {
        fam: float(np.nanmean([scores.get(fam, 0.0) for scores in fold_family_scores]))
        for fam in families
    }
    normalized_importances = {
        fam: float(np.nanmean([scores.get(fam, 0.0) for scores in fold_normalized_scores]))
        for fam in families
    }
    columns = dataset_arrays["columns"]
    avg_feature_scores = {
        col: float(np.nanmean([scores.get(col, 0.0) for scores in fold_feature_scores]))
        for col in columns
    }
    normalized_feature_scores = {
        col: float(np.nanmean([scores.get(col, 0.0) for scores in fold_feature_normalized]))
        for col in columns
    }
    fold_feature_payloads = [res.get("family_feature_payload", {}) for res in valid_results]
    feature_summary = aggregate_family_feature_scores(dataset_arrays["families"], fold_feature_payloads)
    family_scores = feature_summary["family_scores"]
    family_feature_details = feature_summary["family_details"]
    dropped_families = feature_summary["dropped_families"]
    hard_drops = feature_summary["hard_drops"]
    negative_families = feature_summary["negative_families"]
    for fam in dropped_families:
        normalized_importances[fam] = 0.0
    for fam in hard_drops:
        avg_family_scores[fam] = 0.0
    mean_family_score = float(np.mean(list(family_scores.values()))) if family_scores else 0.0
    drop_penalty = FAMILY_NEGATIVE_PENALTY_WEIGHT * len(negative_families)
    sparsity_penalty = (
        runtime_cfg["family_sparsity_coef"] * float(np.mean(active_ratios)) if active_ratios else 0.0
    )
    composite = (
        runtime_cfg["w_rwa"] * aggregated["rwa"]
        + runtime_cfg["w_sharpe"] * aggregated["sharpe"]
        + runtime_cfg["w_stability"] * stability_score
        + FAMILY_SCORE_OBJECTIVE_WEIGHT * mean_family_score
        - drop_penalty
        - sparsity_penalty
    )
    return {
        "status": "ok",
        "value": composite,
        "metrics": aggregated,
        "sparsity_penalty": sparsity_penalty,
        "normalized_scores": normalized_importances,
        "average_family_scores": avg_family_scores,
        "fold_normalized_scores": fold_normalized_scores,
        "fold_family_scores": fold_family_scores,
        "fold_feature_scores": fold_feature_scores,
        "fold_feature_normalized_scores": fold_feature_normalized,
        "average_feature_scores": avg_feature_scores,
        "normalized_feature_scores": normalized_feature_scores,
        "fold_coefficients": [res["coefficients"] for res in valid_results],
        "fold_ids": [res["fold_id"] for res in valid_results],
        "stability": stability,
        "stability_score": stability_score,
        "family_scores": family_scores,
        "family_feature_details": family_feature_details,
        "dropped_families": dropped_families,
        "hard_drops": hard_drops,
        "negative_families": negative_families,
        "mean_family_score": mean_family_score,
        "family_drop_penalty": drop_penalty,
    }


def create_objective(
    dataset_arrays: Dict[str, Any],
    fold_specs: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    runtime_payload: Dict[str, Any],
):
    def objective(trial: optuna.trial.Trial) -> float:
        # Attach pooled universe metadata for post-hoc verification.
        try:
            if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
                trial.set_user_attr("universe_mode", "global")
                trial.set_user_attr("symbols", tuple(resolve_universe_symbols(args)))
            else:
                trial.set_user_attr("universe_mode", "grid")
        except Exception:
            pass
        try:
            trial.set_user_attr("ray_parallelism", int(getattr(args, "ray_parallelism", 0) or 0))
            trial.set_user_attr("n_jobs", int(getattr(args, "n_jobs", 1) or 1))
        except Exception:
            pass
        params = sample_trial_params(trial, args)
        trial_seed = make_noise_seed(args.seed, trial.number + 1)
        params = dict(params)
        params.setdefault("seed", trial_seed)
        result = evaluate_trial_core(
            dataset_arrays=dataset_arrays,
            fold_specs=fold_specs,
            params=params,
            runtime_cfg=runtime_payload,
            trial_seed=trial_seed,
            fold_workers=args.fold_workers,
            trial_obj=trial,
        )
        if result["status"] != "ok":
            raise optuna.TrialPruned(result.get("reason", "Unknown failure"))
        trial.set_user_attr("metrics", result["metrics"])
        trial.set_user_attr("sparsity_penalty", result["sparsity_penalty"])
        trial.set_user_attr("normalized_scores", result["normalized_scores"])
        trial.set_user_attr("average_family_scores", result["average_family_scores"])
        trial.set_user_attr("fold_normalized_scores", result["fold_normalized_scores"])
        trial.set_user_attr("fold_family_scores", result["fold_family_scores"])
        trial.set_user_attr("fold_feature_scores", result["fold_feature_scores"])
        trial.set_user_attr(
            "fold_feature_normalized_scores",
            result["fold_feature_normalized_scores"],
        )
        trial.set_user_attr("average_feature_scores", result["average_feature_scores"])
        trial.set_user_attr("normalized_feature_scores", result["normalized_feature_scores"])
        trial.set_user_attr("fold_coefficients", result["fold_coefficients"])
        trial.set_user_attr("fold_ids", result["fold_ids"])
        trial.set_user_attr("stability", result["stability"])
        trial.set_user_attr("stability_score", result.get("stability_score"))
        trial.set_user_attr("family_scores", result.get("family_scores"))
        trial.set_user_attr("family_feature_details", result.get("family_feature_details"))
        trial.set_user_attr("dropped_families", result.get("dropped_families"))
        trial.set_user_attr("hard_drops", result.get("hard_drops"))
        trial.set_user_attr("mean_family_score", result.get("mean_family_score"))
        return result["value"]

    return objective


def run_study(
    dataset: Dataset,
    splitter: RollingWindowSplitter,
    args: argparse.Namespace,
) -> optuna.study.Study:
    dataset_arrays = build_dataset_arrays(dataset)
    dataset_arrays["normalize_family_variance"] = args.normalize_family_variance
    fold_specs = build_fold_specs(splitter, dataset.features)
    runtime_payload = build_runtime_payload(args)

    if not fold_specs:
        raise RuntimeError("No folds produced by splitter")

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=bool(args.storage and args.study_name),
    )

    use_ray = bool(args.ray_parallelism and args.ray_parallelism > 0)
    if use_ray and ray is None:
        LOGGER.warning("Ray requested but not available; reverting to local execution")
        use_ray = False

    if use_ray and _ray_fold_runner is None:
        LOGGER.warning("Ray runtime not initialized; reverting to local execution")
        use_ray = False

    if use_ray:
        run_parallel_trials_with_ray(
            study=study,
            dataset_arrays=dataset_arrays,
            fold_specs=fold_specs,
            args=args,
            runtime_payload=runtime_payload,
        )
    else:
        objective = create_objective(dataset_arrays, fold_specs, args, runtime_payload)
        LOGGER.info("🚀 Starting Optuna study (%d trials, local)", args.n_trials)
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=min(args.n_jobs, args.n_trials),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    if not study.best_trials:
        raise RuntimeError("Optuna did not produce any trials")
    LOGGER.info("🏆 Best value: %.4f", study.best_value)
    LOGGER.info("🏅 Best params: %s", study.best_trial.params)
    return study


def run_study_with_fold_specs(
    dataset: Dataset,
    fold_specs: List[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> optuna.study.Study:
    """Run Optuna using an explicit set of fold index arrays."""

    dataset_arrays = build_dataset_arrays(dataset)
    dataset_arrays["normalize_family_variance"] = args.normalize_family_variance
    runtime_payload = build_runtime_payload(args)

    if not fold_specs:
        raise RuntimeError("No folds produced")

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=bool(args.storage and args.study_name),
    )

    use_ray = bool(args.ray_parallelism and args.ray_parallelism > 0)
    if use_ray and ray is None:
        LOGGER.warning("Ray requested but not available; reverting to local execution")
        use_ray = False
    if use_ray and _ray_fold_runner is None:
        LOGGER.warning("Ray runtime not initialized; reverting to local execution")
        use_ray = False

    if use_ray:
        run_parallel_trials_with_ray(
            study=study,
            dataset_arrays=dataset_arrays,
            fold_specs=fold_specs,
            args=args,
            runtime_payload=runtime_payload,
        )
    else:
        objective = create_objective(dataset_arrays, fold_specs, args, runtime_payload)
        LOGGER.info("🚀 Starting Optuna study (%d trials, local)", args.n_trials)
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=min(args.n_jobs, args.n_trials),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    if not study.best_trials:
        raise RuntimeError("Optuna did not produce any trials")
    LOGGER.info("🏆 Best value: %.4f", study.best_value)
    LOGGER.info("🏅 Best params: %s", study.best_trial.params)
    return study


def normalize_importances(scores: Dict[str, float]) -> Dict[str, float]:
    positive = {k: max(0.0, float(v)) for k, v in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: value / total for k, value in positive.items()}


def compute_fold_stability(fold_scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not fold_scores:
        return {}
    all_keys = {key for scores in fold_scores for key in scores.keys()}
    stability: Dict[str, float] = {}
    for key in sorted(all_keys):
        values = np.array([scores.get(key, 0.0) for scores in fold_scores], dtype=float)
        stability[key] = float(np.nanstd(values))
    return stability


def compute_stability_score(stability: Mapping[str, float]) -> float:
    if not stability:
        return 0.0
    values = np.array(list(stability.values()), dtype=float)
    if values.size == 0:
        return 0.0
    mean_std = float(np.nanmean(values))
    return 1.0 / (1.0 + max(mean_std, 0.0))


def fit_best_model(
    dataset: Dataset,
    params: Dict[str, object],
    seed: int,
) -> GroupLassoSoftmax:
    scaler = StandardScaler(with_mean=True, with_std=True)
    feature_matrix = dataset.features.to_numpy(dtype=np.float32, copy=True)
    scaler.fit(feature_matrix)
    X_scaled = scaler.transform(feature_matrix).astype(np.float32, copy=False)
    columns = list(dataset.features.columns)
    group_ids, _, _ = build_group_metadata(columns, dataset.family_columns, dataset.families)
    lam = float(params.get("lambda", 0.0))
    l1_ratio = float(params.get("group_l1_ratio", 0.0))
    model = GroupLassoSoftmax(
        groups=group_ids,
        group_reg=lam,
        l1_reg=lam * max(l1_ratio, 0.0),
        learning_rate=float(params.get("group_learning_rate", 0.1)),
        max_iter=int(params.get("group_max_iter", 200)),
        tol=float(params.get("group_tol", 1e-4)),
    )
    model.fit(X_scaled, dataset.labels.to_numpy())
    if not getattr(model, "converged_", True):
        LOGGER.warning(
            "⚠️ Final GroupLasso solver hit max_iter=%d (lambda=%.6f)",
            model.max_iter,
            lam,
        )
    model.scaler_ = scaler  # type: ignore[attr-defined]
    return model


def build_family_weights(
    model: GroupLassoSoftmax,
    dataset: Dataset,
    temperature: float,
    normalize_variance: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    columns = list(dataset.features.columns)
    _, _, family_column_indices = build_group_metadata(
        columns,
        dataset.family_columns,
        dataset.families,
    )
    fam_scores = compute_family_group_norms(model.coef_, family_column_indices)
    for fam in dataset.families:
        fam_scores.setdefault(fam, 0.0)
    for fam in FORCED_FAMILIES:
        fam_scores.setdefault(fam, 0.0)
    if normalize_variance:
        LOGGER.info("⚖️ Applying inverse-variance normalization to family logits")
        fam_scores = normalize_scores_by_family_variance(
            fam_scores,
            dataset.family_columns,
            dataset.features.var(axis=0, ddof=0).fillna(0.0).to_dict(),
        )
    if not any(score > 0 for score in fam_scores.values()):
        LOGGER.warning("No non-zero family scores; defaulting to uniform weights")
        fam_scores = {fam: 1.0 for fam in dataset.families}
    normalized = normalize_importances(fam_scores)
    weights = softmax(fam_scores, temperature=temperature)
    norm_preview = sorted(((fam, round(score, 6)) for fam, score in fam_scores.items()), key=lambda kv: kv[1], reverse=True)
    LOGGER.info("📊 Family group norms (top 8): %s", norm_preview[:8])
    return fam_scores, weights, normalized


def select_families(
    weights: Dict[str, float],
    coverage: Dict[str, float],
    min_weight: float,
    min_coverage: float,
    stability: Optional[Dict[str, float]] = None,
    stability_alpha: float = 1.0,
    family_scores: Optional[Mapping[str, float]] = None,
    score_threshold: float = FAMILY_SCORE_LOWER_THRESHOLD,
    hard_cut_threshold: float = FAMILY_SCORE_HARD_CUT_THRESHOLD,
) -> List[str]:
    selected: List[str] = []
    for fam, weight in weights.items():
        fam_cov = coverage.get(fam, 0.0)
        fam_score = family_scores.get(fam) if family_scores else None
        if fam_score is not None and fam_score < hard_cut_threshold and fam not in FORCED_FAMILIES:
            LOGGER.info(
                "🚫 Dropping family %s due to hard score threshold (score=%.3f < %.2f)",
                fam,
                fam_score,
                hard_cut_threshold,
            )
            continue
        if fam_score is not None and fam_score < score_threshold and fam not in FORCED_FAMILIES:
            LOGGER.info(
                "🚫 Dropping family %s due to low family score (score=%.3f < %.2f)",
                fam,
                fam_score,
                score_threshold,
            )
            continue
        if fam in FORCED_FAMILIES:
            selected.append(fam)
            continue
        penalty = 1.0
        if stability and stability_alpha > 0:
            penalty = math.exp(-stability.get(fam, 0.0) * stability_alpha)
        effective_weight = weight * penalty
        if effective_weight >= min_weight and fam_cov >= min_coverage:
            selected.append(fam)
        else:
            LOGGER.info(
                "🚫 Dropping family %s during selection (weight=%.3f coverage=%.2f)",
                fam,
                effective_weight,
                fam_cov,
            )
    if not selected:
        logger_msg = "Selection removed all families; falling back to top-weighted entry"
        LOGGER.warning("⚠️ %s", logger_msg)
        if weights:
            selected = [max(weights.items(), key=lambda kv: kv[1])[0]]
    return selected


def write_outputs(
    outdir: Path,
    weights: Dict[str, float],
    logits: Dict[str, float],
    normalized: Dict[str, float],
    average_scores: Dict[str, float],
    stability: Dict[str, float],
    metrics: Dict[str, float],
    extras: Dict[str, object],
    best_trial: optuna.trial.FrozenTrial,
    selected_families: List[str],
    coverage: Dict[str, float],
    config: Dict[str, object],
    symbol: str,
    horizon: int,
    fold_normalized: Sequence[Dict[str, float]],
    fold_scores: Sequence[Dict[str, float]],
    fold_coefficients: Sequence[Sequence[List[float]]],
    fold_ids: Sequence[int],
    feature_columns: Sequence[str],
    full_model_coefficients: Sequence[List[float]],
    feature_metadata: Dict[str, Dict[str, Any]],
    average_feature_scores: Dict[str, float],
    normalized_feature_scores: Dict[str, float],
    fold_feature_scores: Sequence[Dict[str, float]],
    fold_feature_normalized: Sequence[Dict[str, float]],
    family_scores: Mapping[str, float],
    family_feature_details: Mapping[str, Any],
    dropped_families: Sequence[str],
    hard_drops: Sequence[str],
) -> None:
    top_features_by_family = summarize_top_features_by_family(
        feature_metadata,
        average_feature_scores,
        normalized_feature_scores,
    )
    if top_features_by_family:
        preview_entries: List[str] = []
        for family, entries in top_features_by_family.items():
            if not entries:
                continue
            label = f"{family}:{entries[0]['column']}"
            preview_entries.append(label)
            if len(preview_entries) >= 6:
                break
        if preview_entries:
            LOGGER.info("🏅 Top feature per family: %s", ", ".join(preview_entries))
    payload = {
        "symbol": symbol,
        "horizon": horizon,
        "timestamp": extras.get("generated_at"),
        "selected_families": selected_families,
        "family_weights": weights,
        "family_logits": logits,
        "family_normalized_scores": normalized,
        "family_average_scores": average_scores,
        "family_stability": stability,
        "family_coverage": coverage,
        "family_scores": dict(family_scores),
        "family_feature_details": family_feature_details,
        "family_drop_summary": {
            "soft": list(dropped_families),
            "hard": list(hard_drops),
        },
        "metrics": metrics,
        "config": config,
        "extras": extras,
        "family_feature_leaders": top_features_by_family,
        "global": {
            "weights": {f"family_{k}": v for k, v in weights.items()},
            "logits": {f"family_{k}": v for k, v in logits.items()},
            "normalized": {f"family_{k}": v for k, v in normalized.items()},
            "average_scores": {f"family_{k}": v for k, v in average_scores.items()},
            "stability": {f"family_{k}": v for k, v in stability.items()},
            "metrics": metrics,
            "extras": extras,
        },
        "weights_by_regime": {},
        "fold_analysis": {
            "fold_ids": list(fold_ids),
            "normalized_scores": list(fold_normalized),
            "raw_scores": list(fold_scores),
            "coefficients": {
                "feature_columns": list(feature_columns),
                "folds": [list(map(list, coef_matrix)) for coef_matrix in fold_coefficients],
            },
            "full_model_coefficients": {
                "feature_columns": list(feature_columns),
                "coef": [list(row) for row in full_model_coefficients],
            },
        },
        "feature_analysis": {
            "metadata": feature_metadata,
            "average_scores": average_feature_scores,
            "normalized_scores": normalized_feature_scores,
            "fold_scores": list(fold_feature_scores),
            "fold_normalized": list(fold_feature_normalized),
            "top_features_by_family": top_features_by_family,
        },
    }
    weights_path = outdir / "family_weights_best.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("💾 Saved weights → %s", weights_path)

    trial_summary = {
        "value": best_trial.value,
        "params": best_trial.params,
        "metrics": best_trial.user_attrs.get("metrics", {}),
        "sparsity_penalty": best_trial.user_attrs.get("sparsity_penalty"),
        "stability_score": best_trial.user_attrs.get("stability_score"),
        "selected_families": selected_families,
        "config": config,
    }
    (outdir / "best_trial.json").write_text(json.dumps(trial_summary, indent=2))


def _coerce_multiindex_levels(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(df.index, pd.MultiIndex):
        return None, None
    levels = list(getattr(df.index, "names", []) or [])
    date_level = None
    sym_level = None
    for name in levels:
        if str(name).lower().strip() == "date":
            date_level = name
        if str(name).lower().strip() == "symbol":
            sym_level = name
    # Fallback: assume 0=date, 1=symbol if unnamed.
    if date_level is None and len(levels) >= 1:
        date_level = levels[0]
    if sym_level is None and len(levels) >= 2:
        sym_level = levels[1]
    return date_level, sym_level


def _build_proxy_schedule_from_dataset(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    universe: Sequence[str],
    families_override: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (schedule_entries, proxy_meta) for causal time-adaptive family weights.

    Uses only matured history at each boundary T:
      train window = [T-L, T-H]
    where H = horizon (sessions) and L = lookback (sessions).
    """

    # Reconstruct per-symbol frames from pooled Dataset.
    date_level, sym_level = _coerce_multiindex_levels(dataset.features)
    if isinstance(dataset.features.index, pd.MultiIndex) and sym_level is not None:
        feat_by_sym = {sym: dataset.features.xs(sym, level=sym_level, drop_level=True) for sym in universe}
        y_by_sym = {sym: dataset.forward_returns.xs(sym, level=sym_level, drop_level=True) for sym in universe}
    else:
        # Single-symbol dataset.
        sym = str(getattr(args, "symbol", "AAPL")).upper()
        feat_by_sym = {sym: dataset.features}
        y_by_sym = {sym: dataset.forward_returns}

    all_cols = list(dataset.features.columns)
    all_families = [str(f) for f in (dataset.families or [])]

    col_to_family: Dict[str, str] = {}
    for fam, cols in (dataset.family_columns or {}).items():
        for c in cols or []:
            col_to_family[str(c)] = str(fam)
    if not col_to_family and all_cols and all_families:
        # Fallback: infer family mapping from Track-C naming.
        col_to_family.update(_build_trackc_family_mapping(columns=[str(c) for c in all_cols], families=all_families))

    families = list(all_families)
    if families_override is not None:
        wanted = [str(f) for f in families_override]
        wanted_set = set(wanted)
        intersect = [f for f in wanted if f in set(all_families)]
        if not intersect:
            LOGGER.warning("⚠️ families_override had no overlap with dataset families; falling back to all families")
        else:
            families = intersect

    union_cols: List[Any]
    if families_override is not None and isinstance(getattr(dataset, "family_columns", None), dict) and dataset.family_columns:
        # Prefer authoritative mapping when present.
        feat_set = set(all_cols)
        seen: set = set()
        union_cols = []
        for fam in families:
            cols = dataset.family_columns.get(fam) or dataset.family_columns.get(str(fam)) or []
            for c in cols:
                if c in feat_set and c not in seen:
                    union_cols.append(c)
                    seen.add(c)
    else:
        allowed = set(str(f) for f in families)
        union_cols = [c for c in all_cols if str(col_to_family.get(str(c), "")) in allowed]

    if not union_cols or not families:
        return [], {}

    # Build pooled rows for alpha selection (cheap proxy).
    pooled_rows: List[pd.DataFrame] = []
    for sym in universe:
        X_df = feat_by_sym.get(sym)
        y = y_by_sym.get(sym)
        if X_df is None or y is None or X_df.empty:
            continue
        X_df = X_df.reindex(columns=union_cols).fillna(0.0)
        y = y.reindex(X_df.index).astype(float)
        tmp = X_df.copy()
        tmp["__target__"] = y
        tmp["__sym__"] = sym
        pooled_rows.append(tmp)
    if not pooled_rows:
        return [], {}
    pooled = pd.concat(pooled_rows, axis=0)
    # Preserve time ordering: sort by date then symbol when possible.
    try:
        pooled = pooled.sort_index()
    except Exception:
        pass

    pooled_y = pooled["__target__"].astype(float)
    pooled_X_df = pooled.drop(columns=["__target__", "__sym__"], errors="ignore")
    pooled_X = pooled_X_df.to_numpy(dtype=np.float64)
    pooled_y_arr = pooled_y.to_numpy(dtype=np.float64)

    gap = int(getattr(args, "proxy_gap", None) or getattr(args, "horizon", 63))
    splits = _purged_time_series_splits(
        n_samples=len(pooled_X_df),
        n_splits=int(getattr(args, "proxy_folds", 5)),
        val_size=int(getattr(args, "proxy_val_size", 252)),
        gap=int(gap),
        min_train=int(getattr(args, "proxy_min_train", 756)),
    )

    method = str(getattr(args, "selector_method", "ridge_proxy")).lower().strip()
    if method not in {"ridge_proxy", "elasticnet_proxy"}:
        # Legacy Optuna still uses ridge as the default proxy.
        method = "ridge_proxy"
    alpha_grid = [float(a) for a in (getattr(args, "proxy_alpha_grid", None) or []) if float(a) > 0]
    if not alpha_grid:
        alpha_grid = [1e-2]

    def score_alpha(alpha: float) -> float:
        scores: List[float] = []
        for train_idx, val_idx in splits:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr = scaler.fit_transform(pooled_X[train_idx])
            X_va = scaler.transform(pooled_X[val_idx])
            y_tr = pooled_y_arr[train_idx]
            y_va = pooled_y_arr[val_idx]
            if str(method) == "elasticnet_proxy":
                model = ElasticNet(alpha=float(alpha), l1_ratio=float(getattr(args, "proxy_l1_ratio", 0.1)), max_iter=5000)
            else:
                model = Ridge(alpha=float(alpha))
            model.fit(X_tr, y_tr)
            pred = np.asarray(model.predict(X_va), dtype=np.float64)
            if pred.size < 5:
                continue
            denom = float(np.std(pred) * np.std(y_va) + 1e-12)
            ic = float(np.mean((pred - pred.mean()) * (y_va - y_va.mean())) / denom)
            if np.isfinite(ic):
                scores.append(ic)
        return float(np.mean(scores)) if scores else -1e9

    alpha_scores = {float(a): float(score_alpha(float(a))) for a in alpha_grid}
    best_alpha = float(max(alpha_scores.items(), key=lambda kv: kv[1])[0])

    master_dates = pd.DatetimeIndex(sorted({ts for sym in universe for ts in feat_by_sym.get(sym, pd.DataFrame()).index}))
    t0 = pd.to_datetime(getattr(args, "wf_start", None) or getattr(args, "start", None) or str(master_dates.min()))
    t1 = pd.to_datetime(getattr(args, "wf_end", None) or getattr(args, "end", None) or str(master_dates.max()))
    master_dates = master_dates[(master_dates >= t0) & (master_dates <= t1)]
    if master_dates.empty:
        return [], {}

    lookback = int(max(10, int(getattr(args, "proxy_lookback", 1008))))
    update_every = int(max(1, int(getattr(args, "proxy_update_every", 21))))
    smooth_alpha = float(np.clip(float(getattr(args, "proxy_smooth_alpha", 0.2)), 0.0, 1.0))
    normalization = str(getattr(args, "proxy_weight_normalization", "mean1"))
    clip_min = float(getattr(args, "proxy_clip_min", 0.25))
    clip_max = float(getattr(args, "proxy_clip_max", 4.0))
    shrink = float(getattr(args, "proxy_shrink_to_uniform", 0.10))

    start_pos = int(lookback + int(getattr(args, "horizon", 63)))
    if start_pos >= len(master_dates):
        return [], {}

    schedule_entries: List[Dict[str, Any]] = []
    prev_w: Dict[str, float] = {str(f): 1.0 for f in families}
    boundary_positions = list(range(start_pos, len(master_dates), update_every))
    for pos in boundary_positions:
        asof = master_dates[pos]
        # Use only matured history: [T-L, T-H]
        end_pos = int(max(0, pos - int(getattr(args, "horizon", 63))))
        start_pos_win = int(max(0, end_pos - lookback))
        start_date = master_dates[start_pos_win]
        end_date = master_dates[end_pos - 1] if end_pos - 1 >= 0 else master_dates[0]

        X_blocks: List[np.ndarray] = []
        y_blocks: List[np.ndarray] = []
        for sym in universe:
            X_df = feat_by_sym.get(sym)
            y = y_by_sym.get(sym)
            if X_df is None or y is None or X_df.empty:
                continue
            idx = X_df.index
            win_mask = (idx >= start_date) & (idx <= end_date)
            if not bool(np.any(win_mask)):
                continue
            Xw = X_df.loc[win_mask].reindex(columns=union_cols).fillna(0.0)
            yw = y.loc[Xw.index].astype(float)
            if len(Xw) < int(getattr(args, "proxy_min_train", 756)):
                continue
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xw_std = scaler.fit_transform(Xw.to_numpy(dtype=np.float64))
            X_blocks.append(Xw_std)
            y_blocks.append(yw.to_numpy(dtype=np.float64))

        if not X_blocks:
            continue
        X_fit = np.vstack(X_blocks)
        y_fit = np.concatenate(y_blocks)

        if str(method) == "elasticnet_proxy":
            model = ElasticNet(alpha=float(best_alpha), l1_ratio=float(getattr(args, "proxy_l1_ratio", 0.1)), max_iter=5000)
        else:
            model = Ridge(alpha=float(best_alpha))
        model.fit(X_fit, y_fit)
        coef = np.asarray(getattr(model, "coef_", np.zeros((X_fit.shape[1],), dtype=np.float64)), dtype=np.float64)
        raw_norms = _coef_norms_by_family(coef=coef, columns=union_cols, col_to_family=col_to_family, families=families)
        w_new = _normalize_clip_weights(
            raw=raw_norms,
            families=families,
            shrink_to_uniform=shrink,
            clip_min=clip_min,
            clip_max=clip_max,
            normalization=normalization,
        )
        w_smooth = {f: (1.0 - smooth_alpha) * float(prev_w.get(f, 1.0)) + smooth_alpha * float(w_new.get(f, 1.0)) for f in families}
        w_smooth = _normalize_clip_weights(
            raw=w_smooth,
            families=families,
            shrink_to_uniform=0.0,
            clip_min=clip_min,
            clip_max=clip_max,
            normalization=normalization,
        )
        prev_w = dict(w_smooth)

        schedule_entries.append(
            {
                "asof": str(pd.to_datetime(asof).date()),
                "weights": {k: float(v) for k, v in w_smooth.items()},
                "window": {"start": str(pd.to_datetime(start_date).date()), "end": str(pd.to_datetime(end_date).date())},
                "n_samples": int(len(y_fit)),
            }
        )

    proxy_meta = {
        "method": str(method),
        "best_alpha": float(best_alpha),
        "alpha_grid": [float(a) for a in alpha_grid],
        "alpha_scores": {str(k): float(v) for k, v in alpha_scores.items()},
        "gap": int(gap),
        "val_size": int(getattr(args, "proxy_val_size", 252)),
        "min_train": int(getattr(args, "proxy_min_train", 756)),
        "folds": int(getattr(args, "proxy_folds", 5)),
        "weight_normalization": str(normalization),
        "clip": {"min": float(clip_min), "max": float(clip_max)},
        "schedule": {
            "enabled": True,
            "update_every": int(update_every),
            "lookback": int(lookback),
            "smooth_alpha": float(smooth_alpha),
        },
    }
    return schedule_entries, proxy_meta


def _maybe_attach_proxy_schedule(outdir: Path, dataset: Dataset, args: argparse.Namespace) -> None:
    """Attach a time-adaptive Stage-A schedule to the existing weights artifact.

    Phase-2 consumes `schedule` automatically when present.
    """

    if not bool(getattr(args, "proxy_time_adaptive", False)):
        return
    weights_path = outdir / "family_weights_best.json"
    if not weights_path.exists():
        return
    try:
        payload = json.loads(weights_path.read_text())
    except Exception:
        return
    if isinstance(payload, dict) and isinstance(payload.get("schedule"), list) and payload.get("schedule"):
        return

    # Determine universe symbols for schedule pooling.
    universe: List[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("symbols"), list) and payload.get("symbols"):
        universe = [str(s).upper() for s in payload.get("symbols")]
    elif isinstance(dataset.features.index, pd.MultiIndex):
        _, sym_level = _coerce_multiindex_levels(dataset.features)
        if sym_level is not None:
            try:
                universe = [str(s).upper() for s in sorted(set(dataset.features.index.get_level_values(sym_level)))]
            except Exception:
                universe = []
    if not universe:
        universe = [str(getattr(args, "symbol", "AAPL")).upper()]

    # Restrict schedule generation to the best trial's selected families.
    families_override: Optional[List[str]] = None
    best_path = outdir / "best_trial.json"
    if best_path.exists():
        try:
            best_payload = json.loads(best_path.read_text())
            sel = best_payload.get("selected_families") if isinstance(best_payload, dict) else None
            if isinstance(sel, list) and sel:
                families_override = [str(f) for f in sel]
        except Exception:
            families_override = None

    schedule_entries, proxy_meta = _build_proxy_schedule_from_dataset(
        dataset,
        args,
        universe=universe,
        families_override=families_override,
    )
    if not schedule_entries:
        return

    payload["schedule"] = schedule_entries
    proxy_block = payload.get("proxy")
    if not isinstance(proxy_block, dict):
        proxy_block = {}
    # Keep schedule metadata for auditing.
    proxy_block.update(proxy_meta)
    payload["proxy"] = proxy_block

    # Prefer the last schedule snapshot as headline weights.
    last_w = schedule_entries[-1].get("weights")
    if isinstance(last_w, dict) and last_w:
        payload["family_normalized_scores"] = {str(k): float(v) for k, v in last_w.items() if isinstance(v, (int, float))}
        payload["family_weights"] = {str(k): float(v) for k, v in last_w.items() if isinstance(v, (int, float))}

    weights_path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("🕒 Attached time-adaptive proxy schedule (%d entries) → %s", len(schedule_entries), weights_path)


def save_trial_history(study: optuna.study.Study, outdir: Path) -> None:
    history_path = outdir / "trial_history.csv"
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        df = study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "state",
                "params",
                "user_attrs",
                "datetime_start",
                "datetime_complete",
            )
        )
        df.to_csv(history_path, index=False)
        LOGGER.info("💾 Saved trial history → %s", history_path)
    except Exception as exc:
        LOGGER.warning("⚠️ Failed to persist trial history: %s", exc)


def run_stage_a(args: argparse.Namespace) -> None:
    if str(getattr(args, "selector_method", "ridge_proxy")).lower().strip() in {
        "ridge_proxy",
        "elasticnet_proxy",
    }:
        run_stage_a_proxy(args)
        return

    # Retrofit mode: attach a schedule without running Optuna.
    if bool(getattr(args, "attach_proxy_schedule_only", False)):
        outdir = Path(args.outdir)
        dataset = make_dataset(args)
        _maybe_attach_proxy_schedule(outdir=outdir, dataset=dataset, args=args)
        return

    # Cached-only default: rely on unified Track-C parquets under cache/features.
    # No prep_families orchestration happens inside Stage A.
    if not getattr(args, "panel_cache_dir", None):
        args.panel_cache_dir = str((REPO_ROOT / "cache" / "features").resolve())

    if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
        syms = resolve_universe_symbols(args)
        missing = [
            str(sym)
            for sym in syms
            if find_unified_panel_parquet(str(sym), int(args.horizon), getattr(args, "panel_cache_dir", None)) is None
        ]
        if missing:
            raise FileNotFoundError(
                "Global pooled Stage-A requires unified per-symbol Track-C panels for all symbols (cached-only). "
                f"Missing unified panels for: {', '.join(missing)}"
            )
        LOGGER.info("🌍 Global pooled legacy Optuna: using %d cached Track-C panels", len(syms))
    else:
        if find_unified_panel_parquet(str(args.symbol), int(args.horizon), getattr(args, "panel_cache_dir", None)) is None:
            raise FileNotFoundError(
                f"Missing unified Track-C panel for {str(args.symbol).upper()} h{int(args.horizon)}; "
                "refusing to rebuild panels in cached-only mode."
            )
    if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
        dataset = make_dataset(args)
    else:
        setattr(args, "_auto_regen_attempts", set())
        regen_attempts = 0
        max_regen_attempts = max(0, getattr(args, "auto_regen_max_attempts", 0))
        while True:
            try:
                dataset = make_dataset(args)
                break
            except MissingFamilyColumnsError as exc:
                if regen_attempts >= max_regen_attempts:
                    raise
                if not maybe_auto_regen_missing_families(args, exc):
                    raise
                regen_attempts += 1
                LOGGER.info(
                    "🔁 Retrying dataset build after auto regeneration attempt %d/%d",
                    regen_attempts,
                    max_regen_attempts,
                )
    if str(getattr(args, "universe_mode", "grid")).lower().strip() == "global":
        fold_specs = build_fold_specs_by_date(
            dataset.features,
            min_train_days=int(args.min_train_window),
            val_days=int(args.fold_size),
            max_splits=int(args.folds),
        )
        LOGGER.info("🧮 Global pooled folds (date-based): %d folds", len(fold_specs))
        study = run_study_with_fold_specs(dataset, fold_specs, args)
    else:
        splitter = make_splitter(args, len(dataset.features))
        study = run_study(dataset, splitter, args)
    outdir = Path(args.outdir)
    save_trial_history(study, outdir)
    best_trial = study.best_trial
    LOGGER.info("🔁 Re-fitting best model on full dataset")
    model = fit_best_model(dataset, best_trial.params, args.seed)
    family_logits, weights, normalized_scores = build_family_weights(
        model,
        dataset,
        args.weight_temp,
        normalize_variance=args.normalize_family_variance,
    )
    stability = best_trial.user_attrs.get("stability", {})
    family_scores_attr = best_trial.user_attrs.get("family_scores") or {}
    selected = select_families(
        weights,
        dataset.family_coverage,
        args.selection_min_weight,
        args.selection_min_coverage,
        stability=stability,
        stability_alpha=args.stability_alpha,
        family_scores=family_scores_attr,
    )
    LOGGER.info("✅ Selected families (%d): %s", len(selected), ", ".join(selected))
    entropy = compute_entropy(weights)
    extras: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "fold_count": args.folds,
        "samples": int(len(dataset.features)),
        "entropy": entropy,
        "family_count": len(weights),
        "sparsity_coef": args.family_sparsity_coef,
        "study_best_value": study.best_value,
        "selected_family_count": len(selected),
        "stability": best_trial.user_attrs.get("stability", {}),
        "stability_score": best_trial.user_attrs.get("stability_score"),
        "family_scores": family_scores_attr,
        "dropped_families": best_trial.user_attrs.get("dropped_families", []),
        "hard_drops": best_trial.user_attrs.get("hard_drops", []),
        "mean_family_score": best_trial.user_attrs.get("mean_family_score"),
    }
    config_snapshot = snapshot_config(args)
    fold_normalized = best_trial.user_attrs.get("fold_normalized_scores") or []
    fold_scores = best_trial.user_attrs.get("fold_family_scores") or []
    fold_coefficients = best_trial.user_attrs.get("fold_coefficients") or []
    fold_ids = best_trial.user_attrs.get("fold_ids") or list(range(len(fold_coefficients)))
    average_family_scores = best_trial.user_attrs.get("average_family_scores", family_logits)
    stability = best_trial.user_attrs.get("stability", {})
    metrics = best_trial.user_attrs.get("metrics", {})
    full_model_coefficients = model.coef_.tolist()
    feature_columns = dataset.features.columns.tolist()
    average_feature_scores = best_trial.user_attrs.get("average_feature_scores", {})
    normalized_feature_scores = best_trial.user_attrs.get("normalized_feature_scores", {})
    fold_feature_scores = best_trial.user_attrs.get("fold_feature_scores") or []
    fold_feature_normalized = best_trial.user_attrs.get("fold_feature_normalized_scores") or []
    write_outputs(
        outdir=outdir,
        weights=weights,
        logits=family_logits,
        normalized=normalized_scores,
        average_scores=average_family_scores,
        stability=stability,
        metrics=metrics,
        extras=extras,
        best_trial=best_trial,
        selected_families=selected,
        coverage=dataset.family_coverage,
        config=config_snapshot,
        symbol=args.symbol,
        horizon=args.horizon,
        fold_normalized=fold_normalized,
        fold_scores=fold_scores,
        fold_coefficients=fold_coefficients,
        fold_ids=fold_ids,
        feature_columns=feature_columns,
        full_model_coefficients=full_model_coefficients,
        feature_metadata=dataset.feature_metadata,
        average_feature_scores=average_feature_scores,
        normalized_feature_scores=normalized_feature_scores,
        fold_feature_scores=fold_feature_scores,
        fold_feature_normalized=fold_feature_normalized,
        family_scores=family_scores_attr,
        family_feature_details=best_trial.user_attrs.get("family_feature_details") or {},
        dropped_families=best_trial.user_attrs.get("dropped_families") or [],
        hard_drops=best_trial.user_attrs.get("hard_drops") or [],
    )

    # Default behavior: emit a causal, slowly evolving Stage-A weight schedule that
    # Phase-2 can consume. This does not change the legacy Optuna selection logic;
    # it only augments the exported weights artifact.
    try:
        _maybe_attach_proxy_schedule(outdir=outdir, dataset=dataset, args=args)
    except Exception as exc:
        LOGGER.warning("⚠️ Failed to attach proxy schedule (continuing with static weights): %s", exc)


def main() -> None:
    base_args = parse_args()
    setup_logging(base_args.log_level)
    base_args.ray_parallelism = resolve_ray_parallelism(base_args.ray_parallelism)
    base_args.fold_workers = resolve_fold_workers(base_args.fold_workers)
    base_args.families = resolve_families(base_args.families)
    base_args.require_cached_families = resolve_required_cache_families(
        base_args.require_cached_families
    )
    if not base_args.wf_start:
        base_args.wf_start = base_args.start
    if not base_args.wf_end:
        base_args.wf_end = base_args.end
    if base_args.wf_step_days is None and (base_args.wf_step_years is None or base_args.wf_step_years <= 0):
        raise ValueError("Provide a positive --wf-step-years or --wf-step-days")
    set_deterministic_seed(base_args.seed)
    if str(getattr(base_args, "universe_mode", "grid")).lower().strip() == "global" and not base_args.symbols:
        raise ValueError("--universe-mode=global requires --symbols")
    run_grid = build_run_grid(base_args)
    multi_mode = len(run_grid) > 1
    LOGGER.info("🧮 Stage A run grid: %d combos", len(run_grid))
    for symbol, horizon in run_grid:
        run_args = clone_args(base_args)
        run_args.families = list(base_args.families)
        run_args.require_cached_families = list(base_args.require_cached_families)
        run_args.symbol = symbol
        run_args.horizon = horizon
        outdir_path = resolve_outdir(base_args, symbol, horizon, multi_mode)
        run_args.outdir = str(outdir_path)
        if should_skip_run(outdir_path, run_args.reuse_existing, run_args.reuse_max_age_hours):
            continue
        LOGGER.info("===== Stage A selector: %s H%d =====", symbol, horizon)
        run_stage_a(run_args)


if __name__ == "__main__":
    main()
