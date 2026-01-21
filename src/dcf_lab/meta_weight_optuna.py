"""Optuna-based training loop for meta-module weights.

This module coordinates nested cross-validation runs that couple per-fold
calibration with the meta-weight blending stage. Safeguards ensure every trial is
reproducible, logged, and ready for regime-aware extensions.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from collections import OrderedDict

import numpy as np
import optuna
import pandas as pd

from .calibration import CalibratorStore, PlattScaler
from .meta_weights import MetaWeights, project_weights_with_bounds
from .signal_bus import ModuleSignal
from .universal_aggregator import blend_module_signals, get_module_metadata
from .regime_router import RegimeRouter

# Configure parallel processing for maximum throughput
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Single thread per worker (70 workers × 1 thread)
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # NumExpr threads

logger = logging.getLogger(__name__)

WEIGHTS_BUNDLE_NAME = "weights.json"

DEFAULT_GROUP_CAPS: Mapping[str, float] = {
    "hf": 0.5,
}

DEFAULT_GROUP_PENALTIES: Mapping[str, float] = {
    "hf": 1.0,
}

HF_MODULE_NAMES: Tuple[str, ...] = (
    "doc_embedding_novelty_hf",
    "earnings_transcript_hf",
    "macro_tst_hf",
)


@dataclass
class FoldBundle:
    """Container describing a single nested CV fold."""

    fold_id: str
    symbol: str
    horizon: int
    train_signals: Sequence[ModuleSignal]
    train_returns: pd.Series
    val_signals: Sequence[ModuleSignal]
    val_returns: pd.Series
    val_prices: Optional[pd.Series] = None
    regime_label: Optional[str] = None
    calibrator_path: Optional[Path] = None
    calibrator_out: Optional[Path] = None


@dataclass
class TrialSnapshot:
    """Snapshot of a trial's proposal and resulting metrics."""

    trial_number: int
    logits: Mapping[str, float]
    weights: Mapping[str, float]
    metrics: Mapping[str, float]
    raw_score: float
    penalty: float = 0.0
    extras: Optional[Mapping[str, float]] = None

    def to_dict(self) -> Mapping[str, object]:
        payload = {
            "trial": self.trial_number,
            "logits": dict(self.logits),
            "weights": dict(self.weights),
            "metrics": dict(self.metrics),
            "raw_score": float(self.raw_score),
            "penalty": float(self.penalty),
            "score": float(self.raw_score - self.penalty),
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


METRIC_ALIASES: Mapping[str, str] = {
    "acc": "acc",
    "accuracy": "acc",
    "hit_rate": "acc",
    "sharpe": "sharpe",
    "sortino": "sortino",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "return_weighted_acc": "return_weighted_acc",
    "returnweightedacc": "return_weighted_acc",
    "rank_ic": "rank_ic",
    "rankic": "rank_ic",
    "objective": "score",
    "score": "score",
}


@dataclass(frozen=True)
class ScoreConfig:
    weights: Mapping[str, float]
    primary: str
    raw_spec: Optional[str] = None

    @classmethod
    def default(cls) -> "ScoreConfig":
        return cls(weights={"acc": 1.0, "sharpe": 0.15, "sortino": 0.10}, primary="acc", raw_spec=None)

    def compute(self, metrics: Mapping[str, float]) -> float:
        total = 0.0
        for metric, weight in self.weights.items():
            total += weight * float(metrics.get(metric, 0.0))
        return total


def _resolve_metric_name(name: str) -> Optional[str]:
    token = str(name or "").strip().lower().replace("-", "_")
    if not token:
        return None
    return METRIC_ALIASES.get(token, token if token in METRIC_ALIASES.values() else None)


def resolve_score_config(spec: Optional[str]) -> ScoreConfig:
    if not spec:
        return ScoreConfig.default()

    weights: "OrderedDict[str, float]" = OrderedDict()
    primary_metric: Optional[str] = None

    tokens = [token.strip() for token in str(spec).split(",") if token.strip()]
    fallback_weight = 0.1
    default_secondary = 0.15
    default_tertiary = 0.10

    for idx, token in enumerate(tokens):
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"primary", "secondary", "tertiary"}:
                metric = _resolve_metric_name(value)
                if metric is None:
                    continue
                if key == "primary":
                    primary_metric = metric
                    weights[metric] = 1.0
                elif key == "secondary":
                    weights[metric] = default_secondary
                else:
                    weights[metric] = default_tertiary
                continue
            metric = _resolve_metric_name(key)
            if metric is None:
                continue
            try:
                weight = float(value)
            except Exception:
                weight = fallback_weight
            if primary_metric is None:
                primary_metric = metric
            weights[metric] = weight
            continue

        if ":" in token:
            metric_name, weight_str = token.split(":", 1)
            metric = _resolve_metric_name(metric_name)
            if metric is None:
                continue
            try:
                weight = float(weight_str)
            except Exception:
                weight = fallback_weight
            if primary_metric is None:
                primary_metric = metric
            weights[metric] = weight
            continue

        metric = _resolve_metric_name(token)
        if metric is None:
            continue
        if primary_metric is None:
            primary_metric = metric
            weights[metric] = 1.0
        else:
            weights.setdefault(metric, fallback_weight)

    if not weights:
        return ScoreConfig.default()

    if primary_metric is None or primary_metric not in weights:
        primary_metric = next(iter(weights.keys()))

    return ScoreConfig(weights=dict(weights), primary=primary_metric, raw_spec=spec)


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _sanitize_param_name(module: str) -> str:
    token = str(module or "").strip()
    if not token:
        token = "module"
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in token)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"wlog_{safe}"


def _collect_module_names(folds: Sequence[FoldBundle]) -> list[str]:
    names: set[str] = set()
    for fold in folds:
        names.update(signal.name for signal in fold.train_signals)
        names.update(signal.name for signal in fold.val_signals)
    return sorted(names)


def _build_bound_maps(
    modules: Sequence[str],
    default_min: float,
    default_max: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    lower: Dict[str, float] = {}
    upper: Dict[str, float] = {}
    for module in modules:
        try:
            meta = get_module_metadata(module)
        except Exception:  # pragma: no cover - defensive
            meta = {}
        lo = float(max(0.0, meta.get("w_min", default_min)))
        hi = float(max(lo, meta.get("w_max", default_max)))
        lower[module] = lo
        upper[module] = hi
    return lower, upper


def _read_env_float(*keys: str) -> Optional[float]:
    for key in keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            logger.debug("Invalid float for env var %s: %s", key, raw)
    return None


def _collect_group_constraints(modules: Sequence[str]) -> Dict[str, dict]:
    constraints: Dict[str, dict] = {}

    for module in modules:
        try:
            meta = get_module_metadata(module)
        except Exception:  # pragma: no cover - defensive logging
            meta = {}

        if not isinstance(meta, Mapping):
            continue

        groups = meta.get("groups")
        if not groups:
            continue
        if isinstance(groups, str):
            groups = [groups]

        for group in groups:
            group_name = str(group or "").strip().lower()
            if not group_name:
                continue

            cap_value = meta.get("group_cap")
            if not isinstance(cap_value, (int, float)):
                env_keys = (
                    f"AUTO_OPT_GROUP_CAP_{group_name.upper()}",
                    f"STAGE_A_GROUP_CAP_{group_name.upper()}",
                    f"STAGEA_{group_name.upper()}_GROUP_CAP",
                )
                if group_name == "hf":
                    env_keys = (
                        "AUTO_OPT_HF_GROUP_CAP",
                        "STAGE_A_HF_GROUP_CAP",
                        *env_keys,
                    )
                cap_value = _read_env_float(*env_keys)
            if cap_value is None:
                cap_value = DEFAULT_GROUP_CAPS.get(group_name)

            penalty_value = meta.get("group_penalty")
            if not isinstance(penalty_value, (int, float)):
                env_keys = (
                    f"AUTO_OPT_GROUP_PENALTY_{group_name.upper()}",
                    f"STAGE_A_GROUP_PENALTY_{group_name.upper()}",
                    f"STAGEA_{group_name.upper()}_GROUP_PENALTY",
                )
                if group_name == "hf":
                    env_keys = (
                        "AUTO_OPT_HF_GROUP_PENALTY",
                        "STAGE_A_HF_GROUP_PENALTY",
                        *env_keys,
                    )
                penalty_value = _read_env_float(*env_keys)
            if penalty_value is None:
                penalty_value = DEFAULT_GROUP_PENALTIES.get(group_name, 0.0)

            info = constraints.setdefault(
                group_name,
                {
                    "members": [],
                    "cap": None,
                    "penalty": float(penalty_value),
                },
            )
            info["members"].append(module)

            if cap_value is None:
                continue

            cap_float = float(cap_value)
            if cap_float <= 0.0 or cap_float >= 1.0:
                continue

            if info["cap"] is None:
                info["cap"] = cap_float
            else:
                info["cap"] = min(info["cap"], cap_float)

    return constraints


def _is_hf_module(name: str) -> bool:
    token = str(name or "").strip().lower()
    if not token:
        return False
    normalized = token.replace("-", "_")
    return (
        normalized.endswith("_hf")
        or normalized.startswith("hf_")
        or "_hf_" in normalized
        or normalized == "hf"
    )


def _persist_hf_calibrators(*, fold: "FoldBundle", calibrator_store: CalibratorStore) -> None:
    target = fold.calibrator_out
    if target is None:
        return

    subset = CalibratorStore(fold_id=calibrator_store.fold_id)
    retained = 0
    for module, horizon, scaler in calibrator_store.iter_items():
        if not _is_hf_module(module):
            continue
        subset.put(module, horizon, scaler, fold_id=calibrator_store.fold_id)
        retained += 1

    if retained == 0:
        return

    payload = subset.to_dict()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(target)
    fold.calibrator_path = target
    logger.info("Persisted %d HF calibrators for fold %s → %s", retained, fold.fold_id, target)


def _fit_fold_calibrators(
    *, fold: FoldBundle, calibrator_store: CalibratorStore, label_func: Callable[[pd.Series], pd.Series]
) -> None:
    """Fit Platt scalers for every module within the fold's training window."""

    if fold.calibrator_path is None and fold.calibrator_out is not None and fold.calibrator_out.exists():
        fold.calibrator_path = fold.calibrator_out

    if fold.calibrator_path is not None:
        if not fold.calibrator_path.exists():
            logger.info(
                "Fold %s calibrator path '%s' not found; fitting calibrators from training data",
                fold.fold_id,
                fold.calibrator_path,
            )
        else:
            try:
                with fold.calibrator_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                external_store = CalibratorStore.from_dict(payload)
                items = list(external_store.iter_items())
                source_fold = external_store.fold_id if external_store.fold_id else fold.fold_id
                for module, horizon, scaler in items:
                    calibrator_store.put(module, horizon, scaler, fold_id=source_fold)
                calibrator_store.assert_fold(source_fold or fold.fold_id)
                logger.info(
                    "Loaded %d external calibrators for fold %s from %s",
                    len(items),
                    fold.fold_id,
                    fold.calibrator_path,
                )
                return
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to load calibrators for fold %s from '%s': %s. Falling back to on-the-fly fitting.",
                    fold.fold_id,
                    fold.calibrator_path,
                    exc,
                )

    returns = label_func(fold.train_returns)

    for signal in fold.train_signals:
        aligned_index = signal.df.index.intersection(returns.index)
        if aligned_index.empty:
            logger.debug("Fold %s module %s has no overlap for calibration", fold.fold_id, signal.name)
            continue

        y = returns.loc[aligned_index]
        x = signal.df.loc[aligned_index, "score"]
        try:
            scaler = PlattScaler().fit(x, y)
        except ValueError as exc:
            logger.debug(
                "Platt fit skipped for fold %s module %s horizon %s: %s",
                fold.fold_id,
                signal.name,
                signal.horizon,
                exc,
            )
            continue

        calibrator_store.put(signal.name, signal.horizon, scaler, fold_id=fold.fold_id)

    logger.debug("Fitted %d calibrators for fold %s", len(list(calibrator_store.iter_items())), fold.fold_id)

    _persist_hf_calibrators(fold=fold, calibrator_store=calibrator_store)


def _apply_calibrators_to_signals(
    *, calibrator_store: CalibratorStore, fold_id: str, signals: Sequence[ModuleSignal]
) -> Sequence[ModuleSignal]:
    """Apply per-module Platt scalers to the provided signals."""

    calibrated: list[ModuleSignal] = []

    for signal in signals:
        cloned = ModuleSignal(
            name=signal.name,
            horizon=signal.horizon,
            df=signal.df.copy(),
            symbol=signal.symbol,
        )
        scaler = calibrator_store.get(signal.name, signal.horizon, fold_id=fold_id)
        if scaler is not None:
            proba = scaler.transform(cloned.df["score"].values, return_proba=True)
            cloned.df["score"] = 2.0 * proba - 1.0
        calibrated.append(cloned)

    return calibrated


def _derive_labels(raw_returns: pd.Series) -> pd.Series:
    """Default label function turning returns into binary outcomes."""

    return (raw_returns > 0).astype(int)


def _ensure_price_series(returns: pd.Series, prices: Optional[pd.Series]) -> pd.Series:
    if prices is not None and not prices.dropna().empty:
        return prices.dropna()

    reindexed_returns = returns.fillna(0.0)
    cumulative = (1.0 + reindexed_returns).cumprod()
    start_price = 1.0
    return cumulative * start_price


def _infer_regime(prices: pd.Series) -> str:
    router = RegimeRouter(hysteresis_days=0)
    return router.detect(prices)


def _compute_fold_metrics(meta_score: pd.Series, returns: pd.Series) -> Mapping[str, float]:
    # Forward-fill returns to match meta_score index for sparse data tolerance
    aligned_returns = returns.reindex(meta_score.index).ffill().fillna(0.0)
    
    # FAIL-FAST GUARD: Alignment must be substantial
    # 🔧 FIXED: Reduced to 10 for walk-forward with sparse recent data
    MIN_ALIGNED_ROWS = 10  # Minimum required aligned data points (tolerates sparse signals)
    if len(aligned_returns) < MIN_ALIGNED_ROWS:
        raise RuntimeError(
            f"Alignment failed: only {len(aligned_returns)} aligned rows "
            f"(minimum required: {MIN_ALIGNED_ROWS}). "
            f"meta_score range: {meta_score.index.min()} to {meta_score.index.max()}, "
            f"returns range: {returns.index.min()} to {returns.index.max()}. "
            "Check date windows in config/date_ranges.yaml and regenerate signals/returns."
        )
    
    if aligned_returns.empty:
        return {"acc": 0.0, "sharpe": 0.0, "sortino": 0.0}

    # Directional hit rate
    hits = np.sign(meta_score) * np.sign(aligned_returns) > 0
    acc = float(hits.mean()) if hits.size else 0.0

    pnl = meta_score.clip(-1.0, 1.0) * aligned_returns
    mean_pnl = pnl.mean()
    std_pnl = pnl.std(ddof=1)
    sharpe = float(mean_pnl / std_pnl) if std_pnl > 1e-9 else 0.0

    downside = pnl[pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(mean_pnl / downside_std) if downside_std > 1e-9 else 0.0

    return {"acc": acc, "sharpe": sharpe, "sortino": sortino}


def _detect_regime(returns: pd.Series) -> str:
    """
    Classify market regime based on returns characteristics.
    
    Args:
        returns: Series of forward returns for a time period
        
    Returns:
        'bull', 'bear', or 'sideways'
    """
    if returns.empty:
        return 'sideways'
    
    # Calculate regime indicators
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    # Annualized Sharpe-like metric
    sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0
    
    # Classify regime
    if sharpe > 0.5:
        return 'bull'
    elif sharpe < -0.3:
        return 'bear'
    else:
        return 'sideways'


def _score_from_metrics(metrics: Mapping[str, float], score_config: Optional[ScoreConfig] = None) -> float:
    config = score_config or ScoreConfig.default()
    return float(config.compute(metrics))


def _build_payload(
    *,
    logits: Mapping[str, float],
    weights: Mapping[str, float],
    metrics: Mapping[str, float],
    score: float,
    raw_score: Optional[float] = None,
    penalty: Optional[float] = None,
    extras: Optional[Mapping[str, float]] = None,
) -> Mapping[str, object]:
    payload = {
        "logits": dict(logits),
        "weights": dict(weights),
        "metrics": dict(metrics),
        "score": float(score),
    }
    if raw_score is not None:
        payload["raw_score"] = float(raw_score)
    if penalty is not None:
        payload["penalty"] = float(penalty)
    if extras:
        payload["extras"] = dict(extras)
    return payload


def _signals_to_matrix(
    signals: Sequence[ModuleSignal],
    modules: Sequence[str],
    target_index: pd.Index,
) -> np.ndarray:
    if not isinstance(target_index, pd.Index):
        target_index = pd.Index(target_index)
    if target_index.empty:
        return np.zeros((0, len(modules)), dtype=float)

    data = np.zeros((len(target_index), len(modules)), dtype=float)
    signal_map: Dict[str, ModuleSignal] = {signal.name: signal for signal in signals}

    for col_idx, module in enumerate(modules):
        signal = signal_map.get(module)
        if signal is None or signal.df is None or "score" not in signal.df.columns:
            continue
        series = signal.df["score"].reindex(target_index).fillna(0.0)
        data[:, col_idx] = series.to_numpy(dtype=float, copy=False)

    return data


def _standardize_features(
    train: np.ndarray, val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train.size == 0:
        cols = train.shape[1] if train.ndim == 2 else 0
        mean = np.zeros(cols, dtype=float)
        std = np.ones(cols, dtype=float)
        return train, val, mean, std

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (val - mean) / std, mean, std


def train_meta_mlp(
    *,
    folds: Sequence[FoldBundle],
    outdir: Path,
    w_min: float = 0.0,
    w_max: float = 0.40,
    label_func: Callable[[pd.Series], pd.Series] = _derive_labels,
    dropout: Optional[float] = None,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    normalize_weights: bool = True,
    score_config: Optional[ScoreConfig] = None,
    hf_weight_scale: Optional[float] = None,
    cv_folds: int = 1,
    sparsity_type: Optional[str] = None,
    sparsity_lambda: Optional[float] = None,
) -> Mapping[str, object]:
    if not folds:
        raise ValueError("train_meta_mlp requires at least one fold")

    try:
        import torch  # type: ignore[import]
        from torch import nn  # type: ignore[import]
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import]
        
        # Configure PyTorch to use all available threads
        torch.set_num_threads(80)  # Intra-op parallelism (matrix ops)
        torch.set_num_interop_threads(40)  # Inter-op parallelism (operations)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Stage A MLP meta-combiner requires PyTorch to be installed") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modules = _collect_module_names(folds)
    if not modules:
        raise ValueError("No module signals available for Stage A MLP")

    outdir.mkdir(parents=True, exist_ok=True)

    w_min_map_global, w_max_map_global = _build_bound_maps(modules, w_min, w_max)
    group_constraints = _collect_group_constraints(modules)
    score_cfg = score_config or ScoreConfig.default()
    dropout_value = float(dropout) if dropout is not None else 0.0
    hidden_dim = max(8, len(modules) * 2)

    class MetaMLP(nn.Module):  # pragma: no cover - simple feed-forward network
        def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x).squeeze(-1)

    fold_metrics: list[Mapping[str, float]] = []
    importance_acc = np.zeros(len(modules), dtype=float)
    all_train_features: list[np.ndarray] = []
    all_train_labels: list[np.ndarray] = []
    
    # 🔧 FIX #18: Load enhanced training data from Stage F (if available)
    enhanced_samples = None
    enhanced_weights = None
    if outdir.exists():
        enhanced_path = outdir.parent / "stage_f" / "entry_models" / "training_data_enhanced.csv"
        if enhanced_path.exists():
            try:
                logger.info(f"🎯 [FIX #18] Loading enhanced training data: {enhanced_path}")
                enhanced_df = pd.read_csv(enhanced_path)
                logger.info(f"   📊 Enhanced data: {len(enhanced_df)} samples with {len(enhanced_df.columns)} features")
                
                # Extract sample weights if available
                if "capped_weight" in enhanced_df.columns:
                    enhanced_weights = enhanced_df["capped_weight"].values
                    logger.info(f"   ⚖️  Using sample weights (min={enhanced_weights.min():.2f}, max={enhanced_weights.max():.2f}, mean={enhanced_weights.mean():.2f})")
                else:
                    enhanced_weights = np.ones(len(enhanced_df))
                
                enhanced_samples = enhanced_df
                logger.info(f"   ✅ Enhanced training data loaded successfully")
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to load enhanced data: {e}")
                enhanced_samples = None
                enhanced_weights = None

    for fold in folds:
        calibrator_store = CalibratorStore(fold_id=fold.fold_id)
        _fit_fold_calibrators(
            fold=fold,
            calibrator_store=calibrator_store,
            label_func=label_func,
        )

        calibrated_train_signals = _apply_calibrators_to_signals(
            calibrator_store=calibrator_store,
            fold_id=fold.fold_id,
            signals=fold.train_signals,
        )
        calibrated_val_signals = _apply_calibrators_to_signals(
            calibrator_store=calibrator_store,
            fold_id=fold.fold_id,
            signals=fold.val_signals,
        )

        train_index = fold.train_returns.index
        val_index = fold.val_returns.index
        if train_index.empty or val_index.empty:
            continue

        X_train_raw = _signals_to_matrix(calibrated_train_signals, modules, train_index)
        X_val_raw = _signals_to_matrix(calibrated_val_signals, modules, val_index)
        if X_train_raw.shape[0] < 10:
            continue

        y_train_series = label_func(fold.train_returns).reindex(train_index).fillna(0.0)
        y_train = y_train_series.to_numpy(dtype=float)

        X_train_std, X_val_std, _, _ = _standardize_features(X_train_raw, X_val_raw)

        dataset = TensorDataset(
            torch.from_numpy(X_train_std).float(),
            torch.from_numpy(y_train.astype(np.float32)),
        )
        batch_size = min(256, max(1, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = MetaMLP(len(modules), hidden_dim, dropout_value).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Compute sparsity penalty lambda
        sparse_lambda = float(sparsity_lambda) if sparsity_lambda is not None else 0.0
        sparse_type = str(sparsity_type or "").lower() if sparsity_type else None

        for _ in range(max(10, epochs)):
            model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                # Add sparsity regularization
                if sparse_lambda > 0.0 and sparse_type:
                    fc1_weights = model.fc1.weight
                    if sparse_type == "l1":
                        sparsity_penalty = sparse_lambda * torch.sum(torch.abs(fc1_weights))
                    elif sparse_type == "l2":
                        sparsity_penalty = sparse_lambda * torch.sum(fc1_weights ** 2)
                    elif sparse_type == "elastic":
                        l1_penalty = torch.sum(torch.abs(fc1_weights))
                        l2_penalty = torch.sum(fc1_weights ** 2)
                        sparsity_penalty = sparse_lambda * (0.5 * l1_penalty + 0.5 * l2_penalty)
                    else:
                        sparsity_penalty = 0.0
                    loss = loss + sparsity_penalty
                
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_tensor = torch.from_numpy(X_val_std).float().to(device)
            logits_val = model(val_tensor)
            probs_val = torch.sigmoid(logits_val).cpu().numpy()

        meta_scores = 2.0 * probs_val - 1.0
        meta_series = pd.Series(meta_scores, index=val_index, name="meta_score")
        metrics = _compute_fold_metrics(meta_series, fold.val_returns)
        fold_metrics.append(metrics)

        first_layer = model.fc1.weight.detach().cpu().numpy()
        importance_acc += np.abs(first_layer).mean(axis=0)

        all_train_features.append(X_train_raw)
        all_train_labels.append(y_train)

        _persist_hf_calibrators(fold=fold, calibrator_store=calibrator_store)

    aggregate_keys: set[str] = set()
    for metrics in fold_metrics:
        aggregate_keys.update(metrics.keys())

    aggregated_metrics = (
        {key: float(np.mean([metrics.get(key, 0.0) for metrics in fold_metrics])) for key in aggregate_keys}
        if fold_metrics
        else {"acc": 0.0, "sharpe": 0.0, "sortino": 0.0}
    )

    if np.allclose(importance_acc.sum(), 0.0):
        raw_importance = {module: 1.0 for module in modules}
    else:
        raw_importance = {
            module: float(importance_acc[idx] / max(1, len(fold_metrics)))
            for idx, module in enumerate(modules)
        }

    importance_adjusted = dict(raw_importance)
    hf_scale_value = float(hf_weight_scale) if hf_weight_scale is not None else None
    if hf_scale_value is not None:
        for module in importance_adjusted:
            if _is_hf_module(module):
                importance_adjusted[module] = max(0.0, importance_adjusted[module]) * hf_scale_value

    target_sum = 1.0 if normalize_weights else None
    weight_map = project_weights_with_bounds(
        importance_adjusted,
        w_min_map_global,
        w_max_map_global,
        target_sum=target_sum,
    )

    total_weight = sum(max(0.0, weight) for weight in weight_map.values())
    normalized_weights = (
        {module: max(0.0, weight_map[module]) / total_weight for module in modules}
        if total_weight > 1e-12
        else {module: 1.0 / len(modules) for module in modules}
    )

    group_penalty = 0.0
    for group, info in group_constraints.items():
        members = info.get("members", [])
        if not members:
            continue
        group_weight = sum(normalized_weights.get(member, 0.0) for member in members)
        cap_value = info.get("cap")
        penalty_weight = float(info.get("penalty", 0.0))
        if cap_value is None or cap_value <= 0.0:
            continue
        if group_weight <= float(cap_value) + 1e-6:
            continue
        overshoot = group_weight - float(cap_value)
        basis = overshoot / float(cap_value) if float(cap_value) > 0 else overshoot
        group_penalty += (penalty_weight if penalty_weight > 0.0 else 1.0) * basis

    hf_share = sum(normalized_weights.get(name, 0.0) for name in HF_MODULE_NAMES)
    entropy = -sum(weight * log(max(weight, 1e-12)) for weight in normalized_weights.values())
    diversity_penalty = max(0.0, hf_share - 0.50) * 2.0 + (-0.01 * entropy)
    penalty_total = group_penalty + diversity_penalty

    raw_score = _score_from_metrics(aggregated_metrics, score_cfg)
    penalized_score = raw_score - penalty_total

    model_path: Optional[Path] = None
    if all_train_features:
        X_all = np.vstack(all_train_features)
        y_all = np.concatenate(all_train_labels)
        mean_vec = X_all.mean(axis=0)
        std_vec = X_all.std(axis=0)
        std_vec = np.where(std_vec < 1e-6, 1.0, std_vec)
        X_all_std = (X_all - mean_vec) / std_vec

        dataset_all = TensorDataset(
            torch.from_numpy(X_all_std).float(),
            torch.from_numpy(y_all.astype(np.float32)),
        )
        loader_all = DataLoader(dataset_all, batch_size=min(512, max(1, len(dataset_all))), shuffle=True)

        final_model = MetaMLP(len(modules), hidden_dim, dropout_value).to(device)
        optimizer_all = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion_all = nn.BCEWithLogitsLoss()

        for _ in range(max(10, epochs)):
            final_model.train()
            for batch_x, batch_y in loader_all:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer_all.zero_grad()
                logits = final_model(batch_x)
                loss = criterion_all(logits, batch_y)
                
                # Add sparsity regularization to final model
                if sparse_lambda > 0.0 and sparse_type:
                    fc1_weights = final_model.fc1.weight
                    if sparse_type == "l1":
                        sparsity_penalty = sparse_lambda * torch.sum(torch.abs(fc1_weights))
                    elif sparse_type == "l2":
                        sparsity_penalty = sparse_lambda * torch.sum(fc1_weights ** 2)
                    elif sparse_type == "elastic":
                        l1_penalty = torch.sum(torch.abs(fc1_weights))
                        l2_penalty = torch.sum(fc1_weights ** 2)
                        sparsity_penalty = sparse_lambda * (0.5 * l1_penalty + 0.5 * l2_penalty)
                    else:
                        sparsity_penalty = 0.0
                    loss = loss + sparsity_penalty
                
                loss.backward()
                optimizer_all.step()

        model_path = outdir / "meta_mlp_model.pt"
        torch.save(
            {
                "state_dict": final_model.state_dict(),
                "modules": modules,
                "mean": mean_vec,
                "std": std_vec,
                "dropout": dropout_value,
            },
            model_path,
        )

    logits_map = {
        module: float(np.log(max(weight_map.get(module, 1e-12), 1e-12)))
        for module in modules
    }

    extras = {
        "group_penalty": float(group_penalty),
        "diversity_penalty": float(diversity_penalty),
        "hf_share": float(hf_share),
        "entropy": float(entropy),
        "weight_sum": float(total_weight),
        "model_type": "mlp",
        "mlp_dropout": float(dropout_value),
        "mlp_hidden_dim": int(hidden_dim),
        "normalize_weights": bool(normalize_weights),
        "fold_metrics": [dict(metric) for metric in fold_metrics],
        "fold_count": int(len(folds)),
        "cv_folds_requested": int(max(1, cv_folds)),
    }
    if model_path is not None:
        extras["mlp_model_path"] = str(model_path)
    if score_cfg.raw_spec:
        extras["score_spec"] = score_cfg.raw_spec
    if hf_scale_value is not None:
        extras["hf_weight_scale"] = float(hf_scale_value)
    if sparse_type is not None:
        extras["sparsity_type"] = str(sparse_type)
    if sparse_lambda is not None:
        extras["sparsity_lambda"] = float(sparse_lambda)

    payload = _build_payload(
        logits=logits_map,
        weights=weight_map,
        metrics=aggregated_metrics,
        score=penalized_score,
        raw_score=raw_score,
        penalty=penalty_total,
        extras=extras,
    )
    payload["model_type"] = "mlp"

    bundle: Dict[str, object] = {
        "global": payload,
        "weights_by_regime": {},
    }
    if model_path is not None:
        bundle["meta_model"] = {
            "type": "mlp",
            "path": str(model_path),
            "modules": modules,
            "dropout": dropout_value,
        }

    return bundle


def run_nested_cv_with_logits(
    *,
    folds: Sequence[FoldBundle],
    logits: Mapping[str, float],
    w_min: float = 0.02,
    w_max: float = 0.40,
    w_min_map: Optional[Mapping[str, float]] = None,
    w_max_map: Optional[Mapping[str, float]] = None,
    label_func: Callable[[pd.Series], pd.Series] = _derive_labels,
    normalize_weights: bool = True,
) -> Mapping[str, float]:
    """Evaluate a logits proposal across nested CV folds."""
    fold_metrics = []
    for fold in folds:
        calibrator_store = CalibratorStore(fold_id=fold.fold_id)
        
        _fit_fold_calibrators(
            fold=fold,
            store=calibrator_store,
            module_signals=fold.train_signals,
            module_returns=fold.train_returns,
        )
        
        calibrated_val_signals = _apply_calibrators_to_signals(
            fold=fold,
            module_signals=fold.val_signals,
            store=calibrator_store,
        )
        
        blended = blend_module_signals(
            signals=calibrated_val_signals,
            logits=logits,
            w_min=w_min,
            w_max=w_max,
            w_min_map=w_min_map,
            w_max_map=w_max_map,
            normalize=normalize_weights,
        )
        
        meta_score = blended["meta_score"].reindex(fold.val_returns.index).dropna()
        metrics = _compute_fold_metrics(meta_score, fold.val_returns)
        fold_metrics.append(metrics)
    
    if not fold_metrics:
        return {"acc": 0.0, "sharpe": 0.0, "sortino": 0.0}

    aggregate_keys: set[str] = set()
    for metrics in fold_metrics:
        aggregate_keys.update(metrics.keys())

    aggregated = {
        key: float(np.mean([metrics.get(key, 0.0) for metrics in fold_metrics]))
        for key in aggregate_keys
    }
    
    return aggregated


def run_nested_cv_with_logits_regime_aware(
    *,
    folds: Sequence[FoldBundle],
    logits: Mapping[str, float],
    regime_logits_map: Optional[Mapping[str, Mapping[str, float]]] = None,
    w_min: float = 0.02,
    w_max: float = 0.40,
    w_min_map: Optional[Mapping[str, float]] = None,
    w_max_map: Optional[Mapping[str, float]] = None,
    label_func: Callable[[pd.Series], pd.Series] = _derive_labels,
    normalize_weights: bool = True,
    regime_labels: Optional[Mapping[str, str]] = None,
) -> Mapping[str, float]:
    """
    Evaluate logits with regime-aware blending and aggregation.
    
    This function:
    1. Uses regime-specific weights for each fold based on its regime label
    2. Passes regime detection parameters to blend_module_signals
    3. Groups results by regime and uses worst-case (minimum) aggregation
    
    Args:
        folds: Sequence of fold bundles with regime labels
        logits: Global logits (fallback)
        regime_logits_map: Dict mapping regime name -> logits dict (e.g., {"high_bull": {...}, ...})
        regime_labels: Optional override for fold regime labels
        
    Returns:
        Aggregated metrics using worst-case across regimes
    """
    from collections import defaultdict
    
    # Group folds by regime
    regime_folds: Dict[str, List[Tuple[FoldBundle, Mapping[str, float]]]] = defaultdict(list)
    
    # Log regime detection status
    has_regime_logits = bool(regime_logits_map)
    regime_fold_count = sum(1 for f in folds if f.regime_label is not None)
    logger.info(f"✅ REGIME-AWARE OPTIMIZATION: regime_logits={'enabled' if has_regime_logits else 'disabled'}, "
                f"labeled_folds={regime_fold_count}/{len(folds)}")
    
    for fold in folds:
        # Compute metrics for this fold
        calibrator_store = CalibratorStore(fold_id=fold.fold_id)
        _fit_fold_calibrators(
            fold=fold, calibrator_store=calibrator_store, label_func=label_func
        )
        
        calibrated_val_signals = _apply_calibrators_to_signals(
            calibrator_store=calibrator_store,
            fold_id=fold.fold_id,
            signals=fold.val_signals,
        )
        
        # Determine regime and logits for this fold
        if regime_labels and fold.fold_id in regime_labels:
            regime = regime_labels[fold.fold_id]
        elif fold.regime_label:
            regime = fold.regime_label
        else:
            regime = _detect_regime(fold.val_returns)
        
        # Select regime-specific logits if available
        if regime_logits_map and regime in regime_logits_map:
            fold_logits = regime_logits_map[regime]
        else:
            fold_logits = logits  # Fallback to global
        
        # NEW: Pass regime parameters to blend_module_signals
        blended = blend_module_signals(
            calibrated_val_signals,
            logits=fold_logits,
            regime_logits=regime_logits_map,
            router=RegimeRouter(hysteresis_days=10) if has_regime_logits else None,
            price_history=fold.val_prices,
            w_min=w_min,
            w_max=w_max,
            w_min_map=w_min_map,
            w_max_map=w_max_map,
            symbol=fold.symbol,
            horizon=fold.horizon,
            normalize_weights=normalize_weights,
        )
        
        meta_score = blended["meta_score"].reindex(fold.val_returns.index).dropna()
        metrics = _compute_fold_metrics(meta_score, fold.val_returns)
        
        # Store fold and metrics by regime
        regime_folds[regime].append((fold, metrics))
    
    if not regime_folds:
        return {"acc": 0.0, "sharpe": 0.0, "sortino": 0.0}
    
    # Aggregate metrics by regime
    aggregate_keys: set[str] = set()
    for regime_list in regime_folds.values():
        for _, metrics in regime_list:
            aggregate_keys.update(metrics.keys())
    
    # Calculate regime averages
    regime_averages: Dict[str, Mapping[str, float]] = {}
    for regime, fold_metrics_list in regime_folds.items():
        if not fold_metrics_list:
            continue
        
        regime_avg = {
            key: float(np.mean([metrics.get(key, 0.0) for _, metrics in fold_metrics_list]))
            for key in aggregate_keys
        }
        regime_averages[regime] = regime_avg
    
    # Use MINIMUM across regimes (worst-case optimization)
    aggregated = {}
    for key in aggregate_keys:
        regime_scores = [
            regime_avg.get(key, 0.0)
            for regime_avg in regime_averages.values()
            if regime_avg.get(key, 0.0) != 0.0
        ]
        
        if regime_scores:
            # Use minimum for robustness (worst-case)
            aggregated[key] = float(np.min(regime_scores))
        else:
            aggregated[key] = 0.0
    
    # Log regime breakdown for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Regime-aware evaluation:")
        for regime, avg_metrics in regime_averages.items():
            fold_count = len(regime_folds[regime])
            logger.debug(f"  {regime} ({fold_count} folds): acc={avg_metrics.get('acc', 0):.3f}")
        logger.debug(f"  Worst-case aggregated: acc={aggregated.get('acc', 0):.3f}")
    
    return aggregated


def optimize_meta_weights_by_regime(
    *,
    folds: Sequence[FoldBundle],
    outdir: Path,
    w_min: float = 0.02,
    w_max: float = 0.40,
    seed: int = 42,
    study_name: Optional[str] = None,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    normalize_weights: bool = True,
    score_config: Optional[ScoreConfig] = None,
    hf_weight_scale: Optional[float] = None,
) -> Mapping[str, optuna.study.Study]:
    """Run Optuna sweeps for each volatility/trend regime subset."""

    # Include all 6 possible regimes (2 volatility × 3 trend states)
    regimes = [
        "high_bull", "high_bear", "high_sideways",
        "low_bull", "low_bear", "low_sideways"
    ]
    results: dict[str, optuna.study.Study] = {}
    outdir.mkdir(parents=True, exist_ok=True)

    global_study = optimize_meta_weights(
        folds=folds,
        outdir=outdir / "global",
        w_min=w_min,
        w_max=w_max,
        seed=seed,
        study_name=f"{study_name or 'meta-weights'}_global",
        n_trials=n_trials,
        timeout=timeout,
        normalize_weights=normalize_weights,
        score_config=score_config,
        hf_weight_scale=hf_weight_scale,
    )
    results["global"] = global_study
    global_payload = global_study.user_attrs.get("best_payload", {})

    weights_by_regime: dict[str, Mapping[str, object]] = {}
    
    # Log regime optimization strategy
    logger.info(f"🎯 REGIME-AWARE OPTIMIZATION: Optimizing {len(regimes)} regimes independently")
    regime_distribution = {}
    for regime in regimes:
        count = sum(1 for fold in folds if fold.regime_label == regime or _infer_regime(_ensure_price_series(fold.val_returns, fold.val_prices)) == regime)
        regime_distribution[regime] = count
    logger.info(f"   Fold distribution: {regime_distribution}")

    for regime in regimes:
        # 🔧 REGIME-AWARE FIX: Use ALL walk-forward windows for each regime optimization
        # Don't filter by regime - let the optimizer weight samples appropriately
        # This ensures ALL 6 regimes get optimized even if some have few/no matching windows
        regime_folds: list[FoldBundle] = list(folds)
        
        # Count how many folds actually match this regime (for logging only)
        matching_count = 0
        for fold in folds:
            # Use fold.regime_label if available, otherwise infer
            if fold.regime_label:
                fold_regime = fold.regime_label
            else:
                price_series = _ensure_price_series(fold.val_returns, fold.val_prices)
                fold_regime = _infer_regime(price_series)
            
            if fold_regime == regime:
                matching_count += 1

        logger.info(
            f"🎯 Optimizing regime '{regime}' using ALL {len(regime_folds)} walk-forward windows "
            f"({matching_count} naturally match this regime)"
        )

        study = optimize_meta_weights(
            folds=regime_folds,
            outdir=outdir / regime,
            w_min=w_min,
            w_max=w_max,
            seed=seed,
            study_name=f"{study_name or 'meta-weights'}_{regime}",
            n_trials=n_trials,
            timeout=timeout,
            normalize_weights=normalize_weights,
            score_config=score_config,
            hf_weight_scale=hf_weight_scale,
            regime_aware=False,  # Don't double-apply regime logic
        )

        payload = study.user_attrs.get("best_payload", {})
        if payload:
            weights_by_regime[regime] = payload
        results[regime] = study

    bundle = {
        "global": global_payload,
        "weights_by_regime": weights_by_regime,
    }
    save_json(outdir / WEIGHTS_BUNDLE_NAME, bundle)

    return results


def optimize_meta_weights(
    *,
    folds: Sequence[FoldBundle],
    outdir: Path,
    w_min: float = 0.02,
    w_max: float = 0.40,
    seed: int = 42,
    study_name: Optional[str] = None,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    normalize_weights: bool = True,
    score_config: Optional[ScoreConfig] = None,
    hf_weight_scale: Optional[float] = None,
    regime_aware: bool = True,  # NEW: Enable regime-aware optimization by default
) -> optuna.study.Study:
    """Run an Optuna sweep over meta-weight logits.

    All trials are reproducible (seeded) and create per-trial artifacts capturing
    logits, derived weights, and evaluation metrics.
    """

    if not folds:
        raise ValueError("optimize_meta_weights requires at least one fold")

    modules = _collect_module_names(folds)
    hf_scale_value = float(hf_weight_scale) if hf_weight_scale is not None else None
    w_min_map_global, w_max_map_global = _build_bound_maps(modules, w_min, w_max)
    group_constraints = _collect_group_constraints(modules)
    outdir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    
    # Use RAM-based SQLite storage (/dev/shm) for fast parallel execution
    # This eliminates disk IO bottleneck while allowing n_jobs parallelism
    import tempfile
    shm_dir = "/dev/shm/optuna_studies"
    os.makedirs(shm_dir, exist_ok=True)
    storage_url = f"sqlite:///{shm_dir}/optuna_{study_name.replace('/', '_')}.db"
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,  # Allow parallel workers to share study
    )

    effective_score_config = score_config or ScoreConfig.default()

    if len(modules) == 1:
        module = modules[0]
        logits = {module: 0.0}
        single_min = {module: w_min_map_global.get(module, w_min)}
        single_max = {module: w_max_map_global.get(module, w_max)}
        metrics = run_nested_cv_with_logits(
            folds=folds,
            logits=logits,
            w_min=w_min,
            w_max=w_max,
            w_min_map=single_min,
            w_max_map=single_max,
            normalize_weights=normalize_weights,
        )
        weights_model = MetaWeights(
            [module],
            w_min=w_min,
            w_max=w_max,
            w_min_map=single_min,
            w_max_map=single_max,
            normalize=normalize_weights,
        )
        weights_model.set_from_dict(logits)
        try:
            weights = weights_model.softmax()
        except ValueError:
            weights = {module: 1.0}

        if hf_scale_value is not None:
            scaled_weights = {
                name: value * (hf_scale_value if _is_hf_module(name) else 1.0)
                for name, value in weights.items()
            }
            weights = project_weights_with_bounds(
                scaled_weights,
                single_min,
                single_max,
                target_sum=1.0 if normalize_weights else None,
            )

        total_weight_single = sum(max(0.0, w) for w in weights.values())
        norm_weights_single = (
            {k: max(0.0, v) / total_weight_single for k, v in weights.items()}
            if total_weight_single > 1e-12
            else {module: 1.0}
        )

        score = _score_from_metrics(metrics, effective_score_config)
        hf_share_single = sum(norm_weights_single.get(name, 0.0) for name in HF_MODULE_NAMES)
        entropy_single = -sum(weight * log(max(weight, 1e-12)) for weight in norm_weights_single.values())
        diversity_penalty_single = max(0.0, hf_share_single - 0.50) * 2.0 + (-0.01 * entropy_single)
        extras_single = {
            "hf_share": float(hf_share_single),
            "entropy": float(entropy_single),
            "group_penalty": 0.0,
            "diversity_penalty": float(diversity_penalty_single),
            "weight_sum": float(total_weight_single),
            "fold_count": len(folds),
        }
        if hf_scale_value is not None:
            extras_single["hf_weight_scale"] = float(hf_scale_value)
        snapshot = TrialSnapshot(
            trial_number=0,
            logits=logits,
            weights=weights,
            metrics=metrics,
            raw_score=score,
            penalty=0.0,
            extras=extras_single,
        )
        save_json(outdir / "trial_0000.json", snapshot.to_dict())

        frozen = optuna.trial.create_trial(
            params={},
            distributions={},
            value=score,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={
                "logits": logits,
                "weights": weights,
                "metrics": metrics,
                "raw_score": score,
                "penalty": 0.0,
                "extras": extras_single,
            },
        )
        study.add_trial(frozen)

        payload = _build_payload(
            logits=logits,
            weights=weights,
            metrics=metrics,
            score=score,
            raw_score=score,
            penalty=0.0,
            extras=extras_single,
        )
        study.set_user_attr("best_payload", payload)
        save_json(outdir / WEIGHTS_BUNDLE_NAME, {"global": payload, "weights_by_regime": {}})
        logger.info(
            "Only one module available; skipping optimization and using deterministic weights"
        )
        return study

    param_names: Dict[str, str] = {}
    seen_params: set[str] = set()
    for module in modules:
        base = _sanitize_param_name(module)
        candidate = base
        counter = 1
        while candidate in seen_params:
            candidate = f"{base}_{counter}"
            counter += 1
        seen_params.add(candidate)
        param_names[module] = candidate

    objective_w_min_map = {module: w_min_map_global.get(module, w_min) for module in modules}
    objective_w_max_map = {module: w_max_map_global.get(module, w_max) for module in modules}

    def objective(trial: optuna.Trial) -> float:
        logits = {
            module: trial.suggest_float(param_names[module], -2.5, 2.5)
            for module in modules
        }

        # Use regime-aware CV if enabled (default), otherwise use standard CV
        if regime_aware:
            metrics = run_nested_cv_with_logits_regime_aware(
                folds=folds,
                logits=logits,
                w_min=w_min,
                w_max=w_max,
                w_min_map=objective_w_min_map,
                w_max_map=objective_w_max_map,
                normalize_weights=normalize_weights,
            )
        else:
            metrics = run_nested_cv_with_logits(
                folds=folds,
                logits=logits,
                w_min=w_min,
                w_max=w_max,
                w_min_map=objective_w_min_map,
                w_max_map=objective_w_max_map,
                normalize_weights=normalize_weights,
            )

        weights = MetaWeights(
            modules,
            w_min=w_min,
            w_max=w_max,
            w_min_map=objective_w_min_map,
            w_max_map=objective_w_max_map,
            normalize=normalize_weights,
        )
        weights.set_from_dict(logits)
        try:
            weight_map = weights.softmax()
        except ValueError:
            weight_map = {module: 1.0 / len(modules) for module in modules}

        if hf_scale_value is not None:
            scaled_weights = {
                name: weight_map.get(name, 0.0) * (hf_scale_value if _is_hf_module(name) else 1.0)
                for name in modules
            }
            weight_map = project_weights_with_bounds(
                scaled_weights,
                objective_w_min_map,
                objective_w_max_map,
                target_sum=1.0 if normalize_weights else None,
            )

        total_weight = sum(max(0.0, weight) for weight in weight_map.values())
        if total_weight > 1e-12:
            normalized_weights = {
                module: max(0.0, weight_map.get(module, 0.0)) / total_weight
                for module in modules
            }
        else:
            normalized_weights = {module: 1.0 / len(modules) for module in modules}

        raw_score = _score_from_metrics(metrics, effective_score_config)
        group_penalty = 0.0
        group_totals: Dict[str, float] = {}
        for group, info in group_constraints.items():
            members = info.get("members", [])
            if not members:
                continue
            group_weight = sum(normalized_weights.get(member, 0.0) for member in members)
            group_totals[group] = group_weight
            cap_value = info.get("cap")
            penalty_weight = float(info.get("penalty", 0.0))
            if cap_value is None or cap_value <= 0.0:
                continue
            if group_weight <= float(cap_value) + 1e-6:
                continue
            overshoot = group_weight - float(cap_value)
            basis = overshoot / float(cap_value) if float(cap_value) > 0 else overshoot
            group_penalty += (penalty_weight if penalty_weight > 0.0 else 1.0) * basis

        hf_share = sum(normalized_weights.get(name, 0.0) for name in HF_MODULE_NAMES)
        entropy = -sum(weight * log(max(weight, 1e-12)) for weight in normalized_weights.values())
        diversity_penalty = max(0.0, hf_share - 0.50) * 2.0 + (-0.01 * entropy)
        penalty_total = group_penalty + diversity_penalty

        penalized_score = raw_score - penalty_total

        extras_payload = {
            "group_penalty": float(group_penalty),
            "diversity_penalty": float(diversity_penalty),
            "hf_share": float(hf_share),
            "entropy": float(entropy),
            "weight_sum": float(total_weight),
            "fold_count": len(folds),
        }
        if hf_scale_value is not None:
            extras_payload["hf_weight_scale"] = float(hf_scale_value)

        snapshot = TrialSnapshot(
            trial_number=trial.number,
            logits=logits,
            weights=weight_map,
            metrics=metrics,
            raw_score=raw_score,
            penalty=penalty_total,
            extras=extras_payload,
        )

        save_json(outdir / f"trial_{trial.number:04d}.json", snapshot.to_dict())

        trial.set_user_attr("logits", logits)
        trial.set_user_attr("weights", weight_map)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("raw_score", raw_score)
        trial.set_user_attr("penalty", penalty_total)
        trial.set_user_attr("penalized_score", penalized_score)
        if group_totals:
            trial.set_user_attr("group_totals", group_totals)
        extras_user = dict(extras_payload)
        trial.set_user_attr("extras", extras_user)
        if hf_scale_value is not None:
            trial.set_user_attr("hf_weight_scale", float(hf_scale_value))

        return penalized_score

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        n_jobs=1,  # Single trial at a time, but parallel fold evaluation inside
    )

    best_trial = study.best_trial
    best_payload = _build_payload(
        logits=best_trial.user_attrs.get("logits", {}),
        weights=best_trial.user_attrs.get("weights", {}),
        metrics=best_trial.user_attrs.get("metrics", {}),
        score=best_trial.value,
        raw_score=best_trial.user_attrs.get("raw_score"),
        penalty=best_trial.user_attrs.get("penalty"),
        extras=best_trial.user_attrs.get("extras"),
    )
    study.set_user_attr("best_payload", best_payload)
    save_json(outdir / WEIGHTS_BUNDLE_NAME, {"global": best_payload, "weights_by_regime": {}})

    return study
