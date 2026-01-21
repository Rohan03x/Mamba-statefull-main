"""Universal module aggregation utilities.

This module centralizes the ingestion of module outputs and ensures every
signal is standardized via the shared signal bus before further processing.
"""
from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import CalibratorStore
from .regime_router import RegimeRouter
from .meta_weights import MetaWeights
from .signal_bus import ModuleSignal, standardize_module

logger = logging.getLogger(__name__)

# Registry containing the active modules keyed by their human-readable name.
LOADED_MODULES: MutableMapping[str, object] = {}
# Optional metadata declared by modules (e.g., weight bounds, provenance).
MODULE_METADATA: MutableMapping[str, dict] = {}

DEFAULT_GROUP_CAPS: Mapping[str, float] = {
    "hf": 0.5,
}


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


def _extract_metadata(module: object) -> dict:
    meta: dict = {}
    candidate = getattr(module, "metadata", None)
    if isinstance(candidate, Mapping):
        meta.update(candidate)
    getter = getattr(module, "get_metadata", None)
    if callable(getter):
        try:
            result = getter()
        except Exception:  # pragma: no cover - defensive logging
            result = None
        if isinstance(result, Mapping):
            meta.update(result)
    return meta


def _normalize_signal_payload(
    module_name: str, signal: ModuleSignal | None
) -> ModuleSignal | None:
    """Apply alignment, standardization, and schema enforcement."""

    if signal is None:
        logger.debug("module '%s' returned no signal", module_name)
        return None

    if not isinstance(signal, ModuleSignal):
        logger.warning(
            "module '%s' emitted unexpected payload type %s; skipping",
            module_name,
            type(signal).__name__,
        )
        return None

    if signal.df is None:
        logger.debug("module '%s' returned an empty dataframe", module_name)
        return None

    if not isinstance(signal.df, pd.DataFrame):
        logger.warning(
            "module '%s' produced %s instead of a DataFrame; skipping",
            module_name,
            type(signal.df).__name__,
        )
        return None

    df = signal.df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "module '%s' dataframe index conversion failed: %s",
                module_name,
                exc,
            )
            return None

    df.index = df.index.normalize()
    df = df.sort_index().shift(1)

    try:
        standardized_df = standardize_module(df)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "module '%s' produced non-standard dataframe: %s", module_name, exc
        )
        return None

    standardized_df = standardized_df.reindex(columns=["score", "conf"])
    standardized_df = standardized_df.fillna({"score": 0.0, "conf": 0.0})

    signal.df = standardized_df
    return signal


def _apply_calibration(
    module_name: str,
    horizon: int,
    signal: ModuleSignal,
    calibrator_store: CalibratorStore | None,
    fold_id: str | None,
) -> ModuleSignal:
    if calibrator_store is None:
        return signal

    try:
        scaler = calibrator_store.get(module_name, horizon, fold_id=fold_id)
    except ValueError as exc:  # pragma: no cover - defensive logging
        logger.error("calibrator store fold mismatch: %s", exc)
        raise

    if scaler is None:
        return signal

    try:
        proba = scaler.transform(signal.df["score"].values, return_proba=True)
        signal.df["score"] = 2.0 * proba - 1.0
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Calibration failed for %s, h=%s: %s", module_name, horizon, exc
        )
    return signal


def _extract_section_logits(
    section: Mapping[str, object],
    *,
    path: Path,
    section_name: str,
) -> Mapping[str, float]:
    if not isinstance(section, Mapping):
        logger.warning(
            "module weights payload in '%s' section '%s' is not a mapping",
            path,
            section_name,
        )
        return {}

    if "logits" in section and isinstance(section["logits"], Mapping):
        raw = section["logits"]
    elif "weights" in section and isinstance(section["weights"], Mapping):
        raw = section["weights"]
        if raw:
            return {
                k: float(np.log(max(float(v), 1e-12)))
                for k, v in _coerce_logits(raw).items()
            }
    else:
        raw = section

    logits = dict(_coerce_logits(raw))

    if logits and _looks_like_weights(logits):
        logger.warning(
            "module weights payload in '%s' section '%s' contained weights; converting",
            path,
            section_name,
        )
        logits = {
            k: float(np.log(max(v, 1e-12)))
            for k, v in logits.items()
        }

    return logits


def _extract_logits_payload(
    payload: object, path: Path
) -> Optional[Mapping[str, object]]:
    if isinstance(payload, Mapping) and "logits" in payload:
        inner = payload["logits"]
        if isinstance(inner, Mapping):
            return inner
        logger.warning(
            "module weights payload in '%s' has non-mapping 'logits'",
            path,
        )
        return None

    if isinstance(payload, Mapping):
        return payload

    logger.warning("module weights payload in '%s' is not a mapping", path)
    return None


def _parse_structured_weights_payload(
    payload: Mapping[str, object], *, path: Path
) -> tuple[Mapping[str, float], Mapping[str, Mapping[str, float]]]:
    global_logits: Mapping[str, float] = {}
    regimes: dict[str, Mapping[str, float]] = {}

    global_section = payload.get("global")
    if isinstance(global_section, Mapping):
        global_logits = _extract_section_logits(
            global_section, path=path, section_name="global"
        )

    weights_by_regime = payload.get("weights_by_regime")
    if isinstance(weights_by_regime, Mapping):
        for regime, section in weights_by_regime.items():
            if not isinstance(section, Mapping):
                logger.warning(
                    "module weights payload in '%s' regime '%s' is not a mapping",
                    path,
                    regime,
                )
                continue
            regimes[regime] = _extract_section_logits(
                section, path=path, section_name=str(regime)
            )

    return global_logits, regimes


def _parse_legacy_weights_payload(
    payload: object, *, path: Path
) -> tuple[Mapping[str, float], Mapping[str, Mapping[str, float]]]:
    data = _extract_logits_payload(payload, path)
    if data is None:
        return {}, {}

    logits = dict(_coerce_logits(data))

    if logits and _looks_like_weights(logits):
        logger.warning(
            "module weights file '%s' looks like weights, converting to logits",
            path,
        )
        logits = {k: float(np.log(max(v, 1e-12))) for k, v in logits.items()}

    return logits, {}


def _load_weights_artifact(
    weights_path: Optional[str | Path],
) -> tuple[Mapping[str, float], Mapping[str, Mapping[str, float]]]:
    if not weights_path:
        return {}, {}

    path = Path(weights_path)
    if not path.exists():
        logger.warning("module weights path '%s' not found", path)
        return {}, {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("failed to load module weights from '%s': %s", path, exc)
        return {}, {}

    if isinstance(payload, Mapping) and (
        "global" in payload or "weights_by_regime" in payload
    ):
        return _parse_structured_weights_payload(payload, path=path)

    return _parse_legacy_weights_payload(payload, path=path)


def _coerce_logits(data: Mapping[str, object]) -> Mapping[str, float]:
    logits: dict[str, float] = {}
    for name, value in data.items():
        try:
            logits[str(name)] = float(value)
        except (TypeError, ValueError):
            logger.debug("skipping non-numeric logit for module '%s'", name)
            continue
    return logits


def _looks_like_weights(logits: Mapping[str, float]) -> bool:
    if not logits:
        return False
    if not all(0.0 <= v <= 1.0 for v in logits.values()):
        return False
    total = float(sum(logits.values()))
    return abs(total - 1.0) < 1e-6


def _select_regime_logits(
    regime_logits: Optional[Mapping[str, Mapping[str, float]]],
    router: Optional[RegimeRouter],
    price_history: Optional[pd.Series],
) -> tuple[Mapping[str, float], Optional[str]]:
    """Select regime-specific weights based on market conditions.
    
    Args:
        regime_logits: Mapping of regime labels to logits.
        router: RegimeRouter instance (will auto-create if None).
        price_history: Price series for regime detection.
        
    Returns:
        Tuple of (selected_logits, active_regime_label)
    """
    if not regime_logits:
        logger.debug("🌍 REGIME DETECTOR: No regime weights available - regime detection DISABLED")
        return {}, None
    
    logger.debug("🌍 REGIME DETECTOR: Regime weights found - attempting detection...")
    
    # 🌍 AUTO-INSTANTIATE ROUTER: Enable regime detection when weights exist
    if router is None:
        router = RegimeRouter(hysteresis_days=10)  # Default 10-day hysteresis
        logger.info("🌍 REGIME DETECTOR: Auto-created RegimeRouter with 10-day hysteresis for OOS detection")
    else:
        logger.debug("🌍 REGIME DETECTOR: Using provided RegimeRouter instance")
    
    if price_history is None:
        logger.warning("🌍 REGIME DETECTOR: No price_history provided - FALLING BACK to global weights")
        return {}, None

    clean_prices = price_history.dropna()
    if len(clean_prices) < 63:  # Need at least 63 days for trend detection
        logger.warning("🌍 REGIME DETECTOR: Insufficient price history (%d days < 63) - FALLING BACK to global weights", len(clean_prices))
        return {}, None
    
    logger.debug("🌍 REGIME DETECTOR: Analyzing %d days of price history...", len(clean_prices))

    try:
        detected = router.detect(clean_prices)
        logger.debug("🌍 REGIME DETECTOR: Raw detection result: %s", detected)
    except Exception as exc:
        logger.error("🌍 REGIME DETECTOR: Detection FAILED (%s) - FALLING BACK to global weights", exc)
        return {}, None

    active_state = router.step(detected)
    selected = regime_logits.get(active_state)
    if not selected and detected != active_state:
        selected = regime_logits.get(detected)
        active_state = detected if selected else active_state

    if selected:
        if detected == active_state:
            logger.info("🌍 REGIME DETECTOR: ✅ ACTIVE regime = %s (stable)", active_state)
        else:
            logger.info("🌍 REGIME DETECTOR: ✅ ACTIVE regime = %s (hysteresis-adjusted from %s)", active_state, detected)
    else:
        logger.error("🌍 REGIME DETECTOR: Regime '%s' has no weights defined - FALLING BACK to global", active_state)

    return (selected or {}), active_state


def _resolve_logits_payload(
    base_logits: Optional[Mapping[str, float]],
    weights_path: Optional[str | Path],
    regime_logits: Optional[Mapping[str, Mapping[str, float]]],
    router: Optional[RegimeRouter],
    price_history: Optional[pd.Series],
) -> tuple[
    Mapping[str, float],
    Optional[str],
    str,
    Mapping[str, float],
    Mapping[str, Mapping[str, float]],
]:
    file_global_logits, file_regime_logits = _load_weights_artifact(weights_path)

    direct_global_logits = dict(base_logits) if base_logits else {}
    combined_global_logits = direct_global_logits or file_global_logits

    explicit_regime_logits = regime_logits or {}
    combined_regime_logits = explicit_regime_logits or file_regime_logits

    regime_payload, regime_label = _select_regime_logits(
        combined_regime_logits, router, price_history
    )
    if regime_payload:
        source_label = f"regime::{regime_label}" if regime_label else "regime"
        return (
            regime_payload,
            regime_label,
            source_label,
            combined_global_logits,
            combined_regime_logits,
        )

    if combined_regime_logits and regime_label:
        logger.debug(
            "No weights found for detected regime '%s'; falling back to global",
            regime_label,
        )

    if combined_global_logits:
        source = "direct" if direct_global_logits else "global"
        return (
            combined_global_logits,
            regime_label,
            source,
            combined_global_logits,
            combined_regime_logits,
        )

    if direct_global_logits:
        return (
            direct_global_logits,
            regime_label,
            "direct",
            combined_global_logits,
            combined_regime_logits,
        )

    return {}, regime_label, "uniform", combined_global_logits, combined_regime_logits


def _safe_compute_weights(
    meta_weights: MetaWeights, modules: Sequence[str]
) -> Mapping[str, float]:
    try:
        weights_map = meta_weights.softmax()
    except ValueError as exc:  # pragma: no cover - defensive logging
        logger.warning("meta-weights normalization failed: %s", exc)
        weights_map = {name: 1.0 / len(modules) for name in modules}

    for module_name, weight_value in weights_map.items():
        lower = meta_weights.w_min_map.get(module_name, meta_weights.default_w_min)
        upper = meta_weights.w_max_map.get(module_name, meta_weights.default_w_max)
        if not (lower - 1e-6 <= weight_value <= upper + 1e-6):
            logger.warning(
                "Weight %s=%.3f out of bounds after renorm", module_name, weight_value
            )
    return dict(weights_map)


def _compute_meta_score(
    module_signals: Sequence[ModuleSignal],
    weights: Mapping[str, float],
    target_index: pd.Index,
) -> pd.Series:
    """Compute meta-score using NumPy-optimized operations (releases GIL).
    
    OPTIMIZATION: Minimized pandas operations to reduce GIL contention.
    Uses NumPy for all heavy computation to enable true parallelization.
    """
    if target_index.empty:
        return pd.Series(index=target_index, dtype=float, name="meta_score")
    
    # Pre-allocate NumPy arrays (avoids repeated pandas operations)
    n_signals = len(module_signals)
    n_timesteps = len(target_index)
    
    scores = np.zeros((n_signals, n_timesteps), dtype=np.float64)
    confs = np.zeros((n_signals, n_timesteps), dtype=np.float64)
    
    # Extract data once into NumPy arrays (minimal pandas operations)
    valid_timesteps_mask = np.ones(n_timesteps, dtype=bool)
    
    for i, signal in enumerate(module_signals):
        # Get aligned scores and confidences
        score_aligned = signal.df["score"].reindex(target_index)
        conf_aligned = signal.df["conf"].reindex(target_index)
        
        # Forward-fill in NumPy (faster than pandas)
        score_vals = score_aligned.values
        conf_vals = conf_aligned.values
        
        # Forward fill using NumPy (releases GIL)
        mask_score = ~np.isnan(score_vals)
        mask_conf = ~np.isnan(conf_vals)
        
        if mask_score.any():
            indices = np.arange(len(score_vals))
            valid_indices = indices[mask_score]
            # Forward fill: each position gets value from last valid index
            filled_score = np.interp(indices, valid_indices, score_vals[mask_score], left=0.0, right=score_vals[mask_score][-1] if mask_score.any() else 0.0)
            scores[i] = filled_score
        
        if mask_conf.any():
            indices = np.arange(len(conf_vals))
            valid_indices = indices[mask_conf]
            filled_conf = np.interp(indices, valid_indices, conf_vals[mask_conf], left=0.0, right=conf_vals[mask_conf][-1] if mask_conf.any() else 0.0)
            confs[i] = filled_conf
        
        # Track which timesteps have ANY valid data
        valid_timesteps_mask &= ~(np.isnan(score_vals) & np.isnan(conf_vals))
    
    # Filter to timesteps with at least some valid data
    if not valid_timesteps_mask.any():
        return pd.Series(index=target_index[valid_timesteps_mask], dtype=float, name="meta_score")
    
    # All subsequent operations in NumPy (releases GIL for parallel execution)
    weights_vec = np.array([weights[signal.name] for signal in module_signals], dtype=np.float64)
    
    # Compute meta-score: sum over families of (score * confidence * weight)
    # FIX: Don't average confidences - averaging dilutes the weight effect!
    meta_values = (scores * confs * weights_vec[:, np.newaxis]).sum(axis=0)
    
    return pd.Series(meta_values, index=target_index, name="meta_score")


def _log_meta_blend_summary(
    module_signals: Sequence[ModuleSignal],
    weights: Mapping[str, float],
    symbol: Optional[str],
    horizon: Optional[int],
    regime: Optional[str],
) -> None:
    if not module_signals or not weights:
        return

    top_weights = dict(
        sorted(weights.items(), key=lambda item: -item[1])[:5]
    )
    regime_tag = regime or "global"

    logger.info(
        "[%s h=%s %s] top weights: %s",
        symbol,
        horizon,
        regime_tag,
        top_weights,
    )


def _resolve_index(
    module_signals: Sequence[ModuleSignal],
    index: Optional[pd.Index],
) -> pd.Index:
    if index is not None:
        resolved = pd.Index(index)
        if not isinstance(resolved, pd.DatetimeIndex):
            resolved = pd.to_datetime(resolved)
        # Normalize timezone: convert to UTC or remove timezone info
        if hasattr(resolved, 'tz') and resolved.tz is not None:
            resolved = resolved.tz_convert('UTC').tz_localize(None)
        return resolved.sort_values()

    if not module_signals:
        return pd.Index([], dtype="datetime64[ns]")

    combined = module_signals[0].df.index
    # Normalize first index
    if hasattr(combined, 'tz') and combined.tz is not None:
        combined = combined.tz_convert('UTC').tz_localize(None)
    
    for signal in module_signals[1:]:
        sig_idx = signal.df.index
        # Normalize each signal index before union
        if hasattr(sig_idx, 'tz') and sig_idx.tz is not None:
            sig_idx = sig_idx.tz_convert('UTC').tz_localize(None)
        combined = combined.union(sig_idx)
    return combined.sort_values()


def blend_module_signals(
    module_signals: Sequence[ModuleSignal],
    *,
    index: Optional[pd.Index] = None,
    weights_path: Optional[str | Path] = None,
    logits: Optional[Mapping[str, float]] = None,
    regime_logits: Optional[Mapping[str, Mapping[str, float]]] = None,
    router: Optional[RegimeRouter] = None,
    price_history: Optional[pd.Series] = None,
    w_min: float = 0.02,
    w_max: float = 0.40,
    w_min_map: Optional[Mapping[str, float]] = None,
    w_max_map: Optional[Mapping[str, float]] = None,
    symbol: Optional[str] = None,
    horizon: Optional[int] = None,
    normalize_weights: bool = True,
) -> Mapping[str, object]:
    """Blend standardized module signals into a meta-score.

    Args:
        module_signals: Sequence of standardized :class:`ModuleSignal` objects.
        index: Target index to align on; defaults to the union of signal indices.
        weights_path: Optional filesystem path to a JSON payload containing saved
            logits under a ``"logits"`` key.
        logits: Optional mapping of module logits to seed the weights directly.
        regime_logits: Optional mapping of regime labels to logits mappings.
        router: Optional :class:`RegimeRouter` instance maintaining hysteresis state.
        price_history: Optional price series used to detect the active regime.
    w_min: Minimum per-module weight after clipping.
    w_max: Maximum per-module weight after clipping.
        w_min_map: Optional per-module lower bounds overriding ``w_min``.
        w_max_map: Optional per-module upper bounds overriding ``w_max``.
        symbol: Optional symbol identifier for logging diagnostics.
        horizon: Optional forecast horizon for logging diagnostics.
        normalize_weights: When ``True`` (default) weights are projected onto the
            unit simplex; when ``False`` the softmax projection is applied without
            enforcing a unit-sum constraint, enabling sparse/unnormalized blends.

    Returns:
    Mapping containing ``meta_score`` (a :class:`pandas.Series`),
    ``weights`` (per-module floats), the effective ``logits`` used, and the
    ``regime`` label if regime-aware logits were applied. Additional keys
    provide diagnostic context: ``weights_source`` (selection origin),
    ``global_logits`` (fallback logits, if present), and ``regime_logits``
    (regime-specific candidate logits).
    """

    if not module_signals:
        empty_index = _resolve_index(module_signals, index)
        return {
            "meta_score": pd.Series(index=empty_index, dtype=float, name="meta_score"),
            "weights": {},
            "logits": {},
        }

    target_index = _resolve_index(module_signals, index)

    modules = [signal.name for signal in module_signals]

    resolved_min, resolved_max = _resolve_module_bounds(
        modules,
        w_min,
        w_max,
        provided_min=w_min_map,
        provided_max=w_max_map,
    )

    if len(modules) == 1:
        only = modules[0]
        resolved_max[only] = max(1.0, resolved_max.get(only, 1.0))

    try:
        meta_weights = MetaWeights(
            modules,
            w_min=w_min,
            w_max=w_max,
            w_min_map=resolved_min,
            w_max_map=resolved_max,
            normalize=normalize_weights,
        )
    except ValueError as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Infeasible weight bounds for modules %s: %s; relaxing constraints",
            modules,
            exc,
        )
        meta_weights = MetaWeights(modules, w_min=0.0, w_max=1.0)

    (
        logits_payload,
        active_regime,
        weight_source,
        global_logits_payload,
        regime_logits_payload,
    ) = _resolve_logits_payload(
        logits,
        weights_path,
        regime_logits,
        router,
        price_history,
    )
    if logits_payload:
        meta_weights.set_from_dict(logits_payload)

    resolved_symbol = symbol or getattr(module_signals[0], "symbol", "unknown")
    resolved_horizon = horizon or getattr(module_signals[0], "horizon", "?")
    regime_for_log = active_regime or ("global" if global_logits_payload else "n/a")
    
    # 🌍 EXPLICIT REGIME DETECTION STATUS LOGGING
    regime_available = bool(regime_logits_payload)
    regime_used = bool(active_regime)
    
    if regime_available and regime_used:
        logger.info(
            "[%s h=%s regime=%s] ✅ REGIME DETECTOR: ENABLED & ACTIVE - using %s weights",
            resolved_symbol,
            resolved_horizon,
            regime_for_log,
            weight_source,
        )
    elif regime_available and not regime_used:
        logger.warning(
            "[%s h=%s regime=%s] ⚠️  REGIME DETECTOR: ENABLED but INACTIVE (using global fallback) - reason: %s",
            resolved_symbol,
            resolved_horizon,
            regime_for_log,
            "insufficient price history or detection failed",
        )
    else:
        # Not using regime switching - either training mode or runtime with global weights only
        logger.debug(
            "[%s h=%s] Using %s weights (regime detection not active)",
            resolved_symbol,
            resolved_horizon,
            weight_source,
        )

    weights = _safe_compute_weights(meta_weights, modules)

    if target_index.empty:
        meta_score = pd.Series(index=target_index, dtype=float, name="meta_score")
    else:
        meta_score = _compute_meta_score(module_signals, weights, target_index)

    _log_meta_blend_summary(
        module_signals, weights, resolved_symbol, resolved_horizon, active_regime
    )

    return {
        "meta_score": meta_score,
        "weights": weights,
        "logits": meta_weights.as_logits_dict(),
        "regime": active_regime,
        "weights_source": weight_source,
        "global_logits": dict(global_logits_payload),
        "regime_logits": {
            key: dict(value) for key, value in regime_logits_payload.items()
        },
    }


def register_module(name: str, module: object) -> None:
    """Register a module that exposes an ``emit_signal`` callable."""

    if not hasattr(module, "emit_signal"):
        raise AttributeError(
            f"module '{name}' must expose an 'emit_signal(symbol, horizon)' method"
        )
    LOADED_MODULES[name] = module
    MODULE_METADATA[name] = _extract_metadata(module)


def register_modules(modules: Iterable[tuple[str, object]]) -> None:
    """Bulk-register multiple modules.

    Args:
        modules: Iterable of ``(name, module)`` pairs.
    """

    for name, module in modules:
        register_module(name, module)


def clear_modules() -> None:
    """Remove all registered modules (useful for tests)."""

    LOADED_MODULES.clear()
    MODULE_METADATA.clear()


def _resolve_module_bounds(
    modules: Sequence[str],
    default_min: float,
    default_max: float,
    *,
    provided_min: Optional[Mapping[str, float]] = None,
    provided_max: Optional[Mapping[str, float]] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    resolved_min: dict[str, float] = {}
    resolved_max: dict[str, float] = {}
    group_members: dict[str, list[str]] = {}
    group_caps: dict[str, float] = {}
    for module in modules:
        meta = get_module_metadata(module)
        lower = float(max(0.0, (provided_min or {}).get(module, meta.get("w_min", default_min))))
        upper = float(max(lower, (provided_max or {}).get(module, meta.get("w_max", default_max))))
        resolved_min[module] = lower
        resolved_max[module] = upper

        group = str(meta.get("group", "")).strip().lower()
        if not group:
            continue

        cap: Optional[float]
        cap_value = meta.get("group_cap")
        if isinstance(cap_value, (int, float)):
            cap = float(cap_value)
        else:
            cap = None

        if cap is None:
            env_keys = (
                f"AUTO_OPT_GROUP_CAP_{group.upper()}",
                f"STAGE_A_GROUP_CAP_{group.upper()}",
                f"STAGEA_{group.upper()}_GROUP_CAP",
            )
            if group == "hf":
                env_keys = ("AUTO_OPT_HF_GROUP_CAP", "STAGE_A_HF_GROUP_CAP", *env_keys)
            cap = _read_env_float(*env_keys)

        if cap is None:
            default_cap = DEFAULT_GROUP_CAPS.get(group)
            cap = float(default_cap) if default_cap is not None else None

        if cap is None or cap <= 0.0 or cap >= 1.0:
            continue

        group_members.setdefault(group, []).append(module)
        if group in group_caps:
            group_caps[group] = min(group_caps[group], cap)
        else:
            group_caps[group] = cap

    for group, members in group_members.items():
        cap = group_caps.get(group)
        if cap is None:
            continue
        lower_sum = sum(resolved_min[m] for m in members)
        if lower_sum > cap + 1e-6:
            logger.warning(
                "Group '%s' lower bound sum %.3f exceeds cap %.3f; skipping cap enforcement",
                group,
                lower_sum,
                cap,
            )
            continue

        extra_cap = cap - lower_sum
        room = {m: max(0.0, resolved_max[m] - resolved_min[m]) for m in members}
        total_room = sum(room.values())
        if total_room <= extra_cap + 1e-9:
            continue
        if total_room <= 0.0:
            continue

        scale = extra_cap / total_room if total_room > 0.0 else 0.0
        for member in members:
            headroom = room[member]
            if headroom <= 0.0:
                continue
            resolved_max[member] = resolved_min[member] + headroom * scale

    return resolved_min, resolved_max


def list_modules() -> Tuple[str, ...]:
    """Return the names of currently registered modules."""

    return tuple(sorted(LOADED_MODULES.keys()))


def get_module_metadata(name: str) -> Mapping[str, object]:
    """Return metadata for a registered module, if available."""

    if name in MODULE_METADATA:
        return dict(MODULE_METADATA[name])

    module = LOADED_MODULES.get(name)
    if module is not None:
        meta = _extract_metadata(module)
        if meta:
            MODULE_METADATA[name] = dict(meta)
            return dict(meta)
    return {}


def set_module_metadata(name: str, meta: Mapping[str, object]) -> None:
    """Override metadata for a registered module."""

    if name not in LOADED_MODULES:
        logger.warning("Cannot set metadata for unregistered module '%s'", name)
        return
    MODULE_METADATA[name] = dict(MODULE_METADATA.get(name) or {}, **meta)


def ingest_signals(
    symbol: str,
    horizon: int,
    calibrator_store: CalibratorStore | None = None,
    *,
    fold_id: str | None = None,
) -> List[ModuleSignal]:
    """Gather standardized signals from every registered module.

    Each module is expected to provide an ``emit_signal`` function returning a
    :class:`ModuleSignal`. The dataframe component is normalized before being
    emitted downstream. Modules that fail to produce a valid signal are skipped
    with a warning instead of breaking the pipeline.
    """

    standardized_signals: List[ModuleSignal] = []

    for module_name, module in LOADED_MODULES.items():
        emit = getattr(module, "emit_signal", None)
        if emit is None:
            logger.warning("module '%s' lacks an 'emit_signal' handler", module_name)
            continue

        try:
            raw_signal = emit(symbol=symbol, horizon=horizon)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "module '%s' failed during emit_signal: %s", module_name, exc
            )
            continue

        prepared_signal = _normalize_signal_payload(module_name, raw_signal)
        if prepared_signal is None:
            continue

        try:
            calibrated_signal = _apply_calibration(
                module_name, horizon, prepared_signal, calibrator_store, fold_id
            )
        except ValueError:
            return standardized_signals

        calibrated_signal.symbol = symbol
        standardized_signals.append(calibrated_signal)

    return standardized_signals


__all__ = [
    "LOADED_MODULES",
    "MODULE_METADATA",
    "register_module",
    "register_modules",
    "clear_modules",
    "list_modules",
    "get_module_metadata",
    "ingest_signals",
    "blend_module_signals",
]
