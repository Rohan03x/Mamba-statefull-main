"""Online meta-weight adaptation utilities.

Implements safe, incremental reweighting of module logits using a recent
reward signal (rolling correlation between scores and future returns).
Supports scheduled weekly updates alongside optional ADWIN-driven immediate
adjustments when distribution drift is detected.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # Optional dependency for drift detection
    from river.drift import ADWIN  # type: ignore
except ImportError:  # pragma: no cover - optional path
    ADWIN = None  # type: ignore


VALID_MODES = {"live", "forecast"}


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""

    if values.size == 0:
        return values

    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if denom <= 0.0:
        return np.full_like(exp_vals, 1.0 / len(exp_vals))
    return exp_vals / denom


def compute_reward(
    sig_df: pd.DataFrame,
    fwd_ret: pd.Series,
    *,
    window: int = 63,
    method: str = "pearson",
) -> float:
    """Compute recent reward for a module.

    The reward is the correlation between the module's score and the
    forward return over the trailing ``window`` observations. Spearman
    rank correlation can be requested via ``method='spearman'``.
    """

    if sig_df.empty:
        return 0.0

    scores = sig_df["score"].tail(window)
    aligned_returns = fwd_ret.reindex_like(scores).fillna(0.0)

    if scores.std(ddof=0) < 1e-12 or aligned_returns.std(ddof=0) < 1e-12:
        return 0.0

    if method == "spearman":
        corr = scores.rank().corr(aligned_returns.rank())
    else:
        corr = scores.corr(aligned_returns)

    return float(corr if pd.notna(corr) else 0.0)


def update_logits(
    logits: Mapping[str, float],
    rewards: Mapping[str, float],
    *,
    eta: float = 0.05,
    clip: tuple[float, float] = (-3.0, 3.0),
    mean_reversion: float = 0.05,
) -> Dict[str, float]:
    """Apply multiplicative-weights styled update to logits."""

    if not logits:
        return {}

    modules: Sequence[str] = list(logits.keys())
    logits_vec = np.array([float(logits[m]) for m in modules], dtype=float)
    reward_vec = np.array([float(rewards.get(m, 0.0)) for m in modules], dtype=float)

    logits_vec = logits_vec + eta * reward_vec
    logits_vec = np.clip(logits_vec, clip[0], clip[1])

    weights = softmax(logits_vec)
    if mean_reversion > 0 and len(modules) > 0:
        uniform = np.full_like(weights, 1.0 / len(modules))
        weights = (1.0 - mean_reversion) * weights + mean_reversion * uniform
        weights = np.clip(weights, 1e-12, 1.0)
        weights = weights / weights.sum()

    new_logits = np.log(np.maximum(weights, 1e-12))
    return dict(zip(modules, new_logits.tolist()))


def weights_from_logits(logits: Mapping[str, float]) -> Dict[str, float]:
    """Return normalized weights implied by logits."""

    if not logits:
        return {}
    modules = list(logits.keys())
    logits_vec = np.array([float(logits[m]) for m in modules], dtype=float)
    weights = softmax(logits_vec)
    return dict(zip(modules, weights.tolist()))


def logits_from_weights(
    weights: Mapping[str, float], *, clip: tuple[float, float] = (-3.0, 3.0)
) -> Dict[str, float]:
    if not weights:
        return {}

    modules = list(weights.keys())
    weight_vec = np.array([max(float(weights[m]), 1e-12) for m in modules], dtype=float)
    weight_vec /= weight_vec.sum()
    logits_vec = np.log(weight_vec)
    logits_vec = np.clip(logits_vec, clip[0], clip[1])
    return dict(zip(modules, logits_vec.tolist()))


def apply_weight_delta_cap(
    *,
    old_weights: Mapping[str, float],
    new_weights: Mapping[str, float],
    cap: float,
) -> Dict[str, float]:
    capped: Dict[str, float] = {}
    for module, new_value in new_weights.items():
        base = float(old_weights.get(module, 0.0))
        delta = new_value - base
        if abs(delta) > cap:
            capped[module] = base + np.sign(delta) * cap
        else:
            capped[module] = new_value

    total = float(sum(capped.values()))
    if total <= 0.0:
        n = len(capped)
        return {module: 1.0 / n for module in capped}

    return {module: value / total for module, value in capped.items()}


def compute_rewards_for_modules(
    module_frames: Mapping[str, pd.DataFrame],
    forward_returns: pd.Series,
    *,
    window: int = 63,
    method: str = "pearson",
) -> Dict[str, float]:
    """Compute rewards for every module dataframe provided."""

    rewards: Dict[str, float] = {}
    for name, df in module_frames.items():
        try:
            rewards[name] = compute_reward(df, forward_returns, window=window, method=method)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("reward computation failed for module '%s': %s", name, exc)
            rewards[name] = 0.0
    return rewards


@dataclass
class DriftSignal:
    triggered: bool
    modules: Sequence[str]


class OnlineReweightingManager:
    """Stateful manager handling online meta-weight adaptation."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        mode: str = "live",
        eta: float = 0.05,
        logit_clip: tuple[float, float] = (-3.0, 3.0),
        mean_reversion: float = 0.05,
        window: int = 63,
        corr_method: str = "pearson",
        adwin_delta: float = 0.002,
        fallback_drift_threshold: float = 0.5,
        history_limit: int = 250,
        reward_ewm_alpha: Optional[float] = None,
        weight_delta_cap: float = 0.05,
    ) -> None:
        self.state_path = Path(state_path)
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES!r}")
        self.mode = mode
        self.eta = eta
        self.logit_clip = logit_clip
        self.mean_reversion = mean_reversion
        self.window = window
        self.corr_method = corr_method
        self.adwin_delta = adwin_delta
        self.fallback_drift_threshold = fallback_drift_threshold
        self.history_limit = history_limit
        self.reward_ewm_alpha = reward_ewm_alpha
        self.weight_delta_cap = max(float(weight_delta_cap), 0.0)

        self._adwins: MutableMapping[str, object] = {}
        self._state: MutableMapping[str, object] = self._load_state()
        self._reward_ema: Dict[str, float] = {
            str(k): float(v)
            for k, v in self._state.get("reward_ema", {}).items()
            if isinstance(v, (int, float))
        }

    @property
    def current_logits(self) -> Dict[str, float]:
        payload = self._state.get("current", {})
        logits = payload.get("logits", {}) if isinstance(payload, Mapping) else {}
        return {str(k): float(v) for k, v in logits.items()}

    def set_mode(self, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES!r}")
        self.mode = mode

    def _load_state(self) -> MutableMapping[str, object]:
        if not self.state_path.exists():
            return {"current": {}, "history": []}
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, Mapping):
                raise ValueError("state payload not a mapping")
            data.setdefault("history", [])
            data.setdefault("reward_ema", {})
            return dict(data)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("failed to load online weights state '%s': %s", self.state_path, exc)
            return {"current": {}, "history": []}

    def _ensure_adwin(self, module: str) -> Optional[object]:
        if ADWIN is None:
            return None
        tracker = self._adwins.get(module)
        if tracker is None:
            tracker = ADWIN(delta=self.adwin_delta)  # type: ignore
            self._adwins[module] = tracker
        return tracker

    def _check_drift(self, rewards: Mapping[str, float]) -> DriftSignal:
        if ADWIN is not None:
            modules = self._check_drift_with_adwin(rewards)
        else:
            modules = self._check_drift_with_threshold(rewards)

        return DriftSignal(triggered=bool(modules), modules=modules)

    def _check_drift_with_adwin(self, rewards: Mapping[str, float]) -> list[str]:
        triggered_modules: list[str] = []
        for module, reward in rewards.items():
            tracker = self._ensure_adwin(module)
            if tracker is None:
                continue
            try:
                tracker.update(float(reward))  # type: ignore[attr-defined]
                if getattr(tracker, "drift_detected", False):
                    triggered_modules.append(module)
                    tracker.reset()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("ADWIN update failed for %s: %s", module, exc)
        return triggered_modules

    def _check_drift_with_threshold(self, rewards: Mapping[str, float]) -> list[str]:
        return [
            module
            for module, reward in rewards.items()
            if abs(reward) >= self.fallback_drift_threshold
        ]

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["reward_ema"] = self._reward_ema
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2, sort_keys=True)

    def _smooth_rewards(self, rewards: Mapping[str, float]) -> Dict[str, float]:
        if not rewards:
            return {}

        alpha = self.reward_ewm_alpha
        if alpha is None or not (0.0 < alpha <= 1.0):
            return dict(rewards)

        smoothed: Dict[str, float] = {}
        for module, reward in rewards.items():
            previous = self._reward_ema.get(module)
            if previous is None:
                smoothed_val = float(reward)
            else:
                smoothed_val = float(alpha * reward + (1.0 - alpha) * previous)
            self._reward_ema[module] = smoothed_val
            smoothed[module] = smoothed_val

        return smoothed

    def apply_update(
        self,
        *,
        logits: Mapping[str, float],
        module_frames: Mapping[str, pd.DataFrame],
        forward_returns: pd.Series,
        timestamp: Optional[pd.Timestamp] = None,
        reason: str = "weekly",
    ) -> Dict[str, float]:
        if self.mode not in VALID_MODES:
            raise RuntimeError(
                "Online reweighting should only be applied in live/forecast mode."
            )
        if not logits:
            logger.info("No logits provided for online reweighting; skipping")
            return {}

        rewards_raw = compute_rewards_for_modules(
            module_frames, forward_returns, window=self.window, method=self.corr_method
        )

        rewards = self._smooth_rewards(rewards_raw)

        drift = self._check_drift(rewards_raw)
        effective_reason = reason
        if drift.triggered:
            effective_reason = f"{reason}+adwin"

        before_logits = {k: float(v) for k, v in logits.items()}
        before_weights = weights_from_logits(before_logits)

        updated_logits = update_logits(
            before_logits,
            rewards,
            eta=self.eta,
            clip=self.logit_clip,
            mean_reversion=self.mean_reversion,
        )
        after_weights = weights_from_logits(updated_logits)

        capped_weights = apply_weight_delta_cap(
            old_weights=before_weights,
            new_weights=after_weights,
            cap=self.weight_delta_cap,
        )
        weights_capped = capped_weights != after_weights
        if weights_capped:
            updated_logits = logits_from_weights(
                capped_weights, clip=self.logit_clip
            )
            after_weights = capped_weights

        event = {
            "timestamp": (timestamp or pd.Timestamp.utcnow()).isoformat(),
            "reason": effective_reason,
            "reason_raw": reason,
            "eta": self.eta,
            "mean_reversion": self.mean_reversion,
            "window": self.window,
            "drift_triggered": drift.triggered,
            "drift_modules": list(drift.modules),
            "rewards_raw": rewards_raw,
            "rewards": rewards,
            "reward_ewm_alpha": self.reward_ewm_alpha,
            "logits_before": before_logits,
            "logits_after": updated_logits,
            "weights_before": before_weights,
            "weights_after": after_weights,
            "weights_capped": weights_capped,
            "weight_delta_cap": self.weight_delta_cap,
        }

        history = self._state.setdefault("history", [])
        history.append(event)
        if isinstance(history, list) and len(history) > self.history_limit:
            del history[:-self.history_limit]

        self._state["current"] = {
            "timestamp": event["timestamp"],
            "reason": effective_reason,
            "logits": updated_logits,
            "weights": after_weights,
        }

        self._persist()
        logger.info(
            "Online weights updated (%s); drift=%s modules=%s",
            effective_reason,
            drift.triggered,
            drift.modules,
        )

        return updated_logits


__all__ = [
    "OnlineReweightingManager",
    "compute_reward",
    "compute_rewards_for_modules",
    "update_logits",
    "weights_from_logits",
    "logits_from_weights",
    "apply_weight_delta_cap",
]
