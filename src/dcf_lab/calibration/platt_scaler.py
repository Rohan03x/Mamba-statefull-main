"""Platt scaling utilities for module calibration.

Implements Platt scaling to map raw module scores onto calibrated probabilities
and provides a fold-aware store for managing calibrators across cross-validation
folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Mapping, MutableMapping, Tuple

import numpy as np

EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    out = np.empty_like(z)
    positive = z >= 0
    negative = ~positive
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)
    return out


@dataclass
class PlattParams:
    """Persisted Platt parameters."""

    a: float
    b: float


class PlattScaler:
    """Logistic calibration following Platt (1999).

    The scaler fits parameters ``a`` and ``b`` such that ``P(y = 1 | s)`` is the
    logistic transform of the standardized module score ``s``. Calibrated
    outputs can be returned as probabilities in ``[0, 1]`` or remapped to
    directional scores in ``[-1, 1]``.
    """

    def __init__(self, reg: float = 1e-6, max_iter: int = 100, tol: float = 1e-5):
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.params: PlattParams | None = None

    @property
    def is_fitted(self) -> bool:
        return self.params is not None

    def fit(self, x: Iterable[float], y: Iterable[int | float]) -> "PlattScaler":
        """Fit Platt parameters from a training fold."""

        x_arr = np.asarray(list(x), dtype=float).ravel()
        y_arr = np.asarray(list(y), dtype=float).ravel()

        if x_arr.size != y_arr.size:
            raise ValueError("x and y must contain the same number of samples")
        if x_arr.size == 0:
            raise ValueError("cannot fit Platt scaler on empty data")

        unique_labels = np.unique(y_arr)
        if unique_labels.size < 2:
            self.params = PlattParams(a=1.0, b=0.0)
            return self

        prior1 = np.sum(y_arr)
        prior0 = y_arr.size - prior1
        a = 0.0
        b = log((prior1 + 1.0) / (prior0 + 1.0))

        for _ in range(self.max_iter):
            linear = a * x_arr + b
            p = _sigmoid(linear)
            q = 1.0 - p

            diff = p - y_arr
            g1 = np.dot(diff, x_arr) + self.reg * a
            g2 = np.sum(diff) + self.reg * b

            w = p * q
            h11 = np.dot(w, x_arr * x_arr) + self.reg
            h22 = np.sum(w) + self.reg
            h21 = np.dot(w, x_arr)

            det = h11 * h22 - h21 * h21
            if abs(det) < EPS:
                break

            delta_a = -(g1 * h22 - g2 * h21) / det
            delta_b = -(g2 * h11 - g1 * h21) / det

            if abs(delta_a) < self.tol and abs(delta_b) < self.tol:
                a += delta_a
                b += delta_b
                break

            a += delta_a
            b += delta_b

        self.params = PlattParams(a=float(a), b=float(b))
        return self

    def transform(
        self, x: Iterable[float], *, return_proba: bool = False
    ) -> np.ndarray:
        """Apply the fitted calibration."""

        if not self.is_fitted:
            raise RuntimeError("PlattScaler must be fitted before calling transform")

        x_arr = np.asarray(list(x), dtype=float).ravel()
        linear = self.params.a * x_arr + self.params.b
        proba = _sigmoid(linear)
        if return_proba:
            return proba
        return proba * 2.0 - 1.0

    def to_dict(self) -> Mapping[str, float]:
        if not self.is_fitted:
            raise RuntimeError("cannot serialize an unfitted PlattScaler")
        return {"type": "platt", "a": self.params.a, "b": self.params.b}

    @classmethod
    def from_dict(cls, payload: Mapping[str, float]) -> "PlattScaler":
        if payload.get("type") != "platt":
            raise ValueError("payload does not describe a PlattScaler")
        scaler = cls()
        scaler.params = PlattParams(a=float(payload["a"]), b=float(payload["b"]))
        return scaler


class CalibratorStore:
    """Persist and retrieve calibrators keyed by (module, horizon).

    The store tracks the originating fold so that training-only calibrators are
    never reused outside their intended cross-validation partition.
    """

    def __init__(self, fold_id: str | None = None) -> None:
        self.fold_id = fold_id
        self._store: MutableMapping[Tuple[str, int], PlattScaler] = {}

    def _ensure_fold(self, fold_id: str | None) -> None:
        if fold_id is None:
            return
        if self.fold_id is None:
            self.fold_id = fold_id
        elif self.fold_id != fold_id:
            raise ValueError(
                f"calibrator store fold mismatch: expected '{self.fold_id}', got '{fold_id}'"
            )

    def put(
        self,
        module: str,
        horizon: int,
        scaler: PlattScaler,
        *,
        fold_id: str | None = None,
    ) -> None:
        if not scaler.is_fitted:
            raise ValueError("only fitted scalers can be stored")
        self._ensure_fold(fold_id)
        self._store[(module, horizon)] = scaler

    def iter_items(self):
        """Yield ``(module, horizon, scaler)`` triples for all stored calibrators."""
        for (module, horizon), scaler in self._store.items():
            yield module, horizon, scaler

    def has(self, module: str, horizon: int, *, fold_id: str | None = None) -> bool:
        self._ensure_fold(fold_id)
        return (module, horizon) in self._store

    def get(
        self, module: str, horizon: int, *, fold_id: str | None = None
    ) -> PlattScaler | None:
        self._ensure_fold(fold_id)
        return self._store.get((module, horizon))

    def assert_fold(self, fold_id: str | None) -> None:
        self._ensure_fold(fold_id)
        if fold_id is not None and self.fold_id != fold_id:
            raise ValueError(
                f"calibrator store fold mismatch: expected '{self.fold_id}', got '{fold_id}'"
            )

    def to_dict(self) -> Mapping[str, Any]:
        items: dict[str, Mapping[str, float]] = {}
        for (module, horizon), scaler in self._store.items():
            key = f"{module}::{horizon}"
            items[key] = scaler.to_dict()
        payload: dict[str, Any] = {"items": items}
        if self.fold_id is not None:
            payload["fold_id"] = self.fold_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibratorStore":
        fold_id = payload.get("fold_id") if isinstance(payload.get("fold_id"), str) else None
        items = payload.get("items", {})
        store = cls(fold_id=fold_id)
        if not isinstance(items, Mapping):
            return store
        for key, params in items.items():
            module, horizon_str = key.split("::", maxsplit=1)
            horizon = int(horizon_str)
            scaler = PlattScaler.from_dict(params)
            store.put(module, horizon, scaler, fold_id=fold_id)
        return store


__all__ = ["PlattScaler", "CalibratorStore"]
