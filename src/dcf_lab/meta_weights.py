"""Meta-weighting utilities for combining module signals.

Implements a bounded simplex projection that transforms module-specific logits
into clipped, normalized weights. These weights can be tuned via Optuna or
other optimizers and later restored for inference while respecting per-module
weight caps.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


def project_weights_with_bounds(
    weights: Mapping[str, float],
    w_min: Mapping[str, float],
    w_max: Mapping[str, float],
    *,
    target_sum: Optional[float] = 1.0,
    iters: int = 50,
) -> Dict[str, float]:
    """Project raw positive weights onto a bounded simplex.

    Args:
        weights: Mapping of module names to positive raw weights.
        w_min: Per-module lower bounds.
        w_max: Per-module upper bounds.
        target_sum: Desired sum of the projected weights.
        iters: Maximum number of clipping/renormalisation iterations.

    Returns:
        Dictionary containing projected weights that satisfy the bounds and
        target simplex constraint.
    """

    keys = list(weights.keys())
    if not keys:
        return {}

    raw = np.array([max(1e-12, float(weights[k])) for k in keys], dtype=float)
    lo = np.array([max(0.0, float(w_min.get(k, 0.0))) for k in keys], dtype=float)
    hi = np.array([max(lo[idx], float(w_max.get(k, 1.0))) for idx, k in enumerate(keys)], dtype=float)

    if target_sum is not None:
        # Initialise inside the simplex using a temperature-scaled softmax.
        raw = np.exp(raw - raw.max())
        denom = raw.sum()
        if denom <= 0.0:
            raw = np.full_like(raw, 1.0 / len(raw))
        else:
            raw = raw / denom * target_sum

    w = raw
    for _ in range(max(1, int(iters))):
        previous = w.copy()
        w = np.clip(w, lo, hi)
        if target_sum is not None:
            total = w.sum()
            if total <= 0.0:
                w = np.maximum(lo, 1e-12)
                total = w.sum()
            if not np.isclose(total, target_sum):
                free = (w > lo + 1e-12) & (w < hi - 1e-12)
                if free.any():
                    w[free] *= (target_sum / total)
                else:
                    w *= (target_sum / total)
                    break
        if np.allclose(w, previous, atol=1e-7):
            break

    return dict(zip(keys, w.tolist()))


class MetaWeights:
    """Softmax-based weighting engine with per-module bounds."""

    def __init__(
        self,
        modules: Sequence[str],
        w_min: float = 0.02,
        w_max: float = 0.40,
        *,
        w_min_map: Optional[Mapping[str, float]] = None,
        w_max_map: Optional[Mapping[str, float]] = None,
        normalize: bool = True,
    ) -> None:
        self.modules = list(modules)
        self.logits: Dict[str, float] = dict(
            zip(self.modules, (0.0,) * len(self.modules))
        )
        self.default_w_min = float(w_min)
        self.default_w_max = float(w_max)
        self.normalize = bool(normalize)

        self.w_min_map = {
            module: float(max(0.0, (w_min_map or {}).get(module, self.default_w_min)))
            for module in self.modules
        }
        self.w_max_map = {
            module: float(max(self.w_min_map.get(module, 0.0), (w_max_map or {}).get(module, self.default_w_max)))
            for module in self.modules
        }
        if not self.modules:
            return

        if not 0.0 <= self.default_w_min <= self.default_w_max <= 1.0:
            raise ValueError("weight bounds must satisfy 0 <= w_min <= w_max <= 1")

        if self.normalize:
            lower_total = sum(self.w_min_map.values())
            upper_total = sum(self.w_max_map.values())
            if lower_total - 1.0 > 1e-6:
                raise ValueError("lower weight bounds sum exceeds 1")
            if upper_total + 1e-6 < 1.0:
                raise ValueError("upper weight bounds sum below 1")

    def softmax(self) -> Dict[str, float]:
        """Return bounded simplex-projected weights keyed by module name."""

        if not self.modules:
            return {}

        logits_vec = np.array([self.logits.get(m, 0.0) for m in self.modules])
        logits_vec = logits_vec - logits_vec.max()
        raw = {
            module: float(np.exp(logit))
            for module, logit in zip(self.modules, logits_vec)
        }

        weights = project_weights_with_bounds(
            raw,
            self.w_min_map,
            self.w_max_map,
            target_sum=1.0 if self.normalize else None,
        )

        return weights

    def set_from_dict(self, logits_dict: Iterable[tuple[str, float]] | Dict[str, float]) -> None:
        """Update logits from an iterable of pairs or mapping."""

        if isinstance(logits_dict, dict):
            items = logits_dict.items()
        else:
            items = logits_dict
        for name, value in items:
            if name in self.logits:
                self.logits[name] = float(value)

    def as_logits_dict(self) -> Dict[str, float]:
        """Return a copy of the current logits keyed by module."""

        return dict(self.logits)


    __all__ = ["MetaWeights", "project_weights_with_bounds"]
