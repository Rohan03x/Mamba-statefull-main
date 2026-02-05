from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PolicyAction:
    """Policy knob set selected by the contextual bandit.
    
    Includes:
    - Regime-aware thresholding parameters
    - Risk/exposure constraints
    - Quantile forecast blending weight (Option 1 from spec)
    """
    name: str
    base_threshold: float
    regime_mult_bull: float
    regime_mult_bear: float
    regime_mult_crisis: float
    target_vol: float
    turnover_cap: float
    max_gross: float
    max_net: float
    max_name: float
    weight_smoothing_alpha: float = 0.0
    vol_scaler: float = 1.0
    # Quantile forecast blending: z = (1-w_q)*z_mamba + w_q*z_quantile
    quantile_blend_weight: float = 0.0  # 0 = pure Mamba, 1 = pure quantile


class PolicyController:
    """Lightweight contextual bandit over a small action space.

    Uses linear Thompson sampling by default. Each action maintains its own
    Bayesian linear regression posterior: A (dxd) and b (d).
    """

    def __init__(
        self,
        actions: Sequence[PolicyAction],
        feature_dim: int,
        *,
        method: str = "lin_ts",  # lin_ts | ewa
        seed: Optional[int] = None,
        ts_prior_var: float = 1.0,
        ts_noise_var: float = 1.0,
        ewa_eta: float = 0.5,
        ewa_temperature: float = 1.0,
        warmup_steps: int = 0,
    ) -> None:
        self.actions = list(actions)
        if not self.actions:
            raise ValueError("PolicyController requires at least one action")
        self.feature_dim = int(feature_dim)
        self.method = str(method or "lin_ts").lower().strip()
        self.rng = np.random.default_rng(int(seed) if seed is not None else None)
        self.ts_prior_var = float(ts_prior_var)
        self.ts_noise_var = float(ts_noise_var)
        self.ewa_eta = float(ewa_eta)
        self.ewa_temperature = float(ewa_temperature)
        self.warmup_steps = int(max(0, warmup_steps))

        n = len(self.actions)
        d = self.feature_dim
        if self.method == "ewa":
            self._weights = np.ones(n, dtype=float)
        else:
            # Linear Thompson sampling state per action.
            self._A = [np.eye(d, dtype=float) * (1.0 / max(1e-6, self.ts_prior_var)) for _ in range(n)]
            self._b = [np.zeros(d, dtype=float) for _ in range(n)]
        self._t = 0

    def select_action(self, x: np.ndarray) -> Tuple[int, PolicyAction]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != int(self.feature_dim):
            raise ValueError(f"policy state dim mismatch: expected {self.feature_dim}, got {x.size}")

        n = len(self.actions)
        if self._t < self.warmup_steps:
            idx = int(self.rng.integers(0, n))
            self._t += 1
            return idx, self.actions[idx]

        if self.method == "ewa":
            w = np.asarray(self._weights, dtype=float)
            # Softmax with temperature for exploration.
            logits = w / max(1e-6, float(self.ewa_temperature))
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            probs = probs / float(np.sum(probs)) if float(np.sum(probs)) > 0 else np.full(n, 1.0 / n)
            idx = int(self.rng.choice(np.arange(n), p=probs))
            self._t += 1
            return idx, self.actions[idx]

        # Linear Thompson sampling.
        best_idx = 0
        best_score = -1e18
        for i in range(n):
            A = self._A[i]
            b = self._b[i]
            try:
                A_inv = np.linalg.pinv(A)
            except Exception:
                A_inv = np.eye(self.feature_dim, dtype=float)
            mu = A_inv @ b
            try:
                cov = float(self.ts_noise_var) * A_inv
                theta = self.rng.multivariate_normal(mean=mu, cov=cov)
            except Exception:
                theta = mu
            score = float(np.dot(theta, x))
            if score > best_score:
                best_score = score
                best_idx = i

        self._t += 1
        return int(best_idx), self.actions[int(best_idx)]

    def update(self, *, action_index: int, x: np.ndarray, reward: float) -> None:
        idx = int(action_index)
        if idx < 0 or idx >= len(self.actions):
            return
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != int(self.feature_dim):
            return
        r = float(reward)

        if self.method == "ewa":
            # Exponentially weighted experts update.
            self._weights[idx] = float(self._weights[idx]) * float(np.exp(self.ewa_eta * r))
            return

        # Linear Thompson sampling update.
        A = self._A[idx]
        b = self._b[idx]
        A = A + np.outer(x, x)
        b = b + r * x
        self._A[idx] = A
        self._b[idx] = b
