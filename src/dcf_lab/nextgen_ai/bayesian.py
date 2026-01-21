from typing import Dict, Tuple
import numpy as np


class BayesianCalibrator:
    """Lightweight Bayesian-style calibrator.

    - Uses rolling class-frequency priors (e.g., from recent labels) when provided.
    - Blends model probabilities with priors using a convex weight.
    - Returns a simple 3-bin distribution over {up, flat, down} where 'flat' is a small mass
      expressing uncertainty; the remaining mass is split between up/down according to the
      calibrated binary probability.
    """

    def __init__(self, blend: float = 0.7, flat_mass: float = 0.1):
        # Weight assigned to model probability vs prior: p_cal = blend*p_model + (1-blend)*p_prior
        self.blend = float(np.clip(blend, 0.0, 1.0))
        # Constant probability mass reserved for the 'flat' bin (uncertainty)
        self.flat_mass = float(np.clip(flat_mass, 0.0, 0.9))
        self._fitted = False

    def fit(self, *_args, **_kwargs):
        # Placeholder; reserved for future param learning
        self._fitted = True

    @staticmethod
    def class_freq_prior(y: np.ndarray) -> float:
        """Compute class-frequency prior P(up) from binary labels y in {0,1}.

        If y is empty, returns 0.5.
        """
        if y is None or len(y) == 0:
            return 0.5
        y = np.asarray(y).astype(int)
        return float(np.mean(y))

    def calibrate_binary(self, proba: np.ndarray, p_prior_up: float) -> np.ndarray:
        """Blend model probabilities with a scalar prior.

        proba: shape (n,)
        p_prior_up: scalar in [0,1]
        returns calibrated probabilities in [0,1] of shape (n,)
        """
        if proba is None or len(proba) == 0:
            return np.array([], dtype=float)
        p_model = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
        p_prior = float(np.clip(p_prior_up, 0.0, 1.0))
        return self.blend * p_model + (1.0 - self.blend) * p_prior

    def predict_distribution(self, proba: np.ndarray, priors: Dict[str, float]) -> Dict[str, float]:
        """Map binary probability to a 3-bin distribution with a reserved flat mass.

        Expected priors: {"p_up": float} or will fallback to 0.5.
        """
        p_up_prior = float(priors.get("p_up", 0.5)) if isinstance(priors, dict) else 0.5
        if proba is None or len(proba) == 0:
            p_up = p_up_prior
        else:
            p_up = float(np.clip(np.mean(self.calibrate_binary(proba, p_up_prior)), 0.0, 1.0))

        # Allocate flat mass and split the rest between up/down
        flat = self.flat_mass
        rem = max(0.0, 1.0 - flat)
        up = rem * p_up
        down = rem * (1.0 - p_up)
        # Final clip to [0,1]
        up = float(np.clip(up, 0.0, 1.0))
        flat = float(np.clip(flat, 0.0, 1.0))
        down = float(np.clip(down, 0.0, 1.0))
        # Renormalize to sum to 1
        s = up + flat + down
        if s <= 0:
            return {"up": 0.3333, "flat": 0.3333, "down": 0.3333}
        return {"up": up / s, "flat": flat / s, "down": down / s}

    @staticmethod
    def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute reliability diagram components: bin centers, accuracies, and counts.

        Returns (bin_centers, bin_acc, bin_counts)
        """
        if p is None or y is None or len(p) == 0 or len(y) == 0:
            return np.array([]), np.array([]), np.array([])
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        inds = np.digitize(p, edges[1:-1], right=True)
        bin_acc = np.zeros(n_bins, dtype=float)
        bin_counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mask = inds == b
            if np.any(mask):
                bin_counts[b] = int(mask.sum())
                bin_acc[b] = float(np.mean(y[mask]))
        centers = (edges[:-1] + edges[1:]) / 2.0
        return centers, bin_acc, bin_counts
