"""Residual learner for correcting ensemble prediction errors."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class ResidualLearner:
    """Gradient boosting meta-model that corrects base predictions."""

    def __init__(self, n_estimators: int = 200) -> None:
        self.model = GradientBoostingRegressor(n_estimators=n_estimators)

    def fit(self, y_true, y_pred) -> None:
        y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true_arr.size != y_pred_arr.size:
            raise ValueError("y_true and y_pred must have the same length")
        if y_true_arr.size == 0:
            raise ValueError("Cannot train residual learner on empty arrays")
        residuals = y_true_arr - y_pred_arr
        self.model.fit(y_pred_arr.reshape(-1, 1), residuals)

    def predict(self, y_pred):
        y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
        correction = self.model.predict(y_pred_arr.reshape(-1, 1))
        return y_pred_arr + correction
