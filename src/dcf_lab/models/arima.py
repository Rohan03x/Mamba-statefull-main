"""
ARIMA model implementation with auto_arima for baseline forecasting.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima.utils import ndiffs, nsdiffs

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """ARIMA model with automatic order selection and seasonal support."""

    def __init__(
        self,
        seasonal: bool = False,
        seasonal_period: Optional[int] = None,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        stationary: bool = False,
        information_criterion: str = "aic",
        alpha: float = 0.05,
    ):
        """
        Initialize ARIMA forecaster.

        Args:
            seasonal: Whether to fit seasonal ARIMA
            seasonal_period: Number of observations per seasonal cycle
            max_p: Maximum AR order
            max_d: Maximum difference order
            max_q: Maximum MA order
            max_P: Maximum seasonal AR order
            max_D: Maximum seasonal difference order
            max_Q: Maximum seasonal MA order
            stationary: Whether the data is already stationary
            information_criterion: Information criterion for model selection
            alpha: Significance level for confidence intervals
        """
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.stationary = stationary
        self.information_criterion = information_criterion
        self.alpha = alpha
        self.model = None
        self.scaler = None

    def _check_stationarity(self, y: np.ndarray) -> Tuple[int, int]:
        """
        Check stationarity and return recommended difference orders.

        Args:
            y: Time series data

        Returns:
            Tuple of (d, D) orders for regular and seasonal differencing
        """
        # Regular differencing
        d = 0 if self.stationary else ndiffs(y, alpha=self.alpha)

        # Seasonal differencing
        D = 0
        if self.seasonal and self.seasonal_period:
            D = nsdiffs(y, m=self.seasonal_period, alpha=self.alpha)

        return d, D

    def fit(self, y: Union[pd.Series, np.ndarray],
            **fit_kwargs) -> "ARIMAForecaster":
        """
        Fit ARIMA model using auto_arima.

        Args:
            y: Time series data
            **fit_kwargs: Additional kwargs for auto_arima

        Returns:
            Self
        """
        # Convert to numpy array if needed
        if isinstance(y, pd.Series):
            y = y.values

        # Check stationarity
        d, D = self._check_stationarity(y)

        # Fit model
        self.model = auto_arima(
            y,
            start_p=0,
            start_q=0,
            max_p=self.max_p,
            max_d=min(d, self.max_d),
            max_q=self.max_q,
            m=self.seasonal_period if self.seasonal else 1,
            seasonal=self.seasonal,
            start_P=0,
            start_Q=0,
            max_P=self.max_P,
            max_D=min(D, self.max_D),
            max_Q=self.max_Q,
            information_criterion=self.information_criterion,
            alpha=self.alpha,
            error_action="warn",
            suppress_warnings=True,
            stepwise=True,
            **fit_kwargs
        )

        return self

    def predict(self,
                n_periods: int) -> Tuple[np.ndarray,
                                         np.ndarray,
                                         np.ndarray]:
        """
        Generate forecasts with confidence intervals.

        Args:
            n_periods: Number of periods to forecast

        Returns:
            Tuple of (predictions, lower_ci, upper_ci)
        """
        if self.model is None:
            raise ValueError("Model must be fit before predicting")

        # Get predictions and confidence intervals
        predictions, conf_int = self.model.predict(
            n_periods=n_periods,
            return_conf_int=True,
            alpha=self.alpha
        )

        lower_ci = conf_int[:, 0]
        upper_ci = conf_int[:, 1]

        return predictions, lower_ci, upper_ci

    def get_info(self) -> Dict:
        """
        Get model information and parameters.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}

        return {
            "order": self.model.order,
            "seasonal_order": self.model.seasonal_order if self.seasonal else None,
            "aic": self.model.aic(),
            "bic": self.model.bic(),
            "params": self.model.params(),
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ARIMAForecaster":
        """
        Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        return joblib.load(path)


def fit_arima(
    train_series: pd.Series,
    seasonal: bool = False,
    seasonal_period: Optional[int] = None,
    **kwargs
) -> ARIMAForecaster:
    """
    Convenience function to fit ARIMA model.

    Args:
        train_series: Training data
        seasonal: Whether to fit seasonal ARIMA
        seasonal_period: Number of observations per seasonal cycle
        **kwargs: Additional kwargs for ARIMAForecaster

    Returns:
        Fitted ARIMAForecaster
    """
    model = ARIMAForecaster(
        seasonal=seasonal,
        seasonal_period=seasonal_period,
        **kwargs
    )
    return model.fit(train_series)
