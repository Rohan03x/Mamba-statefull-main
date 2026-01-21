"""
Probabilistic Forecast Models

This module implements various models for probabilistic forecasting of key financial drivers,
including revenue, margins, capex, and working capital. These models generate probability
distributions rather than point estimates, allowing for more realistic and nuanced valuations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Constants
MODEL_NOT_FITTED = "Model is not fitted yet"

# Import TFT model if available (optional)
try:
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

# Import LightGBM (optional)
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


@dataclass
class ForecastConfig:
    """Configuration for probabilistic forecasting"""
    forecast_horizon: int = 5  # Number of years to forecast
    quantiles: Optional[List[float]] = None  # Default quantiles to output
    seasonality: bool = False  # Whether to model seasonality
    use_macro: bool = True  # Whether to include macroeconomic features
    # Factor to dampen trend over time (1.0 = no dampening)
    trend_dampening: float = 0.9

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]


class QuantileForecaster:
    """
    Base class for probabilistic forecasting models that output quantile predictions

    This provides a common interface for different quantile forecasting approaches,
    whether machine learning-based, statistical, or hybrid.
    """

    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize the forecaster

        Args:
            config: Forecast configuration
        """
        self.config = config or ForecastConfig()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'QuantileForecaster':
        """
        Fit the model to data

        Args:
            X: Features dataframe
            y: Target variable

        Returns:
            self
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Predict quantiles for given features

        Args:
            X: Features dataframe
            quantiles: List of quantiles to predict (if None, use config.quantiles)

        Returns:
            DataFrame with quantile predictions for each sample
        """
        raise NotImplementedError(
            "Subclasses must implement predict_quantiles()")

    def predict_distribution(self, X: pd.DataFrame) -> Any:
        """
        Return full probability distribution for predictions

        Args:
            X: Features dataframe

        Returns:
            Distribution object or parameters
        """
        raise NotImplementedError(
            "Subclasses must implement predict_distribution()")

    def plot_forecast(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        title: str = 'Quantile Forecast',
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot quantile forecasts

        Args:
            X: Features dataframe
            y_true: Optional true values to compare against
            title: Plot title
            figsize: Figure size
        """
        quantiles = self.config.quantiles
        preds = self.predict_quantiles(X, quantiles)

        plt.figure(figsize=figsize)

        # Plot median prediction
        median_idx = quantiles.index(
            0.5) if 0.5 in quantiles else len(quantiles) // 2
        plt.plot(preds.iloc[:, median_idx], 'b-', label='Median forecast')

        # Plot quantile intervals
        for i in range(len(quantiles) // 2):
            lower_idx = i
            upper_idx = len(quantiles) - i - 1

            if lower_idx != upper_idx:
                lower_q = quantiles[lower_idx]
                upper_q = quantiles[upper_idx]

                plt.fill_between(
                    range(len(preds)),
                    preds.iloc[:, lower_idx],
                    preds.iloc[:, upper_idx],
                    alpha=0.3,
                    label=f'{int(lower_q * 100)}-{int(upper_q * 100)} percentile'
                )

        # Plot true values if provided
        if y_true is not None:
            plt.plot(y_true, 'k.', label='Actual values')

        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()


class GBMQuantileForecaster(QuantileForecaster):
    """
    Gradient Boosting-based quantile forecaster

    Uses gradient boosting models trained to predict different quantiles.
    """

    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize the GBM quantile forecaster

        Args:
            config: Forecast configuration
        """
        super().__init__(config)
        self.models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GBMQuantileForecaster':
        """
        Fit GBM models for each quantile

        Args:
            X: Features dataframe
            y: Target variable

        Returns:
            self
        """
        # Train a separate GBM for each quantile
        for q in self.config.quantiles:
            if LGBM_AVAILABLE:
                import os
                params = {
                    'objective': lambda y_true, y_pred, quantile=q: stats.norm.ppf(quantile) * (y_true - y_pred),
                    'n_estimators': 100,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'random_state': 42
                }
                # Add GPU support via environment variable
                device = os.environ.get('LIGHTGBM_DEVICE', 'cpu')
                if device == 'gpu':
                    params['device_type'] = 'cuda'
                    params['gpu_platform_id'] = int(os.environ.get('LIGHTGBM_GPU_PLATFORM_ID', 0))
                    params['gpu_device_id'] = int(os.environ.get('LIGHTGBM_GPU_DEVICE_ID', 0))
                model = lgb.LGBMRegressor(**params)
            else:
                # Fallback to sklearn's GBM
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=q,
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )

            # Create a pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ], memory=None)

            # Fit the model
            pipeline.fit(X, y)
            self.models[q] = pipeline

        self.is_fitted = True
        return self

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Predict quantiles for given features

        Args:
            X: Features dataframe
            quantiles: List of quantiles to predict (if None, use config.quantiles)

        Returns:
            DataFrame with quantile predictions for each sample
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        quantiles = quantiles or self.config.quantiles

        # Check if all requested quantiles are available
        missing_quantiles = [q for q in quantiles if q not in self.models]
        if missing_quantiles:
            raise ValueError(
                f"Models not trained for quantiles: {missing_quantiles}")

        # Make predictions for each quantile
        preds = {}
        for q in quantiles:
            preds[f'q{int(q * 100)}'] = self.models[q].predict(X)

        return pd.DataFrame(preds)

    def predict_distribution(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Return approximate distribution based on quantiles

        Args:
            X: Features dataframe

        Returns:
            Dictionary with quantile values
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        quantile_preds = self.predict_quantiles(X)

        # Convert to dictionary of arrays
        dist = {
            q: quantile_preds[f'q{int(q * 100)} '].values
            for q in self.config.quantiles}

        return dist


class ResidualBootstrapForecaster(QuantileForecaster):
    """
    Residual bootstrap forecaster

    Uses a deterministic model for point prediction combined with
    bootstrapped residuals to generate a distribution.
    """

    def __init__(
        self,
        config: Optional[ForecastConfig] = None,
        base_model: Optional[Any] = None
    ):
        """
        Initialize the residual bootstrap forecaster

        Args:
            config: Forecast configuration
            base_model: Base model for point predictions
        """
        super().__init__(config)

        # Use provided model or create a default one
        self.base_model = base_model or ElasticNet(
            alpha=0.5, l1_ratio=0.7, random_state=42)

        # Create a pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.base_model)
        ], memory=None)

        self.residuals = None

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series) -> 'ResidualBootstrapForecaster':
        """
        Fit the base model and collect residuals

        Args:
            X: Features dataframe
            y: Target variable

        Returns:
            self
        """
        # Fit the base model
        self.pipeline.fit(X, y)

        # Calculate residuals
        y_pred = self.pipeline.predict(X)
        self.residuals = y - y_pred

        self.is_fitted = True
        return self

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        n_bootstrap: int = 1000
    ) -> pd.DataFrame:
        """
        Predict quantiles using bootstrap

        Args:
            X: Features dataframe
            quantiles: List of quantiles to predict (if None, use config.quantiles)
            n_bootstrap: Number of bootstrap samples

        Returns:
            DataFrame with quantile predictions for each sample
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        quantiles = quantiles or self.config.quantiles

        # Get point predictions
        point_preds = self.pipeline.predict(X)

        # Generate bootstrap predictions
        bootstrap_preds = np.zeros((len(X), n_bootstrap))

        # Create random number generator with fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)

        for i in range(n_bootstrap):
            # Sample residuals with replacement
            sampled_residuals = rng.choice(self.residuals, size=len(X))

            # Add to point predictions
            bootstrap_preds[:, i] = point_preds + sampled_residuals

        # Calculate quantiles from bootstrap predictions
        quantile_preds = {}
        for q in quantiles:
            q_values = np.percentile(bootstrap_preds, q * 100, axis=1)
            quantile_preds[f'q{int(q * 100)}'] = q_values

        return pd.DataFrame(quantile_preds)

    def predict_distribution(self, X: pd.DataFrame,
                             n_bootstrap: int = 1000) -> Dict[str, np.ndarray]:
        """
        Return approximate distribution based on bootstrap

        Args:
            X: Features dataframe
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with bootstrap samples
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        # Get point predictions
        point_preds = self.pipeline.predict(X)

        # Generate bootstrap predictions
        bootstrap_preds = np.zeros((len(X), n_bootstrap))

        # Create random number generator with fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)

        for i in range(n_bootstrap):
            # Sample residuals with replacement
            sampled_residuals = rng.choice(self.residuals, size=len(X))

            # Add to point predictions
            bootstrap_preds[:, i] = point_preds + sampled_residuals

        return {
            'point_predictions': point_preds,
            'bootstrap_samples': bootstrap_preds
        }


class TFTForecaster(QuantileForecaster):
    """
    Temporal Fusion Transformer based forecaster

    Uses the Temporal Fusion Transformer for multi-horizon quantile forecasting.
    Requires the TFT module to be available.
    """

    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize the TFT forecaster

        Args:
            config: Forecast configuration
        """
        if not TFT_AVAILABLE:
            raise ImportError(
                "TemporalFusionTransformer is not available. "
                "Please import the TFT module or install the required package."
            )

        super().__init__(config)
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TFTForecaster':
        """
        Fit the TFT model

        Args:
            X: Features dataframe
            y: Target variable

        Returns:
            self
        """
        # Initialize and fit the TFT model
        # Implementation depends on the specific TFT module

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Predict quantiles using TFT

        Args:
            X: Features dataframe
            quantiles: List of quantiles to predict (if None, use config.quantiles)

        Returns:
            DataFrame with quantile predictions for each sample
        """

    def predict_distribution(self, X: pd.DataFrame) -> Any:
        """
        Return distribution from TFT

        Args:
            X: Features dataframe

        Returns:
            Distribution object or parameters
        """


def create_revenue_forecaster(
    hist_data: pd.DataFrame,
    config: Optional[ForecastConfig] = None,
    model_type: str = 'gbm'
) -> QuantileForecaster:
    """
    Create and train a revenue forecaster

    Args:
        hist_data: Historical financial data
        config: Forecast configuration
        model_type: Type of model ('gbm', 'bootstrap', 'tft')

    Returns:
        Trained forecaster
    """
    if config is None:
        config = ForecastConfig()

    # Prepare features and target for revenue forecasting
    # This is a simplified example - in practice, you'd want more features
    features = pd.DataFrame(
        {'year': hist_data.index,
         'lagged_revenue': hist_data['revenue'].shift(1).fillna(
             method='bfill'),
         'revenue_growth': hist_data['revenue'].pct_change().fillna(
             method='bfill')})

    # Add macroeconomic features if specified
    if config.use_macro and 'gdp_growth' in hist_data.columns:
        features['gdp_growth'] = hist_data['gdp_growth']

    if config.use_macro and 'inflation' in hist_data.columns:
        features['inflation'] = hist_data['inflation']

    # Remove rows with NaN
    features = features.dropna()
    target = hist_data.loc[features.index, 'revenue']

    # Create the forecaster
    if model_type == 'gbm':
        forecaster = GBMQuantileForecaster(config)
    elif model_type == 'bootstrap':
        forecaster = ResidualBootstrapForecaster(config)
    elif model_type == 'tft' and TFT_AVAILABLE:
        forecaster = TFTForecaster(config)
    else:
        # Default to GBM if the requested model is not available
        forecaster = GBMQuantileForecaster(config)

    # Fit the model
    forecaster.fit(features, target)

    return forecaster


def create_margin_forecaster(
    hist_data: pd.DataFrame,
    margin_type: str = 'ebit_margin',
    config: Optional[ForecastConfig] = None,
    model_type: str = 'gbm'
) -> QuantileForecaster:
    """
    Create and train a margin forecaster

    Args:
        hist_data: Historical financial data
        margin_type: Type of margin to forecast ('gross_margin', 'ebit_margin', etc.)
        config: Forecast configuration
        model_type: Type of model ('gbm', 'bootstrap', 'tft')

    Returns:
        Trained forecaster
    """
    if config is None:
        config = ForecastConfig()

    # Prepare features and target for margin forecasting
    features = pd.DataFrame(
        {'year': hist_data.index,
         'lagged_margin': hist_data[margin_type].shift(1).fillna(
             method='bfill'),
         'margin_change': hist_data[margin_type].diff().fillna(
             method='bfill'),
         'revenue_growth': hist_data['revenue'].pct_change().fillna(
             method='bfill')})

    # Add macroeconomic features if specified
    if config.use_macro and 'gdp_growth' in hist_data.columns:
        features['gdp_growth'] = hist_data['gdp_growth']

    if config.use_macro and 'inflation' in hist_data.columns:
        features['inflation'] = hist_data['inflation']

    # Remove rows with NaN
    features = features.dropna()
    target = hist_data.loc[features.index, margin_type]

    # Create the forecaster
    if model_type == 'gbm':
        forecaster = GBMQuantileForecaster(config)
    elif model_type == 'bootstrap':
        forecaster = ResidualBootstrapForecaster(config)
    elif model_type == 'tft' and TFT_AVAILABLE:
        forecaster = TFTForecaster(config)
    else:
        # Default to GBM if the requested model is not available
        forecaster = GBMQuantileForecaster(config)

    # Fit the model
    forecaster.fit(features, target)

    return forecaster


def create_capex_forecaster(
    hist_data: pd.DataFrame,
    config: Optional[ForecastConfig] = None,
    model_type: str = 'gbm'
) -> QuantileForecaster:
    """
    Create and train a capital expenditure forecaster

    Args:
        hist_data: Historical financial data
        config: Forecast configuration
        model_type: Type of model ('gbm', 'bootstrap', 'tft')

    Returns:
        Trained forecaster
    """
    if config is None:
        config = ForecastConfig()

    # Calculate capex to revenue ratio
    hist_data['capex_to_revenue'] = hist_data['capex'] / hist_data['revenue']

    # Prepare features and target for capex forecasting
    features = pd.DataFrame(
        {'year': hist_data.index,
         'lagged_capex_ratio': hist_data['capex_to_revenue'].shift(1).fillna(
             method='bfill'),
         'revenue_growth': hist_data['revenue'].pct_change().fillna(
             method='bfill')})

    # Add depreciation feature if available
    if 'depreciation' in hist_data.columns:
        features['depreciation_to_revenue'] = (
            hist_data['depreciation'] / hist_data['revenue']
        ).fillna(method='bfill')

    # Remove rows with NaN
    features = features.dropna()
    target = hist_data.loc[features.index, 'capex_to_revenue']

    # Create the forecaster
    if model_type == 'gbm':
        forecaster = GBMQuantileForecaster(config)
    elif model_type == 'bootstrap':
        forecaster = ResidualBootstrapForecaster(config)
    elif model_type == 'tft' and TFT_AVAILABLE:
        forecaster = TFTForecaster(config)
    else:
        # Default to GBM if the requested model is not available
        forecaster = GBMQuantileForecaster(config)

    # Fit the model
    forecaster.fit(features, target)

    return forecaster


def create_nwc_forecaster(
    hist_data: pd.DataFrame,
    config: Optional[ForecastConfig] = None,
    model_type: str = 'gbm'
) -> QuantileForecaster:
    """
    Create and train a net working capital forecaster

    Args:
        hist_data: Historical financial data
        config: Forecast configuration
        model_type: Type of model ('gbm', 'bootstrap', 'tft')

    Returns:
        Trained forecaster
    """
    if config is None:
        config = ForecastConfig()

    # Calculate NWC to revenue ratio
    if 'nwc' in hist_data.columns:
        hist_data['nwc_to_revenue'] = hist_data['nwc'] / hist_data['revenue']
    else:
        # Calculate NWC from components if available
        if all(
            col in hist_data.columns for col in [
                'current_assets',
                'current_liabilities']):
            hist_data['nwc'] = hist_data['current_assets'] - \
                hist_data['current_liabilities']
            hist_data['nwc_to_revenue'] = hist_data['nwc'] / \
                hist_data['revenue']

    # Prepare features and target for NWC forecasting
    features = pd.DataFrame(
        {'year': hist_data.index,
         'lagged_nwc_ratio': hist_data['nwc_to_revenue'].shift(1).fillna(
             method='bfill'),
         'nwc_change': hist_data['nwc_to_revenue'].diff().fillna(
             method='bfill'),
         'revenue_growth': hist_data['revenue'].pct_change().fillna(
             method='bfill')})

    # Remove rows with NaN
    features = features.dropna()
    target = hist_data.loc[features.index, 'nwc_to_revenue']

    # Create the forecaster
    if model_type == 'gbm':
        forecaster = GBMQuantileForecaster(config)
    elif model_type == 'bootstrap':
        forecaster = ResidualBootstrapForecaster(config)
    elif model_type == 'tft' and TFT_AVAILABLE:
        forecaster = TFTForecaster(config)
    else:
        # Default to GBM if the requested model is not available
        forecaster = GBMQuantileForecaster(config)

    # Fit the model
    forecaster.fit(features, target)

    return forecaster


def generate_driver_distributions(
    hist_data: pd.DataFrame,
    forecast_horizon: int = 5,
    n_simulations: int = 1000,
    use_macro: bool = True,
    model_type: str = 'gbm',
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate correlated driver distributions for Monte Carlo simulation

    Args:
        hist_data: Historical financial data
        forecast_horizon: Number of years to forecast
        n_simulations: Number of Monte Carlo simulations
        use_macro: Whether to include macroeconomic features
        model_type: Type of model for forecasting
        seed: Random seed

    Returns:
        Dictionary of driver arrays (shape: n_simulations x forecast_horizon)
    """
    # Create numpy random number generator with seed
    rng = np.random.default_rng(seed=seed)

    # Create forecast configuration
    config = ForecastConfig(
        forecast_horizon=forecast_horizon,
        use_macro=use_macro
    )

    # Create forecasters
    revenue_forecaster = create_revenue_forecaster(
        hist_data, config, model_type)
    ebit_margin_forecaster = create_margin_forecaster(
        hist_data, 'ebit_margin', config, model_type
    )
    gross_margin_forecaster = create_margin_forecaster(
        hist_data, 'gross_margin', config, model_type
    )
    capex_forecaster = create_capex_forecaster(hist_data, config, model_type)
    nwc_forecaster = create_nwc_forecaster(hist_data, config, model_type)

    # Prepare features for prediction
    # This is simplified - in practice, you'd create proper forecast features
    pred_features = pd.DataFrame({
        'year': range(hist_data.index[-1] + 1, hist_data.index[-1] + 1 + forecast_horizon),
        'lagged_revenue': [hist_data['revenue'].iloc[-1]] * forecast_horizon,
        'revenue_growth': [hist_data['revenue'].pct_change().mean()] * forecast_horizon,
        'lagged_margin': [hist_data['ebit_margin'].iloc[-1]] * forecast_horizon,
        'margin_change': [hist_data['ebit_margin'].diff().mean()] * forecast_horizon,
        'lagged_capex_ratio': [hist_data['capex'].iloc[-1] / hist_data['revenue'].iloc[-1]] * forecast_horizon,
        'lagged_nwc_ratio': [
            (hist_data['current_assets'].iloc[-1] - hist_data['current_liabilities'].iloc[-1]) /
            hist_data['revenue'].iloc[-1]
        ] * forecast_horizon if all(col in hist_data.columns for col in ['current_assets', 'current_liabilities']) else [0.1] * forecast_horizon
    })

    # Generate distribution samples
    revenue_dist = revenue_forecaster.predict_distribution(pred_features)
    ebit_margin_dist = ebit_margin_forecaster.predict_distribution(
        pred_features)
    gross_margin_dist = gross_margin_forecaster.predict_distribution(
        pred_features)
    capex_dist = capex_forecaster.predict_distribution(pred_features)
    nwc_dist = nwc_forecaster.predict_distribution(pred_features)

    # Generate correlated samples
    # This is a simplified approach - in practice, you'd use copulas or more sophisticated methods
    # to model the joint distribution and preserve correlations

    # Initialize output dictionary
    driver_samples = {
        'rev_cagr': np.zeros(n_simulations),
        'ebit_margin': np.zeros((n_simulations, forecast_horizon)),
        'gross_margin': np.zeros((n_simulations, forecast_horizon)),
        'capex_to_rev': np.zeros((n_simulations, forecast_horizon)),
        'nwc_to_rev': np.zeros((n_simulations, forecast_horizon)),
    }

    # Calculate revenue CAGR from revenue forecasts
    if isinstance(revenue_dist, dict) and 'bootstrap_samples' in revenue_dist:
        # For bootstrap forecaster
        bootstrap_samples = revenue_dist['bootstrap_samples']

        for i in range(n_simulations):
            # Sample a random bootstrap trajectory
            sample_idx = rng.integers(bootstrap_samples.shape[1])
            start_revenue = hist_data['revenue'].iloc[-1]
            end_revenue = bootstrap_samples[-1, sample_idx]

            # Calculate CAGR
            driver_samples['rev_cagr'][i] = (
                end_revenue / start_revenue) ** (1 / forecast_horizon) - 1
    else:
        # For other forecasters, use quantile approach
        for i in range(n_simulations):
            quantile = rng.uniform(0.01, 0.99)
            rev_quantile = np.interp(
                quantile, config.quantiles,
                [revenue_dist[q][-1] for q in config.quantiles])
            start_revenue = hist_data['revenue'].iloc[-1]

            # Calculate CAGR
            driver_samples['rev_cagr'][i] = (
                rev_quantile / start_revenue) ** (1 / forecast_horizon) - 1

    # Sample other drivers
    for i in range(n_simulations):
        # Sample a random trajectory for each driver
        for j in range(forecast_horizon):
            # Sample from distributions
            driver_samples['ebit_margin'][i, j] = np.interp(
                rng.uniform(0.01, 0.99),
                config.quantiles,
                [ebit_margin_dist[q][j]
                    for q in config.quantiles if q in ebit_margin_dist]
            )

            driver_samples['gross_margin'][i, j] = np.interp(
                rng.uniform(0.01, 0.99),
                config.quantiles,
                [gross_margin_dist[q][j]
                    for q in config.quantiles if q in gross_margin_dist]
            )

            driver_samples['capex_to_rev'][i, j] = np.interp(
                rng.uniform(0.01, 0.99),
                config.quantiles,
                [capex_dist[q][j] for q in config.quantiles if q in capex_dist]
            )

            driver_samples['nwc_to_rev'][i, j] = np.interp(
                rng.uniform(0.01, 0.99),
                config.quantiles,
                [nwc_dist[q][j] for q in config.quantiles if q in nwc_dist]
            )

    # Add average values over the forecast horizon
    driver_samples['ebit_margin_avg'] = np.mean(
        driver_samples['ebit_margin'], axis=1)
    driver_samples['gross_margin_avg'] = np.mean(
        driver_samples['gross_margin'], axis=1)
    driver_samples['capex_to_rev_avg'] = np.mean(
        driver_samples['capex_to_rev'], axis=1)
    driver_samples['nwc_to_rev_avg'] = np.mean(
        driver_samples['nwc_to_rev'], axis=1)

    return driver_samples
