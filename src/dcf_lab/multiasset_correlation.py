"""
Multi-Asset Correlation Modeling for Portfolio-Level Forecasting

This module implements advanced correlation modeling techniques including:
- Dynamic Conditional Correlation (DCC) models
- Factor models for dimensionality reduction
- Copula-based dependence modeling
- Portfolio-level risk forecasting
- Cross-asset spillover effects

Key Features:
- Handle hundreds of assets simultaneously
- Capture time-varying correlations
- Model tail dependencies and regime changes
- Support portfolio optimization constraints
- Provide risk attribution and decomposition
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import minimize

# Constants
MODEL_NOT_FITTED_ERROR = "Model must be fitted first"

# Optional dependencies with fallbacks
try:
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available - some features disabled")

try:
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    # Don't warn about arch - we'll implement simple GARCH alternative


class SimpleGARCH:
    """Simple GARCH(1,1) implementation as fallback when arch package unavailable"""

    def __init__(self, alpha=0.1, beta=0.8, omega=0.01):
        self.alpha = alpha  # ARCH coefficient
        self.beta = beta    # GARCH coefficient
        self.omega = omega  # Constant term
        self.fitted = False
        self.conditional_volatility = None

    def fit(self, returns):
        """Fit simple GARCH(1,1) model"""
        returns = np.array(returns).flatten()
        n = len(returns)

        # Initialize conditional variance
        variance = np.zeros(n)
        variance[0] = np.var(returns)  # Initial variance

        # Simple GARCH(1,1): σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
        for t in range(1, n):
            variance[t] = (self.omega +
                           self.alpha * returns[t-1]**2 +
                           self.beta * variance[t-1])

        self.conditional_volatility = np.sqrt(variance)
        self.fitted = True
        return self

    def forecast(self, horizon=1):
        """Simple GARCH forecast"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        # Simple persistence forecast
        last_variance = self.conditional_volatility[-1]**2
        forecasts = []

        for h in range(horizon):
            # Mean reversion to long-run variance
            persistence = (self.alpha + self.beta)**h
            long_run_var = self.omega / (1 - self.alpha - self.beta)
            forecast_var = persistence * last_variance + \
                (1 - persistence) * long_run_var
            forecasts.append(np.sqrt(forecast_var))

        return np.array(forecasts)


logger = logging.getLogger(__name__)


@dataclass
class CorrelationForecast:
    """Results from correlation modeling"""
    correlation_matrix: np.ndarray
    factor_loadings: Optional[np.ndarray] = None
    idiosyncratic_variance: Optional[np.ndarray] = None
    portfolio_variance: Optional[float] = None
    risk_attribution: Optional[Dict[str, float]] = None
    confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None
    forecast_date: Optional[datetime] = None
    methodology: str = "unknown"


class DynamicCorrelationModel(nn.Module):
    """Neural network-based dynamic correlation model"""

    def __init__(
        self,
        n_assets: int,
        hidden_dim: int = 64,
        n_factors: int = 5,
        activation: str = 'tanh'
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors

        # Factor extraction network
        self.factor_net = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            self._get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim // 2),
            self._get_activation(activation),
            nn.Linear(hidden_dim // 2, n_factors)
        )

        # Loadings network (maps factors to correlations)
        self.loadings_net = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            self._get_activation(activation),
            nn.Linear(hidden_dim, n_assets * n_factors)
        )

        # Idiosyncratic variance network
        self.idio_net = nn.Sequential(
            nn.Linear(n_assets, hidden_dim // 2),
            self._get_activation(activation),
            nn.Linear(hidden_dim // 2, n_assets),
            nn.Softplus()  # Ensure positive variances
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(activation, nn.Tanh())

    def forward(self,
                returns: torch.Tensor) -> Tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        """
        Forward pass

        Args:
            returns: [batch_size, n_assets] return matrix

        Returns:
            correlation_matrix: [batch_size, n_assets, n_assets]
            factor_loadings: [batch_size, n_assets, n_factors]
            idio_variance: [batch_size, n_assets]
        """
        batch_size = returns.shape[0]

        # Extract factors
        factors = self.factor_net(returns)  # [batch_size, n_factors]

        # Generate loadings
        # [batch_size, n_assets * n_factors]
        loadings_flat = self.loadings_net(factors)
        loadings = loadings_flat.view(
            batch_size, self.n_assets, self.n_factors)

        # Generate idiosyncratic variances
        idio_var = self.idio_net(returns)  # [batch_size, n_assets]

        # Construct correlation matrix: R = L @ L.T + diag(idio_var)
        factor_cov = torch.bmm(loadings, loadings.transpose(-2, -1))
        idio_diag = torch.diag_embed(idio_var)

        correlation_matrix = factor_cov + idio_diag

        # Normalize to correlations (optional - could keep as covariance)
        std_dev = torch.sqrt(
            torch.diagonal(
                correlation_matrix,
                dim1=-2,
                dim2=-1))
        std_outer = torch.bmm(std_dev.unsqueeze(-1), std_dev.unsqueeze(-2))
        correlation_matrix = correlation_matrix / (std_outer + 1e-8)

        return correlation_matrix, loadings, idio_var


class CopulaModel:
    """Copula-based dependence modeling for tail dependencies"""

    def __init__(self, copula_type: str = 'gaussian'):
        self.copula_type = copula_type
        self.fitted_params = None

    def fit(self, returns: np.ndarray) -> 'CopulaModel':
        """Fit copula to return data"""
        n_obs, n_assets = returns.shape

        # Convert to uniform margins using empirical CDF
        uniform_data = np.zeros_like(returns)
        for i in range(n_assets):
            ranks = stats.rankdata(returns[:, i])
            uniform_data[:, i] = ranks / (n_obs + 1)

        if self.copula_type == 'gaussian':
            # Gaussian copula - estimate correlation matrix
            normal_data = stats.norm.ppf(uniform_data)
            correlation_matrix = np.corrcoef(normal_data.T)
            self.fitted_params = {'correlation': correlation_matrix}

        elif self.copula_type == 't':
            # t-copula - estimate correlation and degrees of freedom
            normal_data = stats.norm.ppf(uniform_data)
            correlation_matrix = np.corrcoef(normal_data.T)

            # Estimate degrees of freedom (simplified approach)
            df_estimate = self._estimate_t_df(uniform_data)
            self.fitted_params = {
                'correlation': correlation_matrix,
                'df': df_estimate
            }

        return self

    def _estimate_t_df(self, uniform_data: np.ndarray) -> float:
        """Estimate degrees of freedom for t-copula"""
        # Simplified ML estimation for df
        def neg_log_likelihood(df_param):
            if df_param <= 2:
                return 1e6
            try:
                t_data = stats.t.ppf(uniform_data, df_param)
                if np.any(np.isinf(t_data)) or np.any(np.isnan(t_data)):
                    return 1e6
                return -np.sum(stats.multivariate_t.logpdf(
                    t_data,
                    df=df_param,
                    shape=np.corrcoef(t_data.T)
                ))
            except (ValueError, np.linalg.LinAlgError):
                return 1e6

        result = minimize(neg_log_likelihood, x0=5.0, bounds=[(2.1, 30.0)])
        return max(result.x[0], 3.0)

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate samples from fitted copula"""
        if self.fitted_params is None:
            raise ValueError(MODEL_NOT_FITTED_ERROR)

        n_assets = self.fitted_params['correlation'].shape[0]

        # Create random generator for reproducibility
        rng = np.random.default_rng(42)

        if self.copula_type == 'gaussian':
            # Generate multivariate normal samples
            normal_samples = rng.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=self.fitted_params['correlation'],
                size=n_samples
            )
            # Convert to uniform
            uniform_samples = stats.norm.cdf(normal_samples)

        elif self.copula_type == 't':
            # Generate multivariate t samples
            df_val = self.fitted_params['d']
            correlation = self.fitted_params['correlation']

            # Generate samples using Cholesky decomposition
            normal_samples = rng.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=correlation,
                size=n_samples
            )
            chi2_samples = rng.chisquare(df_val, n_samples)
            t_samples = normal_samples / \
                np.sqrt(chi2_samples[:, None] / df_val)

            # Convert to uniform
            uniform_samples = stats.t.cdf(t_samples, df_val)

        return uniform_samples


class FactorRiskModel:
    """Factor-based risk model for large portfolios"""

    def __init__(
        self,
        n_factors: int = 10,
        factor_type: str = 'pca',
        regularization: float = 0.01
    ):
        self.n_factors = n_factors
        self.factor_type = factor_type
        self.regularization = regularization
        self.factor_model = None
        self.factor_returns = None
        self.factor_loadings = None
        self.idiosyncratic_variance = None

    def fit(self, returns: np.ndarray) -> 'FactorRiskModel':
        """Fit factor model to return data"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for factor modeling")

        # Standardize returns
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)

        # Fit factor model
        if self.factor_type == 'pca':
            self.factor_model = PCA(n_components=self.n_factors)
            self.factor_returns = self.factor_model.fit_transform(
                returns_scaled)
            self.factor_loadings = self.factor_model.components_.T

        elif self.factor_type == 'factor_analysis':
            self.factor_model = FactorAnalysis(n_components=self.n_factors)
            self.factor_returns = self.factor_model.fit_transform(
                returns_scaled)
            self.factor_loadings = self.factor_model.components_.T

        # Calculate idiosyncratic variance
        factor_contribution = self.factor_returns @ self.factor_loadings.T
        residuals = returns_scaled - factor_contribution
        self.idiosyncratic_variance = np.var(residuals, axis=0)

        # Add regularization
        self.idiosyncratic_variance += self.regularization

        return self

    def predict_covariance(
            self, factor_covariance: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict asset covariance matrix"""
        if self.factor_loadings is None:
            raise ValueError(MODEL_NOT_FITTED_ERROR)

        if factor_covariance is None:
            # Use sample covariance of factors
            factor_covariance = np.cov(self.factor_returns.T)

        # Covariance = B @ F @ B.T + D
        # Where B = factor loadings, F = factor covariance, D = idiosyncratic
        # variance
        systematic_cov = self.factor_loadings @ factor_covariance @ self.factor_loadings.T
        total_cov = systematic_cov + np.diag(self.idiosyncratic_variance)

        return total_cov

    def risk_attribution(
            self, portfolio_weights: np.ndarray) -> Dict[str, float]:
        """Decompose portfolio risk into factor and idiosyncratic components"""
        if self.factor_loadings is None:
            raise ValueError(MODEL_NOT_FITTED_ERROR)

        factor_covariance = np.cov(self.factor_returns.T)

        # Portfolio factor exposures
        portfolio_loadings = portfolio_weights @ self.factor_loadings

        # Factor risk contribution
        factor_risk = portfolio_loadings @ factor_covariance @ portfolio_loadings

        # Idiosyncratic risk contribution
        idio_risk = portfolio_weights @ np.diag(
            self.idiosyncratic_variance) @ portfolio_weights

        total_risk = factor_risk + idio_risk

        return {
            'total_risk': total_risk,
            'factor_risk': factor_risk,
            'idiosyncratic_risk': idio_risk,
            'factor_contribution_pct': factor_risk / total_risk * 100,
            'idio_contribution_pct': idio_risk / total_risk * 100
        }


class MultiAssetForecaster:
    """Main class for multi-asset correlation forecasting"""

    def __init__(
        self,
        correlation_method: str = 'dcc',
        n_factors: int = 10,
        lookback_window: int = 252,
        min_observations: int = 100
    ):
        self.correlation_method = correlation_method
        self.n_factors = n_factors
        self.lookback_window = lookback_window
        self.min_observations = min_observations

        # Model components
        self.dynamic_model = None
        self.factor_model = None
        self.copula_model = None

        # Fitted parameters
        self.asset_names = None
        self.last_fit_date = None

    def fit(
        self,
        returns_data: pd.DataFrame,
        fit_copula: bool = True,
        fit_factor_model: bool = True
    ) -> 'MultiAssetForecaster':
        """Fit the multi-asset correlation model"""
        logger.info(
            f"Fitting multi-asset model with {len(returns_data.columns)} assets")

        # Validate data
        if len(returns_data) < self.min_observations:
            raise ValueError(f"Need at least {self.min_observations} observations")

        # Store asset information
        self.asset_names = list(returns_data.columns)
        self.last_fit_date = returns_data.index[-1]
        n_assets = len(self.asset_names)

        # Use recent data for fitting
        recent_data = returns_data.tail(self.lookback_window)
        returns_array = recent_data.values

        # Remove any NaN values
        mask = ~np.isnan(returns_array).any(axis=1)
        returns_clean = returns_array[mask]

        if len(returns_clean) < self.min_observations:
            raise ValueError("Insufficient clean data after removing NaNs")

        logger.info(f"Using {len(returns_clean)} clean observations for fitting")

        # Fit correlation model
        if self.correlation_method == 'dcc':
            self._fit_dcc_model(returns_clean)
        elif self.correlation_method == 'neural':
            self._fit_neural_model(returns_clean)
        elif self.correlation_method == 'ewma':
            self._fit_ewma_model(returns_clean)

        # Store last returns for all methods
        self.last_returns = returns_clean[-1]

        # Fit factor model if requested
        if fit_factor_model and HAS_SKLEARN:
            logger.info("Fitting factor model")
            self.factor_model = FactorRiskModel(
                n_factors=min(self.n_factors, n_assets - 1),
                factor_type='pca'
            )
            self.factor_model.fit(returns_clean)

        # Fit copula model if requested
        if fit_copula:
            logger.info("Fitting copula model")
            self.copula_model = CopulaModel(copula_type='gaussian')
            self.copula_model.fit(returns_clean)

        return self

    def _fit_dcc_model(self, returns: np.ndarray):
        """Fit Dynamic Conditional Correlation model"""
        if HAS_ARCH:
            try:
                # Use full arch package if available
                logger.info("Using arch package for GARCH modeling")
                # For now, use simple exponential smoothing as DCC proxy
                # Full DCC implementation would be more complex
                self._fit_ewma_model(returns)
            except Exception as e:
                logger.warning(
                    f"arch package failed: {e}, falling back to simple GARCH")
                self._fit_simple_dcc_model(returns)
        else:
            logger.info("Using simple GARCH implementation")
            self._fit_simple_dcc_model(returns)

    def _fit_simple_dcc_model(self, returns: np.ndarray):
        """Fit simplified DCC using our SimpleGARCH implementation"""
        _, n_assets = returns.shape

        # Step 1: Fit univariate GARCH models for each asset
        garch_models = []
        standardized_residuals = np.zeros_like(returns)

        for i in range(n_assets):
            # Fit simple GARCH to each asset
            garch = SimpleGARCH()
            garch.fit(returns[:, i])
            garch_models.append(garch)

            # Standardize residuals
            conditional_vol = garch.conditional_volatility
            standardized_residuals[:, i] = returns[:, i] / conditional_vol

        # Step 2: Estimate dynamic correlation
        # Use EWMA on standardized residuals
        correlation_matrix = np.corrcoef(standardized_residuals.T)

        # Store results
        self.garch_models = garch_models
        self.ewma_correlation = correlation_matrix
        self.ewma_decay = 0.94

        logger.info(f"Simple DCC fitted for {n_assets} assets")

    def _fit_neural_model(self, returns: np.ndarray):
        """Fit neural correlation model"""
        n_assets = returns.shape[1]

        # Initialize model
        self.dynamic_model = DynamicCorrelationModel(
            n_assets=n_assets,
            n_factors=min(self.n_factors, n_assets - 1)
        )

        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns)

        # Simple training loop (in practice, would be more sophisticated)
        optimizer = torch.optim.Adam(
            self.dynamic_model.parameters(),
            lr=0.001,
            weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Training target: sample correlation
        target_corr = torch.FloatTensor(np.corrcoef(returns.T)).unsqueeze(0)
        target_corr = target_corr.expand(len(returns), -1, -1)

        for epoch in range(100):  # Limited training for speed
            optimizer.zero_grad()

            # Forward pass
            pred_corr, _, _ = self.dynamic_model(returns_tensor)

            # Loss: correlation prediction error
            loss = criterion(pred_corr, target_corr)

            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    def _fit_ewma_model(self, returns: np.ndarray):

        # Adapter for UniversalFeatureAggregator
        def get_correlation_features(self, ticker: str) -> dict:
            """Return lightweight correlation summaries if model components exist; else empty dict.
            This avoids aggregator warnings without forcing a fit here.
            """
            out = {}
            try:
                if getattr(self, 'factor_model', None) is not None:
                    var = getattr(self.factor_model, 'explained_variance_ratio_', None)
                    if var is not None and len(var) > 0:
                        out['correlation_top_factor_var'] = float(var[0])
                        out['correlation_cum_factor_var_3'] = float(sum(var[:3]))
            except Exception:
                pass
            try:
                cov = getattr(self, 'last_cov', None)
                if cov is not None:
                    d = np.sqrt(np.diag(cov))
                    corr = cov / np.outer(d, d)
                    n = corr.shape[0]
                    if n > 1:
                        off = (np.sum(corr) - np.trace(corr)) / (n*(n-1))
                        out['correlation_offdiag_mean'] = float(off)
            except Exception:
                pass
            return out
        """Fit Exponentially Weighted Moving Average correlation model"""
        # Simple EWMA with decay factor
        decay = 0.94

        # Initialize with sample correlation
        correlation_matrix = np.corrcoef(returns.T)

        # Store for prediction
        self.ewma_correlation = correlation_matrix
        self.ewma_decay = decay

    def forecast_correlation(
        self,
        horizon: int = 1,
        method: str = 'auto',
        confidence_level: float = 0.95
    ) -> CorrelationForecast:
        """Forecast correlation matrix"""
        if self.asset_names is None:
            raise ValueError(MODEL_NOT_FITTED_ERROR)

        n_assets = len(self.asset_names)

        # Choose forecasting method
        if method == 'auto':
            method = self.correlation_method

        # Generate forecast
        if method == 'neural' and self.dynamic_model is not None:
            correlation_matrix = self._forecast_neural()
        elif method == 'ewma':
            correlation_matrix = self._forecast_ewma(horizon)
        elif method == 'factor' and self.factor_model is not None:
            correlation_matrix = self._forecast_factor(horizon)
        else:
            # Fallback to sample correlation
            correlation_matrix = self.ewma_correlation

        # Ensure positive semi-definite
        correlation_matrix = self._ensure_psd(correlation_matrix)

        # Generate confidence intervals if requested
        conf_intervals = None
        if self.copula_model is not None:
            conf_intervals = self._generate_confidence_intervals(
                confidence_level)

        # Risk attribution if factor model available
        risk_attr = None
        if self.factor_model is not None:
            # Equal weight portfolio for demonstration
            equal_weights = np.ones(n_assets) / n_assets
            risk_attr = self.factor_model.risk_attribution(equal_weights)

        return CorrelationForecast(
            correlation_matrix=correlation_matrix,
            factor_loadings=getattr(
                self.factor_model,
                'factor_loadings',
                None),
            idiosyncratic_variance=getattr(
                self.factor_model,
                'idiosyncratic_variance',
                None),
            risk_attribution=risk_attr,
            confidence_interval=conf_intervals,
            forecast_date=datetime.now(),
            methodology=method)

    def _forecast_neural(self) -> np.ndarray:
        """Neural model forecast"""
        if self.dynamic_model is None:
            raise ValueError("Neural model not fitted")

        # Use last observation for prediction
        last_returns = torch.FloatTensor(self.last_returns).unsqueeze(0)

        with torch.no_grad():
            pred_corr, _, _ = self.dynamic_model(last_returns)

        return pred_corr.squeeze(0).numpy()

    def _forecast_ewma(self, horizon: int) -> np.ndarray:
        """EWMA forecast (assumes correlation persistence)"""
        # For longer horizons, correlation slowly reverts to long-term average
        reversion_factor = 0.95 ** horizon

        # Create random generator for reproducibility
        rng = np.random.default_rng(42)
        long_term_corr = np.corrcoef(
            rng.standard_normal((100, len(self.asset_names))).T)

        forecast_corr = (reversion_factor * self.ewma_correlation +
                         (1 - reversion_factor) * long_term_corr)

        return forecast_corr

    def _forecast_factor(self, horizon: int) -> np.ndarray:
        """Factor model forecast"""
        if self.factor_model is None:
            raise ValueError("Factor model not fitted")

        # Simple factor covariance forecast (could be more sophisticated)
        factor_cov = np.cov(self.factor_model.factor_returns.T)

        # Slight mean reversion for longer horizons
        if horizon > 1:
            factor_cov *= (0.98 ** (horizon - 1))

        covariance_matrix = self.factor_model.predict_covariance(factor_cov)

        # Convert to correlation
        std_dev = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_dev, std_dev)

        return correlation_matrix

    def _ensure_psd(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive semi-definite"""
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)

        # Clip negative eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-8)

        # Reconstruct matrix
        psd_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Ensure diagonal is 1 (correlation matrix)
        std_dev = np.sqrt(np.diag(psd_matrix))
        psd_matrix = psd_matrix / np.outer(std_dev, std_dev)
        np.fill_diagonal(psd_matrix, 1.0)

        return psd_matrix

    def _generate_confidence_intervals(
        self,
        confidence_level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate confidence intervals using copula bootstrap"""
        if self.copula_model is None:
            return None, None

        n_bootstrap = 1000

        # Bootstrap correlations
        bootstrap_correlations = []

        for _ in range(n_bootstrap):
            # Generate copula samples
            uniform_samples = self.copula_model.generate_samples(100)

            # Convert to returns (simplified)
            pseudo_returns = stats.norm.ppf(uniform_samples)

            # Calculate correlation
            boot_corr = np.corrcoef(pseudo_returns.T)
            bootstrap_correlations.append(boot_corr)

        bootstrap_correlations = np.array(bootstrap_correlations)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(
            bootstrap_correlations, lower_percentile, axis=0)
        upper_bound = np.percentile(
            bootstrap_correlations, upper_percentile, axis=0)

        return lower_bound, upper_bound

    def portfolio_risk_forecast(
        self,
        weights: np.ndarray,
        horizon: int = 1,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Forecast portfolio risk metrics"""
        # Get correlation forecast
        corr_forecast = self.forecast_correlation(horizon)

        # Portfolio variance
        portfolio_var = weights @ corr_forecast.correlation_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var * 252)  # Annualized

        # Value at Risk (assuming normal distribution)
        z_score = stats.norm.ppf(1 - confidence_level)
        var_daily = -z_score * np.sqrt(portfolio_var)
        var_horizon = var_daily * np.sqrt(horizon)

        results = {
            'portfolio_volatility_annual': portfolio_vol,
            'portfolio_variance_daily': portfolio_var,
            'value_at_risk': {
                f'{int(confidence_level*100)}%_var_daily': var_daily,
                f'{int(confidence_level*100)}%_var_{horizon}d': var_horizon
            },
            'correlation_matrix': corr_forecast.correlation_matrix,
            'forecast_horizon': horizon,
            'methodology': corr_forecast.methodology
        }

        # Add factor attribution if available
        if corr_forecast.risk_attribution:
            results['risk_attribution'] = corr_forecast.risk_attribution

        return results


def test_multiasset_system():
    """Test the multi-asset correlation system"""
    print("=== TESTING MULTI-ASSET CORRELATION SYSTEM ===")

    # Generate synthetic multi-asset data
    # Use modern numpy random generator instead of deprecated global state
    rng = np.random.default_rng(42)
    # Also set legacy global seed for compatibility
    np.random.seed(42)
    n_assets = 5
    n_obs = 500

    # Create correlated returns
    true_correlation = np.array([
        [1.0, 0.6, 0.4, 0.3, 0.2],
        [0.6, 1.0, 0.5, 0.3, 0.1],
        [0.4, 0.5, 1.0, 0.4, 0.3],
        [0.3, 0.3, 0.4, 1.0, 0.5],
        [0.2, 0.1, 0.3, 0.5, 1.0]
    ])

    # Generate correlated returns using Cholesky decomposition
    L = np.linalg.cholesky(true_correlation)
    rng = np.random.default_rng(42)
    independent_returns = rng.normal(0, 0.02, (n_obs, n_assets))
    correlated_returns = independent_returns @ L.T

    # Create DataFrame
    asset_names = [f'ASSET_{i+1}' for i in range(n_assets)]
    dates = pd.date_range('2020-01-01', periods=n_obs, freq='D')
    returns_df = pd.DataFrame(
        correlated_returns,
        index=dates,
        columns=asset_names)

    print(f"Generated {n_obs} observations for {n_assets} assets")
    print("Sample correlation matrix:")
    print(np.corrcoef(correlated_returns.T).round(3))

    # Test different correlation methods
    methods = ['ewma', 'neural'] if torch is not None else ['ewma']

    for method in methods:
        print(f"\n--- Testing {method.upper()} method ---")

        try:
            # Initialize forecaster
            forecaster = MultiAssetForecaster(
                correlation_method=method,
                n_factors=3,
                lookback_window=252
            )

            # Fit model
            forecaster.fit(returns_df)
            print("✓ Model fitted successfully")

            # Generate forecast
            forecast = forecaster.forecast_correlation(horizon=1)
            print("✓ Correlation forecast generated")
            print(f"  Methodology: {forecast.methodology}")
            print("  Forecast correlation matrix:")
            print(forecast.correlation_matrix.round(3))

            # Portfolio risk forecast
            equal_weights = np.ones(n_assets) / n_assets
            portfolio_risk = forecaster.portfolio_risk_forecast(equal_weights)
            print("✓ Portfolio risk forecast:")
            print(f"  Annual volatility: {portfolio_risk['portfolio_volatility_annual']:.2%}")
            print(f"  95% VaR (daily): {portfolio_risk['value_at_risk']['95%_var_daily']:.4f}")

            if portfolio_risk.get('risk_attribution'):
                attr = portfolio_risk['risk_attribution']
                print(f"  Factor contribution: {attr.get('factor_contribution_pct', 0):.1f}%")
                print(f"  Idiosyncratic contribution: {attr.get('idio_contribution_pct', 0):.1f}%")

        except Exception as e:
            print(f"✗ Error with {method} method: {str(e)}")

    # Test factor model separately
    if HAS_SKLEARN:
        print("\n--- Testing Factor Model ---")
        try:
            factor_model = FactorRiskModel(n_factors=3)
            factor_model.fit(correlated_returns)

            # Risk attribution
            equal_weights = np.ones(n_assets) / n_assets
            attribution = factor_model.risk_attribution(equal_weights)

            print("✓ Factor model fitted")
            print(f"  Explained variance ratio: {factor_model.factor_model.explained_variance_ratio_.sum():.2%}")
            print(f"  Factor risk contribution: {attribution['factor_contribution_pct']:.1f}%")
            print(f"  Idiosyncratic risk: {attribution['idio_contribution_pct']:.1f}%")

        except Exception as e:
            print(f"✗ Factor model error: {str(e)}")

    # Test copula model
    print("\n--- Testing Copula Model ---")
    try:
        copula = CopulaModel(copula_type='gaussian')
        copula.fit(correlated_returns)

        # Generate samples
        samples = copula.generate_samples(1000)
        print("✓ Copula model fitted and tested")
        print(f"  Generated {len(samples)} samples")
        print("  Sample correlation of generated data:")
        # Convert back to normal for correlation
        normal_samples = stats.norm.ppf(samples)
        sample_corr = np.corrcoef(normal_samples.T)
        print(sample_corr.round(3))

    except Exception as e:
        print(f"✗ Copula model error: {str(e)}")

    print("\n=== MULTI-ASSET CORRELATION TESTING COMPLETE ===")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_multiasset_system()
