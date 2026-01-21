"""
Objective function for correlated parameter optimization
"""
import numpy as np

from dcf_lab.solvers.correlation_matrix import (
    build_correlation_matrix,
    ensure_valid_correlation_matrix,
)
from dcf_lab.solvers.model_configurator import ModelConfigurator
from dcf_lab.solvers.parameter_config import ParameterConfig


class OptimizationObjective:
    """
    Objective function for correlated parameter optimization
    """

    def __init__(
        self,
        param_config: ParameterConfig,
        model_configurator: ModelConfigurator,
        base_revenue: float,
        target_price: float,
        target_price_std: float
    ):
        """
        Initialize objective function

        Args:
            param_config: Parameter configuration
            model_configurator: Model configurator
            base_revenue: Base revenue value
            target_price: Target price mean
            target_price_std: Target price standard deviation
        """
        self.param_config = param_config
        self.model_configurator = model_configurator
        self.base_revenue = base_revenue
        self.target_price = target_price
        self.target_price_std = target_price_std
        self.parameters = param_config.parameters
        self.n_params = len(self.parameters)

    def __call__(self, params: np.ndarray) -> float:
        """
        Calculate objective function value

        Args:
            params: Array with parameter stds followed by correlation values

        Returns:
            Objective function value (lower is better)
        """
        # Extract parameter stds and correlation values
        param_stds = params[:self.n_params]
        corr_values = params[self.n_params:]

        # Build correlation matrix
        corr_matrix = build_correlation_matrix(self.n_params, corr_values)

        # Ensure correlation matrix is valid
        corr_matrix = ensure_valid_correlation_matrix(corr_matrix)

        # Create distribution specifications
        dist_specs = self.param_config.create_distribution_specs(param_stds)

        # Configure model and run simulation
        self.model_configurator.configure_model(dist_specs, corr_matrix)
        self.model_configurator.run_simulation(self.base_revenue)

        # Calculate model price statistics
        model_values = self.model_configurator.dcf_model.per_share_values
        model_mean = np.median(model_values)
        model_std = np.std(model_values)

        # Calculate the error (distance from target)
        mean_error = abs(model_mean - self.target_price) / self.target_price
        std_error = abs(model_std - self.target_price_std) / \
            self.target_price_std

        # Combined error (weighted towards matching the mean)
        return mean_error * 3 + std_error
