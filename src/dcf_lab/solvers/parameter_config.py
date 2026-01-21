"""
Parameter configuration for correlated parameter solver
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from dcf_lab.valuation.distributions import DistributionSpec


class ParameterConfig:
    """
    Configuration for parameter solving
    """

    def __init__(
        self,
        parameters: List[str],
        initial_means: Optional[Dict[str, float]] = None,
        initial_stds: Optional[Dict[str, float]] = None,
        initial_correlation: float = 0.0,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.parameters = parameters
        self.initial_means = initial_means or {}
        self.initial_stds = initial_stds or {}
        self.initial_correlation = initial_correlation
        self.bounds = bounds or {}

        # Set default values
        self._set_defaults()

    def _set_defaults(self) -> None:
        """
        Set default values for missing parameters
        """
        default_initial_means = {
            'wacc': 0.09,
            'growth_rate': 0.05,
            'ebit_margin': 0.15,
            'terminal_growth': 0.02
        }

        default_initial_stds = {
            'wacc': 0.01,
            'growth_rate': 0.01,
            'ebit_margin': 0.02,
            'terminal_growth': 0.005
        }

        default_bounds = {
            'wacc': (0.05, 0.2),
            'growth_rate': (0.01, 0.2),
            'ebit_margin': (0.05, 0.4),
            'terminal_growth': (0.01, 0.04)
        }

        # Fill in missing values
        for param in self.parameters:
            if param not in self.initial_means:
                self.initial_means[param] = default_initial_means.get(
                    param, 0.05)
            if param not in self.initial_stds:
                self.initial_stds[param] = default_initial_stds.get(
                    param, 0.01)
            if param not in self.bounds:
                self.bounds[param] = default_bounds.get(param, (0.01, 0.2))

    def create_initial_correlation_matrix(self) -> np.ndarray:
        """
        Create initial correlation matrix

        Returns:
            Correlation matrix with the specified initial correlation
        """
        n_params = len(self.parameters)
        correlation_matrix = np.eye(n_params)

        for i in range(n_params):
            for j in range(i+1, n_params):
                correlation_matrix[i, j] = self.initial_correlation
                correlation_matrix[j, i] = self.initial_correlation

        return correlation_matrix

    def get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for optimization

        Returns:
            List of (lower, upper) bounds for each parameter
        """
        n_params = len(self.parameters)
        n_corrs = n_params * (n_params - 1) // 2

        # Bounds for standard deviations
        std_bounds = [(0.001, 0.2)] * n_params

        # Bounds for correlations
        corr_bounds = [(-0.95, 0.95)] * n_corrs

        return std_bounds + corr_bounds

    def create_initial_params_vector(self) -> np.ndarray:
        """
        Create initial parameter vector for optimization

        Returns:
            Initial parameter vector (stds + correlations)
        """
        n_params = len(self.parameters)
        n_corrs = n_params * (n_params - 1) // 2

        # Create vector
        initial_params = np.zeros(n_params + n_corrs)

        # Set initial standard deviations
        for i, param in enumerate(self.parameters):
            initial_params[i] = self.initial_stds[param]

        # Set initial correlations
        initial_params[n_params:] = self.initial_correlation

        return initial_params

    def create_distribution_specs(
        self,
        stds: np.ndarray
    ) -> Dict[str, DistributionSpec]:
        """
        Create distribution specifications

        Args:
            stds: Standard deviations for each parameter

        Returns:
            Dictionary of distribution specifications
        """
        dist_specs = {}
        for i, param in enumerate(self.parameters):
            dist_specs[param] = DistributionSpec(
                dist_type='normal',
                params={'mean': self.initial_means[param], 'std': stds[i]},
                bounds=self.bounds[param]
            )

        return dist_specs
