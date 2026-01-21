"""
Model configuration and simulation
"""
from typing import Any, Dict, List

import numpy as np

from dcf_lab.valuation.distributions import DistributionSpec


class ModelConfigurator:
    """
    Configures DCF model with parameter distributions
    """

    def __init__(self, dcf_model, parameters: List[str]):
        """
        Initialize with DCF model and parameter list

        Args:
            dcf_model: DCF model instance
            parameters: List of parameter names
        """
        self.dcf_model = dcf_model
        self.parameters = parameters

    def configure_model(
        self,
        dist_specs: Dict[str, DistributionSpec],
        correlation_matrix: np.ndarray
    ) -> None:
        """
        Configure DCF model with distribution specifications

        Args:
            dist_specs: Dictionary of distribution specifications
            correlation_matrix: Correlation matrix
        """
        # Set parameters in DCF model
        for param in self.parameters:
            self._set_model_parameter(param, dist_specs[param])

        # Set correlation matrix
        self.dcf_model.set_correlation_matrix(
            correlation_matrix, self.parameters)

    def _set_model_parameter(
            self,
            param: str,
            dist_spec: DistributionSpec) -> None:
        """
        Set a specific parameter in the DCF model

        Args:
            param: Parameter name
            dist_spec: Distribution specification
        """
        if param == 'wacc':
            self.dcf_model.set_wacc(dist_spec)
        elif param == 'growth_rate':
            # Convert rate to factor (1+g)
            growth_dist = self._convert_growth_rate_to_factor(dist_spec)
            self.dcf_model.set_revenue_growth(growth_dist)
        elif param == 'ebit_margin':
            self.dcf_model.set_ebit_margin(dist_spec)
        elif param == 'terminal_growth':
            self.dcf_model.set_terminal_growth(dist_spec)

    def _convert_growth_rate_to_factor(
            self, dist_spec: DistributionSpec) -> DistributionSpec:
        """
        Convert growth rate distribution to growth factor distribution

        Args:
            dist_spec: Growth rate distribution specification

        Returns:
            Growth factor distribution specification
        """
        return DistributionSpec(
            dist_type='normal',
            params={
                'mean': 1.0 + dist_spec.params['mean'],
                'std': dist_spec.params['std']
            },
            bounds=(1.0 + dist_spec.bounds[0], 1.0 + dist_spec.bounds[1])
        )

    def run_simulation(self, base_revenue: float) -> Dict[str, Any]:
        """
        Run DCF simulation

        Args:
            base_revenue: Base revenue value

        Returns:
            Simulation results
        """
        return self.dcf_model.run_simulation(base_revenue)
