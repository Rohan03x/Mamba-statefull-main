"""
Correlated parameter solver
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from dcf_lab.solvers.correlation_matrix import (
    build_correlation_matrix,
    ensure_valid_correlation_matrix,
)
from dcf_lab.solvers.model_configurator import ModelConfigurator
from dcf_lab.solvers.objective_function import OptimizationObjective
from dcf_lab.solvers.parameter_config import ParameterConfig
from dcf_lab.valuation.distributions import DistributionSpec


class CorrelatedParameterSolver:
    """
    Solver for correlated parameter distributions
    """

    def __init__(
            self,
            dcf_model,
            base_revenue: float,
            current_price: float,
            price_std: float):
        """
        Initialize solver

        Args:
            dcf_model: DCF model instance
            base_revenue: Base revenue value
            current_price: Current stock price
            price_std: Price standard deviation
        """
        self.dcf_model = dcf_model
        self.base_revenue = base_revenue
        self.current_price = current_price
        self.price_std = price_std

    def solve(
        self,
        parameters: List[str],
        initial_means: Optional[Dict[str, float]] = None,
        initial_stds: Optional[Dict[str, float]] = None,
        initial_correlation: float = 0.0,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[Dict[str, DistributionSpec], np.ndarray, Dict]:
        """
        Solve for correlated parameter distributions

        Args:
            parameters: List of parameter names
            initial_means: Initial parameter means
            initial_stds: Initial parameter standard deviations
            initial_correlation: Initial correlation value
            bounds: Parameter bounds

        Returns:
            Distribution specifications, correlation matrix, and simulation results
        """
        if len(parameters) < 2:
            raise ValueError("At least two parameters must be provided")

        # Step 1: Create parameter configuration
        param_config = ParameterConfig(
            parameters=parameters,
            initial_means=initial_means,
            initial_stds=initial_stds,
            initial_correlation=initial_correlation,
            bounds=bounds
        )

        # Step 2: Update initial means using reverse DCF solver
        self._update_initial_means(param_config)

        # Step 3: Create model configurator
        model_configurator = ModelConfigurator(self.dcf_model, parameters)

        # Step 4: Create objective function
        objective = OptimizationObjective(
            param_config=param_config,
            model_configurator=model_configurator,
            base_revenue=self.base_revenue,
            target_price=self.current_price,
            target_price_std=self.price_std
        )

        # Step 5: Run optimization
        initial_params = param_config.create_initial_params_vector()
        bounds = param_config.get_optimization_bounds()

        opt_stds, opt_corr_matrix = self._run_optimization(
            objective_function=objective,
            initial_params=initial_params,
            bounds=bounds,
            n_params=len(parameters),
            initial_stds=param_config.initial_stds,
            initial_correlation_matrix=param_config.create_initial_correlation_matrix()
        )

        # Step 6: Create final distributions
        final_dists = self._create_final_distributions(
            parameters=parameters,
            initial_means=param_config.initial_means,
            opt_stds=opt_stds,
            bounds=param_config.bounds
        )

        # Step 7: Configure model with final parameters and run
        model_configurator.configure_model(final_dists, opt_corr_matrix)
        results = model_configurator.run_simulation(self.base_revenue)

        return final_dists, opt_corr_matrix, results

    def _update_initial_means(self, param_config: ParameterConfig) -> None:
        """
        Update initial means using reverse DCF solver

        Args:
            param_config: Parameter configuration
        """
        # Import here to avoid circular import
        from dcf_lab.valuation.reverse_dcf import ReverseDCFSolver

        # Create solver
        solver = ReverseDCFSolver(
            dcf_model=self.dcf_model,
            base_revenue=self.base_revenue,
            current_price=self.current_price
        )

        # Solve for means
        for param in param_config.parameters:
            try:
                if param == 'wacc':
                    mean, _ = solver.solve_for_wacc()
                    param_config.initial_means[param] = mean
                elif param == 'growth_rate':
                    mean, _ = solver.solve_for_growth()
                    # Convert factor to rate
                    param_config.initial_means[param] = mean - 1.0
                elif param == 'ebit_margin':
                    mean, _ = solver.solve_for_margin()
                    param_config.initial_means[param] = mean
                elif param == 'terminal_growth':
                    mean, _ = solver.solve_for_terminal_growth()
                    param_config.initial_means[param] = mean
            except Exception as e:
                print(
                    f"Warning: Could not solve for mean of {param}, using initial guess: {e}")

    def _run_optimization(
        self,
        objective_function: OptimizationObjective,
        initial_params: np.ndarray,
        bounds: List[Tuple[float, float]],
        n_params: int,
        initial_stds: Dict[str, float],
        initial_correlation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run optimization

        Args:
            objective_function: Objective function
            initial_params: Initial parameter vector
            bounds: Parameter bounds
            n_params: Number of parameters
            initial_stds: Initial standard deviations
            initial_correlation_matrix: Initial correlation matrix

        Returns:
            Optimized standard deviations and correlation matrix
        """
        try:
            # Run scipy optimization
            result = optimize.minimize(
                objective_function,
                initial_params,
                bounds=bounds,
                method='L-BFGS-B'
            )

            if not result.success:
                print(
                    "Warning: Optimization did not converge, using best found solution")

            # Extract optimized parameters
            opt_stds = result.x[:n_params]
            opt_corrs = result.x[n_params:]

            # Reconstruct correlation matrix
            opt_corr_matrix = build_correlation_matrix(n_params, opt_corrs)

            # Ensure matrix is valid
            opt_corr_matrix = ensure_valid_correlation_matrix(opt_corr_matrix)

            return opt_stds, opt_corr_matrix

        except Exception as e:
            # Fallback if optimization fails
            print(f"Warning: Optimization failed, using initial values: {e}")

            # Use initial values
            opt_stds = np.array(list(initial_stds.values()))

            return opt_stds, initial_correlation_matrix

    def _create_final_distributions(
        self,
        parameters: List[str],
        initial_means: Dict[str, float],
        opt_stds: np.ndarray,
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, DistributionSpec]:
        """
        Create final distribution specifications

        Args:
            parameters: List of parameter names
            initial_means: Initial parameter means
            opt_stds: Optimized standard deviations
            bounds: Parameter bounds

        Returns:
            Dictionary of distribution specifications
        """
        final_dists = {}
        for i, param in enumerate(parameters):
            final_dists[param] = DistributionSpec(
                dist_type='normal',
                params={'mean': initial_means[param], 'std': opt_stds[i]},
                bounds=bounds[param]
            )

        return final_dists
