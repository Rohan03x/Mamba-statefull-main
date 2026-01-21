"""
Reverse DCF Solver

This module implements solvers to extract market-implied assumptions from current stock prices.
It allows understanding what growth rates, margins, and other drivers the market is currently
pricing in, which can be useful for calibrating forward-looking assumptions.

Key features include:

1. Reverse DCF Solver
   - Extract implied growth rates, margins, and WACC from current stock prices
   - Solve for individual parameters or multiple parameters simultaneously
   - Support for multi-period parameter solving

2. Market Implied Assumptions
   - Extract and analyze all key market-implied assumptions
   - Scenario analysis (base, bull, bear)
   - Compare implied values with historical actuals
   - Visualize implied vs. actual parameters

3. Advanced Distribution Solving
   - Extract parameter distributions that match observed price ranges
   - Solve for correlated parameter distributions
   - Derive implied WACC distributions for more realistic modeling

4. Market Calibration Metrics
   - Compare market-implied distributions with forecast model distributions
   - Calculate statistical measures of distribution alignment
   - Visualize the difference between market consensus and forecasts

5. Enhanced Visualization
   - Parameter sensitivity analysis
   - Multivariate sensitivity analysis
   - Distribution comparison plots
   - Calibration visualization

Example usage:
```python
# Create market-implied assumptions analyzer
market_implied = MarketImpliedAssumptions(
    ticker='AAPL',
    base_revenue=400000,
    current_price=150,
    shares_outstanding=16000,
    total_debt=100000,
    total_cash=50000
)

# Set base assumptions
market_implied.set_base_assumptions(
    ebit_margin=0.25,
    tax_rate=0.15,
    capex_percent=0.05,
    nwc_percent=0.10,
    da_percent=0.04,
    wacc=0.09,
    terminal_growth=0.025,
    terminal_multiple=15.0
)

# Calculate implied growth rate
implied_growth, _ = market_implied.implied_growth_rate()
print(f"Implied growth rate: {(implied_growth - 1) * 100:.2f}%")

# Calculate all implied assumptions
all_implied = market_implied.calculate_all_implied_assumptions()

# Plot scenarios
fig = market_implied.plot_scenarios()
```

See the examples/reverse_dcf_example.ipynb for a complete demonstration.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

from .dcf_mc import DistributionSpec, MonteCarloDCF, constant_value

# Constants
SOLVER_CONVERGENCE_ERROR = "Solver did not converge"
VALUE_PER_SHARE_LABEL = 'Value per Share ($)'
AXES_FRACTION = 'axes fraction'


class ReverseDCFSolver:
    """
    Solver to extract market-implied assumptions from current stock price

    This class allows solving for one or more DCF model parameters that would make
    the model output match the current market price, thereby revealing what
    assumptions are "priced in" by the market.
    """

    def __init__(
        self,
        dcf_model: MonteCarloDCF,
        base_revenue: float,
        current_price: float,
        base_year: int = 0
    ):
        """
        Initialize the reverse DCF solver

        Args:
            dcf_model: Monte Carlo DCF model with all parameters set except the one(s) to solve for
            base_revenue: Starting revenue value
            current_price: Current stock price
            base_year: Base year for calculations
        """
        self.dcf_model = dcf_model
        self.base_revenue = base_revenue
        self.current_price = current_price
        self.base_year = base_year

    def solve_for_growth(
        self,
        _initial_guess: float = 1.05,  # Unused, kept for backward compatibility
        growth_bounds: Tuple[float, float] = (0.98, 1.2),
        _constant_growth: bool = True  # Unused, kept for backward compatibility
    ) -> Tuple[float, Dict]:
        """
        Solve for implied revenue growth rate

        Args:
            _initial_guess: Initial guess for growth factor (e.g., 1.05 for 5% growth) - deprecated
            growth_bounds: Bounds for the growth factor
            _constant_growth: If True, apply the same growth rate to all periods - deprecated

        Returns:
            Implied growth rate and full model results
        """
        def objective_function(growth_factor):
            # Create a constant growth distribution
            growth_dist = constant_value(growth_factor)

            # Set the growth rate in the DCF model
            self.dcf_model.set_revenue_growth(growth_dist)

            # Run the simulation
            self.dcf_model.run_simulation(self.base_revenue, self.base_year)

            # Calculate the median per-share value
            median_value = np.median(self.dcf_model.per_share_values)

            # Return the difference from the current price
            return median_value - self.current_price

        # Solve for the growth rate using scipy's root-finding
        solution = optimize.root_scalar(
            objective_function,
            bracket=growth_bounds,
            method='brentq'
        )

        if not solution.converged:
            raise ValueError(SOLVER_CONVERGENCE_ERROR)

        # Get the solution
        implied_growth = solution.root

        # Run the model with the implied growth rate
        self.dcf_model.set_revenue_growth(constant_value(implied_growth))
        results = self.dcf_model.run_simulation(
            self.base_revenue, self.base_year)

        return implied_growth, results

    def solve_for_margin(
        self,
        _initial_guess: float = 0.15,  # Unused, kept for backward compatibility
        margin_bounds: Tuple[float, float] = (0.05, 0.5),
        _constant_margin: bool = True  # Unused, kept for backward compatibility
    ) -> Tuple[float, Dict]:
        """
        Solve for implied EBIT margin

        Args:
            _initial_guess: Initial guess for EBIT margin - deprecated
            margin_bounds: Bounds for the margin
            _constant_margin: If True, apply the same margin to all periods - deprecated

        Returns:
            Implied margin and full model results
        """
        def objective_function(margin):
            # Create a constant margin distribution
            margin_dist = constant_value(margin)

            # Set the margin in the DCF model
            self.dcf_model.set_ebit_margin(margin_dist)

            # Run the simulation
            self.dcf_model.run_simulation(self.base_revenue, self.base_year)

            # Calculate the median per-share value
            median_value = np.median(self.dcf_model.per_share_values)

            # Return the difference from the current price
            return median_value - self.current_price

        # Solve for the margin using scipy's root-finding
        solution = optimize.root_scalar(
            objective_function,
            bracket=margin_bounds,
            method='brentq'
        )

        if not solution.converged:
            raise ValueError(SOLVER_CONVERGENCE_ERROR)

        # Get the solution
        implied_margin = solution.root

        # Run the model with the implied margin
        self.dcf_model.set_ebit_margin(constant_value(implied_margin))
        results = self.dcf_model.run_simulation(
            self.base_revenue, self.base_year)

        return implied_margin, results

    def solve_for_wacc(
        self,
        _initial_guess: float = 0.09,  # Unused, kept for backward compatibility
        wacc_bounds: Tuple[float, float] = (0.05, 0.2)
    ) -> Tuple[float, Dict]:
        """
        Solve for implied weighted average cost of capital

        Args:
            _initial_guess: Initial guess for WACC - deprecated
            wacc_bounds: Bounds for the WACC

        Returns:
            Implied WACC and full model results
        """
        def objective_function(wacc):
            # Create a constant WACC distribution
            wacc_dist = constant_value(wacc)

            # Set the WACC in the DCF model
            self.dcf_model.set_wacc(wacc_dist)

            # Run the simulation
            self.dcf_model.run_simulation(self.base_revenue, self.base_year)

            # Calculate the median per-share value
            median_value = np.median(self.dcf_model.per_share_values)

            # Return the difference from the current price
            return median_value - self.current_price

        # Solve for the WACC using scipy's root-finding
        solution = optimize.root_scalar(
            objective_function,
            bracket=wacc_bounds,
            method='brentq'
        )

        if not solution.converged:
            raise ValueError(SOLVER_CONVERGENCE_ERROR)

        # Get the solution
        implied_wacc = solution.root

        # Run the model with the implied WACC
        self.dcf_model.set_wacc(constant_value(implied_wacc))
        results = self.dcf_model.run_simulation(
            self.base_revenue, self.base_year)

        return implied_wacc, results

    def solve_for_wacc_distribution(
        self,
        _dist_type: str = 'normal',  # Unused, kept for backward compatibility
        price_range: Tuple[float, float] = None,
        price_std: float = None,
        initial_mean: float = 0.09,
        initial_std: float = 0.01,
        wacc_bounds: Tuple[float, float] = (0.05, 0.2)
    ) -> Tuple[DistributionSpec, Dict]:
        """
        Solve for implied WACC distribution that matches a price range or distribution

        This function finds a WACC distribution (currently supporting normal distributions)
        that would make the model's price output match an observed price range or price
        distribution. It solves for both the mean and standard deviation of the WACC.

        Args:
            dist_type: Type of distribution ('normal' currently supported)
            price_range: Tuple of (min_price, max_price) to match with 95% confidence
            price_std: Standard deviation of observed prices (alternative to price_range)
            initial_mean: Initial guess for WACC mean
            initial_std: Initial guess for WACC standard deviation
            wacc_bounds: Bounds for the WACC mean value

        Returns:
            WACC distribution specification and full model results

        Note:
            Either price_range or price_std must be provided, but not both.
        """
        if price_range is None and price_std is None:
            raise ValueError(
                "Either price_range or price_std must be provided")

        if price_range is not None and price_std is not None:
            raise ValueError(
                "Only one of price_range or price_std should be provided")

        # If price range is provided, convert to standard deviation (assuming
        # normal)
        if price_range is not None:
            # 95% confidence interval is approximately mean ± 2*std
            price_std = (price_range[1] - price_range[0]) / 4

        # First solve for the mean WACC using the standard solve_for_wacc
        # method
        mean_wacc, _ = self.solve_for_wacc(
            _initial_guess=initial_mean, wacc_bounds=wacc_bounds)

        def objective_function(std_wacc):
            """Objective function to match price standard deviation"""
            # Create a normal distribution for WACC
            wacc_dist = DistributionSpec(
                dist_type='normal',
                params={'mean': mean_wacc, 'std': std_wacc},
                bounds=wacc_bounds
            )

            # Set the WACC distribution in the DCF model
            self.dcf_model.set_wacc(wacc_dist)

            # Run the simulation
            self.dcf_model.run_simulation(self.base_revenue, self.base_year)

            # Calculate the standard deviation of per-share values
            model_std = np.std(self.dcf_model.per_share_values)

            # Return the difference from the target price standard deviation
            return model_std - price_std

        # Use a numerical solver to find the standard deviation
        # Initialize search bounds based on initial guess
        std_bounds = (initial_std * 0.1, initial_std * 10)

        # Try to find a bracket where the objective function changes sign
        max_attempts = 10
        for _ in range(max_attempts):
            if objective_function(
                    std_bounds[0]) * objective_function(std_bounds[1]) < 0:
                # Found a bracket
                break

            # Expand search space
            std_bounds = (std_bounds[0] * 0.5, std_bounds[1] * 2)

            # If bounds get too extreme, use a different approach
            if std_bounds[0] < 1e-5 or std_bounds[1] > 0.5:
                break

        try:
            # Try Brent's method if we have a proper bracket
            if objective_function(
                    std_bounds[0]) * objective_function(std_bounds[1]) < 0:
                solution = optimize.root_scalar(
                    objective_function,
                    bracket=std_bounds,
                    method='brentq'
                )
                std_wacc = solution.root
            else:
                # If no bracket, use minimize to find the absolute minimum
                solution = optimize.minimize_scalar(
                    lambda x: abs(objective_function(x)),
                    bounds=std_bounds,
                    method='bounded'
                )
                std_wacc = solution.x
        except Exception as e:
            # Fallback if optimization fails
            std_wacc = initial_std
            error_msg = SOLVER_CONVERGENCE_ERROR.lower()
            print(f"Warning: WACC distribution {error_msg}, using initial std: {e}")

        # Create the distribution specification
        wacc_dist = DistributionSpec(
            dist_type='normal',
            params={'mean': mean_wacc, 'std': std_wacc},
            bounds=wacc_bounds
        )

        # Run the model with the implied WACC distribution
        self.dcf_model.set_wacc(wacc_dist)
        results = self.dcf_model.run_simulation(
            self.base_revenue, self.base_year)

        return wacc_dist, results

    def solve_for_terminal_growth(
        self,
        _initial_guess: float = 0.02,  # Unused, kept for backward compatibility
        growth_bounds: Tuple[float, float] = (0.01, 0.04)
    ) -> Tuple[float, Dict]:
        """
        Solve for implied terminal growth rate

        Args:
            initial_guess: Initial guess for terminal growth rate
            growth_bounds: Bounds for the terminal growth rate

        Returns:
            Implied terminal growth rate and full model results
        """
        def objective_function(terminal_growth):
            # Create a constant terminal growth distribution
            growth_dist = constant_value(terminal_growth)

            # Set the terminal growth rate in the DCF model
            self.dcf_model.set_terminal_growth(growth_dist)

            # Run the simulation
            self.dcf_model.run_simulation(self.base_revenue, self.base_year)

            # Calculate the median per-share value
            median_value = np.median(self.dcf_model.per_share_values)

            # Return the difference from the current price
            return median_value - self.current_price

        # Solve for the terminal growth rate using scipy's root-finding
        solution = optimize.root_scalar(
            objective_function,
            bracket=growth_bounds,
            method='brentq'
        )

        if not solution.converged:
            raise ValueError(SOLVER_CONVERGENCE_ERROR)

        # Get the solution
        implied_growth = solution.root

        # Run the model with the implied terminal growth rate
        self.dcf_model.set_terminal_growth(constant_value(implied_growth))
        results = self.dcf_model.run_simulation(
            self.base_revenue, self.base_year)

        return implied_growth, results

    def solve_multi_period_growth(
        self,
        period_splits: List[int],
        initial_guesses: List[float] = None,
        bounds: List[Tuple[float, float]] = None
    ) -> Tuple[List[float], Dict]:
        """
        Solve for implied multi-period revenue growth rates

        Args:
            period_splits: List of period indices where growth rates change
            initial_guesses: Initial guesses for growth factors in each period range
            bounds: Bounds for the growth factors in each period range

        Returns:
            List of implied growth rates and full model results
        """
        n_periods = self.dcf_model.n_periods
        n_splits = len(period_splits)

        # Ensure period_splits is in ascending order
        period_splits = sorted(period_splits)

        # Add the final period if not included
        if period_splits[-1] < n_periods:
            period_splits.append(n_periods)
            n_splits += 1

        # Set default initial guesses and bounds if not provided
        if initial_guesses is None:
            initial_guesses = [1.05] * n_splits

        if bounds is None:
            bounds = [(0.98, 1.2)] * n_splits

        def objective_function(growth_factors):
            # Create a list of growth distributions for each period range
            growth_dists = []

            start_idx = 0
            for i, end_idx in enumerate(period_splits):
                # Create a constant growth distribution for this period range
                growth_dist = constant_value(growth_factors[i])

                # Add the distribution for the number of periods in this range
                for _ in range(start_idx, end_idx):
                    growth_dists.append(growth_dist)

                start_idx = end_idx

            # Set the growth rates in the DCF model
            self.dcf_model.set_revenue_growth(growth_dists)

            # Run the simulation
            self.dcf_model.run_simulation(self.base_revenue, self.base_year)

            # Calculate the median per-share value
            median_value = np.median(self.dcf_model.per_share_values)

            # Return the difference from the current price
            return median_value - self.current_price

        # Solve for the growth rates using scipy's minimize
        result = optimize.minimize(
            lambda x: objective_function(x) ** 2,  # Minimize squared error
            x0=initial_guesses,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if not result.success:
            raise ValueError(SOLVER_CONVERGENCE_ERROR)

        # Get the solution
        implied_growth_rates = result.x

        # Run the model with the implied growth rates
        growth_dists = []
        start_idx = 0
        for i, end_idx in enumerate(period_splits):
            growth_dist = constant_value(implied_growth_rates[i])
            for _ in range(start_idx, end_idx):
                growth_dists.append(growth_dist)
            start_idx = end_idx

        self.dcf_model.set_revenue_growth(growth_dists)
        results = self.dcf_model.run_simulation(
            self.base_revenue, self.base_year)

        return implied_growth_rates.tolist(), results


class MarketImpliedAssumptions:
    """
    Extract and analyze market-implied assumptions from current stock prices

    This class combines the ReverseDCFSolver with additional analysis and visualization
    capabilities to extract and interpret market-implied assumptions.
    """

    def __init__(
        self,
        ticker: str,
        base_revenue: float,
        current_price: float,
        shares_outstanding: float,
        total_debt: float,
        total_cash: float,
        n_periods: int = 10,
        n_paths: int = 5000
    ):
        """
        Initialize the market-implied assumptions analyzer

        Args:
            ticker: Stock ticker symbol
            base_revenue: Current annual revenue
            current_price: Current stock price
            shares_outstanding: Number of shares outstanding
            total_debt: Total debt
            total_cash: Total cash and cash equivalents
            n_periods: Number of periods for DCF model
            n_paths: Number of simulation paths
        """
        self.ticker = ticker
        self.base_revenue = base_revenue
        self.current_price = current_price
        self.shares_outstanding = shares_outstanding
        self.total_debt = total_debt
        self.total_cash = total_cash

        # Create a DCF model
        self.dcf_model = MonteCarloDCF(n_periods=n_periods, n_paths=n_paths)

        # Set financial information
        self.dcf_model.set_debt(total_debt)
        self.dcf_model.set_cash(total_cash)
        self.dcf_model.set_shares_outstanding(shares_outstanding)

        # Create a solver
        self.solver = ReverseDCFSolver(
            self.dcf_model,
            base_revenue=base_revenue,
            current_price=current_price
        )

    def set_base_assumptions(
        self,
        ebit_margin: float = 0.15,
        tax_rate: float = 0.25,
        capex_percent: float = 0.07,
        nwc_percent: float = 0.10,
        da_percent: float = 0.05,
        wacc: float = 0.09,
        terminal_growth: float = 0.02,
        terminal_multiple: float = 12.0
    ) -> None:
        """
        Set base assumptions for the DCF model

        Args:
            ebit_margin: EBIT margin as percentage of revenue
            tax_rate: Tax rate
            capex_percent: Capital expenditure as percentage of revenue
            nwc_percent: Net working capital change as percentage of revenue change
            da_percent: Depreciation & amortization as percentage of revenue
            wacc: Weighted average cost of capital
            terminal_growth: Terminal growth rate
            terminal_multiple: Terminal multiple (EV/EBITDA or EV/EBIT)
        """
        self.dcf_model.set_ebit_margin(constant_value(ebit_margin))
        self.dcf_model.set_tax_rate(constant_value(tax_rate))
        self.dcf_model.set_capex_percent(constant_value(capex_percent))
        self.dcf_model.set_nwc_percent(constant_value(nwc_percent))
        self.dcf_model.set_da_percent(constant_value(da_percent))
        self.dcf_model.set_wacc(constant_value(wacc))
        self.dcf_model.set_terminal_growth(constant_value(terminal_growth))
        self.dcf_model.set_terminal_multiple(constant_value(terminal_multiple))

    def implied_growth_rate(self) -> Tuple[float, Dict]:
        """
        Calculate the implied constant growth rate

        Returns:
            Implied growth rate and full model results
        """
        return self.solver.solve_for_growth()

    def implied_margin(self) -> Tuple[float, Dict]:
        """
        Calculate the implied EBIT margin

        Returns:
            Implied margin and full model results
        """
        return self.solver.solve_for_margin()

    def implied_wacc(self) -> Tuple[float, Dict]:
        """
        Calculate the implied WACC

        Returns:
            Implied WACC and full model results
        """
        return self.solver.solve_for_wacc()

    def implied_wacc_distribution(
        self,
        price_range: Tuple[float, float] = None,
        price_std: float = None
    ) -> Tuple[DistributionSpec, Dict]:
        """
        Calculate the implied WACC distribution that matches observed price variance

        Args:
            price_range: Min and max stock price range (95% confidence interval)
            price_std: Standard deviation of stock prices

        Returns:
            WACC distribution specification and model results

        Note:
            Either price_range or price_std must be provided.
        """
        return self.solver.solve_for_wacc_distribution(
            price_range=price_range,
            price_std=price_std
        )

    def implied_terminal_growth(self) -> Tuple[float, Dict]:
        """
        Calculate the implied terminal growth rate

        Returns:
            Implied terminal growth rate and full model results
        """
        return self.solver.solve_for_terminal_growth()

    def implied_multi_period_growth(
        self,
        period_splits: List[int]
    ) -> Tuple[List[float], Dict]:
        """
        Calculate implied growth rates for multiple periods

        Args:
            period_splits: List of period indices where growth rates change

        Returns:
            List of implied growth rates and full model results
        """
        return self.solver.solve_multi_period_growth(period_splits)

    def calculate_all_implied_assumptions(self) -> Dict[str, float]:
        """
        Calculate all key implied assumptions

        Returns:
            Dictionary of implied assumption values
        """
        # Save original assumptions
        orig_growth = self.dcf_model.drivers.get('revenue_growth', None)
        orig_margin = self.dcf_model.drivers.get('ebit_margin', None)
        orig_wacc = self.dcf_model.drivers.get('wacc', None)
        orig_term_growth = self.dcf_model.drivers.get('terminal_growth', None)

        # Calculate implied growth with other parameters fixed
        implied_growth, _ = self.implied_growth_rate()

        # Restore original assumptions and calculate implied margin
        if orig_growth:
            self.dcf_model.set_revenue_growth(orig_growth)
        implied_margin, _ = self.implied_margin()

        # Restore original assumptions and calculate implied WACC
        if orig_growth:
            self.dcf_model.set_revenue_growth(orig_growth)
        if orig_margin:
            self.dcf_model.set_ebit_margin(orig_margin)
        implied_wacc, _ = self.implied_wacc()

        # Restore original assumptions and calculate implied terminal growth
        if orig_growth:
            self.dcf_model.set_revenue_growth(orig_growth)
        if orig_margin:
            self.dcf_model.set_ebit_margin(orig_margin)
        if orig_wacc:
            self.dcf_model.set_wacc(orig_wacc)
        implied_term_growth, _ = self.implied_terminal_growth()

        # Calculate implied multi-period growth (short-term vs long-term)
        if orig_growth:
            self.dcf_model.set_revenue_growth(orig_growth)
        if orig_margin:
            self.dcf_model.set_ebit_margin(orig_margin)
        if orig_wacc:
            self.dcf_model.set_wacc(orig_wacc)
        if orig_term_growth:
            self.dcf_model.set_terminal_growth(orig_term_growth)

        # Split at year 5 (short-term vs long-term)
        implied_multi_growth, _ = self.implied_multi_period_growth([5])

        return {
            'growth_rate': implied_growth - 1.0,  # Convert factor to rate
            'ebit_margin': implied_margin,
            'wacc': implied_wacc,
            'terminal_growth': implied_term_growth,
            # Convert factor to rate
            'short_term_growth': implied_multi_growth[0] - 1.0,
            # Convert factor to rate
            'long_term_growth': implied_multi_growth[1] - 1.0,
        }

    def plot_implied_vs_actual(
        self,
        actual_growth_history: List[float] = None,
        actual_margin_history: List[float] = None,
        analyst_growth_forecast: float = None,
        analyst_margin_forecast: float = None,
        years_history: int = 5,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot implied assumptions against historical actuals and analyst forecasts

        Args:
            actual_growth_history: List of historical growth rates
            actual_margin_history: List of historical margins
            analyst_growth_forecast: Analyst consensus growth forecast
            analyst_margin_forecast: Analyst consensus margin forecast
            years_history: Number of years of history to show
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Calculate implied assumptions
        implied = self.calculate_all_implied_assumptions()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot growth rates
        x_hist = list(range(-years_history, 0))
        x_future = list(range(1, self.dcf_model.n_periods + 1))

        # Plot historical growth
        if actual_growth_history:
            hist_data = actual_growth_history[-years_history:]
            ax1.bar(
                x_hist,
                hist_data,
                alpha=0.7,
                color='gray',
                label='Historical')

        # Plot implied growth (constant)
        ax1.axhline(implied['growth_rate'], color='red', linestyle='-',
                    label=f'Implied (Constant): {implied["growth_rate"]:.1%}')

        # Plot implied multi-period growth
        ax1.plot(x_future[:5], [implied['short_term_growth']] * 5, 'b--',
                 label=f'Implied (ST): {implied["short_term_growth"]:.1%}')
        ax1.plot(x_future[5:],
                 [implied['long_term_growth']] * (len(x_future) - 5),
                 'g--',
                 label=f'Implied (LT): {implied["long_term_growth"]:.1%}')

        # Plot analyst forecast
        if analyst_growth_forecast is not None:
            ax1.axhline(analyst_growth_forecast, color='purple', linestyle=':',
                        label=f'Analyst: {analyst_growth_forecast:.1%}')

        ax1.set_title(f'{self.ticker}: Growth Rate Comparison')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Annual Growth Rate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(bottom=min(implied['long_term_growth'] - 0.05,
                                implied['growth_rate'] - 0.05,
                                min(actual_growth_history or [0]) - 0.05))

        # Plot margins
        if actual_margin_history:
            hist_data = actual_margin_history[-years_history:]
            ax2.bar(
                x_hist,
                hist_data,
                alpha=0.7,
                color='gray',
                label='Historical')

        # Plot implied margin
        ax2.axhline(implied['ebit_margin'], color='red', linestyle='-',
                    label=f'Implied: {implied["ebit_margin"]:.1%}')

        # Plot analyst forecast
        if analyst_margin_forecast is not None:
            ax2.axhline(analyst_margin_forecast, color='purple', linestyle=':',
                        label=f'Analyst: {analyst_margin_forecast:.1%}')

        ax2.set_title(f'{self.ticker}: EBIT Margin Comparison')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('EBIT Margin')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        return fig

    def generate_scenarios(self) -> Dict[str, Dict]:
        """
        Generate different valuation scenarios

        Returns:
            Dictionary of scenario results
        """
        # Save current assumptions
        orig_growth = self.dcf_model.drivers.get('revenue_growth', None)
        orig_margin = self.dcf_model.drivers.get('ebit_margin', None)
        orig_wacc = self.dcf_model.drivers.get('wacc', None)
        orig_term_growth = self.dcf_model.drivers.get('terminal_growth', None)

        # Calculate implied growth (base case)
        implied_growth, base_results = self.implied_growth_rate()

        # Restore original assumptions
        if orig_growth:
            self.dcf_model.set_revenue_growth(orig_growth)
        if orig_margin:
            self.dcf_model.set_ebit_margin(orig_margin)
        if orig_wacc:
            self.dcf_model.set_wacc(orig_wacc)
        if orig_term_growth:
            self.dcf_model.set_terminal_growth(orig_term_growth)

        # Bullish scenario: Higher growth and margins
        self.dcf_model.set_revenue_growth(constant_value(implied_growth * 1.2))
        self.dcf_model.set_ebit_margin(constant_value(
            min(self.dcf_model.drivers['ebit_margin'][0][0] * 1.2, 0.4)
        ))
        bull_results = self.dcf_model.run_simulation(self.base_revenue)

        # Restore original assumptions
        if orig_growth:
            self.dcf_model.set_revenue_growth(orig_growth)
        if orig_margin:
            self.dcf_model.set_ebit_margin(orig_margin)

        # Bearish scenario: Lower growth and margins
        self.dcf_model.set_revenue_growth(constant_value(
            max(implied_growth * 0.8, 1.01)  # Ensure at least 1% growth
        ))
        self.dcf_model.set_ebit_margin(constant_value(
            max(self.dcf_model.drivers['ebit_margin'][0][0] * 0.8, 0.05)
        ))
        bear_results = self.dcf_model.run_simulation(self.base_revenue)

        return {
            'base': base_results,
            'bull': bull_results,
            'bear': bear_results
        }

    def plot_scenarios(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot different valuation scenarios

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        scenarios = self.generate_scenarios()

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot distributions
        for name, color, results in [
            ('Bear Case', 'red', scenarios['bear']),
            ('Base Case', 'blue', scenarios['base']),
            ('Bull Case', 'green', scenarios['bull'])
        ]:
            values = results['per_share_values']
            median = np.median(values)

            # Plot KDE
            kde = stats.gaussian_kde(values)
            x = np.linspace(min(values), max(values), 1000)
            ax.plot(x, kde(x), color=color, label=f'{name} (${median:.2f})')

            # Add vertical line for median
            ax.axvline(median, color=color, linestyle='--', alpha=0.5)

        # Add current price line
        ax.axvline(self.current_price, color='black', linestyle='-',
                   label=f'Current: ${self.current_price:.2f}')

        # Calculate probability of each scenario
        base_prob = np.mean(
            scenarios['base']['per_share_values'] > self.current_price) * 100
        bull_prob = np.mean(
            scenarios['bull']['per_share_values'] > self.current_price) * 100
        bear_prob = np.mean(
            scenarios['bear']['per_share_values'] > self.current_price) * 100

        # Add annotations
        plt.annotate(f'P(Undervalued): {base_prob:.1f}%', xy=(0.05, 0.9),
                     xycoords=AXES_FRACTION, color='blue')
        plt.annotate(f'Bull Prob: {bull_prob:.1f}%', xy=(0.05, 0.85),
                     xycoords=AXES_FRACTION, color='green')
        plt.annotate(f'Bear Prob: {bear_prob:.1f}%', xy=(0.05, 0.8),
                     xycoords=AXES_FRACTION, color='red')

        ax.set_title(f'{self.ticker}: Valuation Scenario Analysis')
        ax.set_xlabel(VALUE_PER_SHARE_LABEL)
        ax.set_ylabel('Probability Density')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    def plot_implied_vs_forecast_distributions(
        self,
        forecast_model: MonteCarloDCF,
        parameters: List[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot comparison between market-implied and forecast distributions

        Args:
            forecast_model: Monte Carlo DCF model with forecast assumptions
            parameters: List of parameters to compare (defaults to all key parameters)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Create market calibration metrics object
        calibration = MarketCalibrationMetrics(self.dcf_model, forecast_model)

        # Use calibration's plotting functionality
        return calibration.plot_calibration_comparison(
            parameters=parameters,
            figsize=figsize
        )

    def _get_parameter_details(self, param_name, implied_assumptions):
        """Helper method to get parameter values and implied values"""
        orig_value = None
        implied_value = 0.0

        if param_name == 'growth_rate':
            orig_value = self.dcf_model.drivers.get('revenue_growth', None)
            implied_value = implied_assumptions.get('growth_rate', 0.05)
        elif param_name == 'wacc':
            orig_value = self.dcf_model.drivers.get('wacc', None)
            implied_value = implied_assumptions.get('wacc', 0.09)
        elif param_name == 'ebit_margin':
            orig_value = self.dcf_model.drivers.get('ebit_margin', None)
            implied_value = implied_assumptions.get('ebit_margin', 0.15)
        elif param_name == 'terminal_growth':
            orig_value = self.dcf_model.drivers.get('terminal_growth', None)
            implied_value = implied_assumptions.get('terminal_growth', 0.02)

        return orig_value, implied_value

    def _set_parameter_value(self, param_name, value):
        """Helper method to set parameter value in the model"""
        if param_name == 'growth_rate':
            # Growth rate needs to be a factor (1+g)
            self.dcf_model.set_revenue_growth(constant_value(1.0 + value))
        elif param_name == 'wacc':
            self.dcf_model.set_wacc(constant_value(value))
        elif param_name == 'ebit_margin':
            self.dcf_model.set_ebit_margin(constant_value(value))
        elif param_name == 'terminal_growth':
            self.dcf_model.set_terminal_growth(constant_value(value))

    def _restore_parameter_value(self, param_name, orig_value):
        """Helper method to restore original parameter value"""
        if orig_value is None:
            return

        if param_name == 'growth_rate':
            self.dcf_model.set_revenue_growth(orig_value)
        elif param_name == 'wacc':
            self.dcf_model.set_wacc(orig_value)
        elif param_name == 'ebit_margin':
            self.dcf_model.set_ebit_margin(orig_value)
        elif param_name == 'terminal_growth':
            self.dcf_model.set_terminal_growth(orig_value)

    def _format_parameter_plot(
            self,
            ax,
            param_name,
            test_values,
            prices,
            implied_value):
        """Helper method to format the parameter sensitivity plot"""
        # Plot sensitivity curve
        ax.plot(test_values, prices, 'b-')

        # Add current price line
        ax.axhline(self.current_price, color='k', linestyle='--',
                   label=f'Current Price: ${self.current_price:.2f}')

        # Add implied value line
        ax.axvline(implied_value, color='r', linestyle='--',
                   label=f'Implied Value: {implied_value:.2%}')

        # Calculate implied fair value at current price
        implied_price_idx = np.argmin(
            np.abs(np.array(prices) - self.current_price))
        implied_fair_value = test_values[implied_price_idx]

        # Add point where sensitivity curve crosses current price
        ax.plot(implied_fair_value, self.current_price, 'ro', markersize=8)

        # Add elasticity calculations
        mid_idx = len(test_values) // 2
        price_elasticity = (
            (prices[mid_idx + 5] - prices[mid_idx - 5]) / prices[mid_idx]) / (
            (test_values[mid_idx + 5] - test_values[mid_idx - 5]) /
            test_values[mid_idx])

        # Format x-axis as percentage for rates
        if param_name in [
            'growth_rate',
            'wacc',
            'ebit_margin',
                'terminal_growth']:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

        # Add parameter name as title with more readable format
        title = param_name.replace('_', ' ').title()
        ax.set_title(f'{title} Sensitivity')

        # Add elasticity annotation
        ax.text(0.05, 0.9, f'Elasticity: {price_elasticity:.2f}',
                transform=ax.transAxes)

        ax.set_xlabel(title)
        ax.set_ylabel(VALUE_PER_SHARE_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_parameter_sensitivities(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot sensitivity of price to different parameters

        This method shows how stock price varies across different parameter ranges,
        and highlights the current market-implied values.

        Args:
            parameter_ranges: Dictionary of parameter names and their ranges to test
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Number of parameters to analyze
        n_params = len(parameter_ranges)

        # Create subplot grid - aim for 2 columns if more than 2 parameters
        n_cols = min(n_params, 2)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Make axes iterable even with a single subplot
        if n_params == 1:
            axes = np.array([axes])

        # Flatten axes array for easy iteration
        axes = axes.flatten()

        # Calculate implied assumptions for reference
        implied_assumptions = self.calculate_all_implied_assumptions()

        # For each parameter, create a sensitivity plot
        param_idx = 0
        for param_name, param_range in parameter_ranges.items():
            ax = axes[param_idx]

            # Get parameter details
            orig_value, implied_value = self._get_parameter_details(
                param_name, implied_assumptions)

            # Skip unknown parameters
            if orig_value is None and abs(implied_value) < 1e-10 and param_name not in [
                    'growth_rate', 'wacc', 'ebit_margin', 'terminal_growth']:
                ax.text(0.5, 0.5, f"Unknown parameter: {param_name}",
                        ha='center', va='center', transform=ax.transAxes)
                param_idx += 1
                continue

            # Create a range of values to test
            test_values = np.linspace(param_range[0], param_range[1], 50)

            # Array to store resulting prices
            prices = []

            # For each test value, calculate the resulting price
            for value in test_values:
                # Set parameter value
                self._set_parameter_value(param_name, value)

                # Run simulation
                self.dcf_model.run_simulation(self.base_revenue)

                # Store median price
                prices.append(np.median(self.dcf_model.per_share_values))

            # Restore original parameter value
            self._restore_parameter_value(param_name, orig_value)

            # Format the plot
            self._format_parameter_plot(
                ax, param_name, test_values, prices, implied_value)

            param_idx += 1

        # Hide any unused subplots
        for i in range(param_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def _set_model_parameter(self, model_name, value, transform_func=None):
        """Helper method to set parameter value in the model"""
        def identity_transform(x):
            return x

        if transform_func is None:
            transform_func = identity_transform

        value = transform_func(value)

        if model_name == 'revenue_growth':
            self.dcf_model.set_revenue_growth(constant_value(value))
        elif model_name == 'wacc':
            self.dcf_model.set_wacc(constant_value(value))
        elif model_name == 'ebit_margin':
            self.dcf_model.set_ebit_margin(constant_value(value))
        elif model_name == 'terminal_growth':
            self.dcf_model.set_terminal_growth(constant_value(value))

    def _restore_model_parameter(self, model_name, orig_value):
        """Helper method to restore original parameter value"""
        if orig_value is None:
            return

        if model_name == 'revenue_growth':
            self.dcf_model.set_revenue_growth(orig_value)
        elif model_name == 'wacc':
            self.dcf_model.set_wacc(orig_value)
        elif model_name == 'ebit_margin':
            self.dcf_model.set_ebit_margin(orig_value)
        elif model_name == 'terminal_growth':
            self.dcf_model.set_terminal_growth(orig_value)

    def _get_parameter_mapping(self):
        """Get mapping of parameter names to model attributes"""
        return {
            # Convert rate to factor
            'growth_rate': ('revenue_growth', lambda x: 1.0 + x),
            'wacc': ('wacc', lambda x: x),
            'ebit_margin': ('ebit_margin', lambda x: x),
            'terminal_growth': ('terminal_growth', lambda x: x)
        }

    def plot_multivariate_sensitivity(
        self,
        param1: str,
        param2: str,
        param1_range: Tuple[float, float],
        param2_range: Tuple[float, float],
        n_points: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot sensitivity of price to two parameters simultaneously

        Creates a 2D heatmap showing how two parameters jointly affect valuation

        Args:
            param1: First parameter name
            param2: Second parameter name
            param1_range: Range of values for first parameter
            param2_range: Range of values for second parameter
            n_points: Number of points to sample in each dimension
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Create parameter grids
        p1_values = np.linspace(param1_range[0], param1_range[1], n_points)
        p2_values = np.linspace(param2_range[0], param2_range[1], n_points)
        P1, P2 = np.meshgrid(p1_values, p2_values)

        # Map parameters to model attributes
        param_mapping = self._get_parameter_mapping()

        # Check if parameters are valid
        if param1 not in param_mapping or param2 not in param_mapping:
            raise ValueError(
                "Invalid parameter name. Must be one of: growth_rate, wacc, ebit_margin, terminal_growth")

        # Get model parameter names and transform functions
        p1_model_name, p1_transform = param_mapping[param1]
        p2_model_name, p2_transform = param_mapping[param2]

        # Save original values
        orig_p1 = self.dcf_model.drivers.get(p1_model_name, None)
        orig_p2 = self.dcf_model.drivers.get(p2_model_name, None)

        # Initialize results matrix
        values = np.zeros_like(P1)

        try:
            # For each parameter combination, calculate price
            for i in range(n_points):
                for j in range(n_points):
                    p1 = p1_values[i]
                    p2 = p2_values[j]

                    # Set parameter values
                    self._set_model_parameter(p1_model_name, p1, p1_transform)
                    self._set_model_parameter(p2_model_name, p2, p2_transform)

                    # Run simulation
                    self.dcf_model.run_simulation(self.base_revenue)

                    # Store median price
                    values[j, i] = np.median(self.dcf_model.per_share_values)
        finally:
            # Restore original parameter values
            self._restore_model_parameter(p1_model_name, orig_p1)
            self._restore_model_parameter(p2_model_name, orig_p2)

        # Create the visualization
        return self._create_sensitivity_plot(
            values, figsize, param1, param2, P1, P2
        )

    def _create_sensitivity_plot(
            self,
            values,
            figsize,
            param1,
            param2,
            p1_grid,
            p2_grid):
        """Helper method to create the sensitivity plot"""
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.contourf(p1_grid, p2_grid, values, 20, cmap='viridis')

        # Add contour lines with labels
        contour = ax.contour(
            p1_grid,
            p2_grid,
            values,
            5,
            colors='white',
            alpha=0.5)
        plt.clabel(contour, inline=True, fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(VALUE_PER_SHARE_LABEL)

        # Add current price contour
        price_contour = ax.contour(
            p1_grid, p2_grid, values, levels=[
                self.current_price], colors='r', linewidths=2)
        price_label = f'${self.current_price:.2f}'
        plt.clabel(
            price_contour,
            inline=True,
            fontsize=10,
            fmt=price_label)

        # Add implied values as a point
        self._add_implied_point(ax, param1, param2)

        # Format axes
        self._format_sensitivity_axes(ax, param1, param2)

        return fig

    def _add_implied_point(self, ax, param1, param2):
        """Add implied values as a point on the plot"""
        # Calculate implied assumptions
        implied_assumptions = self.calculate_all_implied_assumptions()

        # Add implied values as a point
        implied_p1 = implied_assumptions.get(param1, None)
        implied_p2 = implied_assumptions.get(param2, None)

        if implied_p1 is not None and implied_p2 is not None:
            ax.plot(implied_p1, implied_p2, 'ro', markersize=10,
                    label=f'Market-Implied (${self.current_price:.2f})')
            ax.legend()

    def _format_sensitivity_axes(self, ax, param1, param2):
        """Format the axes for the sensitivity plot"""
        # Format axes as percentages for rates
        if param1 in ['growth_rate', 'wacc', 'ebit_margin', 'terminal_growth']:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        if param2 in ['growth_rate', 'wacc', 'ebit_margin', 'terminal_growth']:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

        # Set labels with more readable parameter names
        p1_label = param1.replace('_', ' ').title()
        p2_label = param2.replace('_', ' ').title()

        ax.set_xlabel(p1_label)
        ax.set_ylabel(p2_label)
        ax.set_title(f'{self.ticker}: {p1_label} vs. {p2_label} Sensitivity')


class MarketCalibrationMetrics:
    """
    Compare market-implied distributions with forecast model distributions

    This class provides metrics to evaluate how well a forecast model's
    assumptions align with market-implied assumptions, helping calibrate
    models to reflect market consensus.
    """

    def __init__(
        self,
        market_model: MonteCarloDCF,
        forecast_model: MonteCarloDCF
    ):
        """
        Initialize the market calibration metrics calculator

        Args:
            market_model: DCF model with market-implied parameters
            forecast_model: DCF model with forecasted parameters
        """
        self.market_model = market_model
        self.forecast_model = forecast_model

    def calculate_calibration_metrics(
        self,
        base_revenue: float,
        parameters: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate calibration metrics comparing market and forecast distributions

        Args:
            base_revenue: Base revenue for both models
            parameters: List of parameters to compare (defaults to all common parameters)

        Returns:
            Dictionary of calibration metrics for each parameter
        """
        # Import here to avoid circular imports
        from ..services.calibration import (
            calculate_distribution_comparison,
            calculate_normal_distribution_metrics,
        )

        # Get common parameters if not specified
        parameters = self._get_common_parameters(parameters)

        # Ensure simulations are run
        self._ensure_simulations_run(base_revenue)

        # Initialize metrics dictionary
        metrics = {}

        # Compare value distributions
        market_values = self.market_model.per_share_values
        forecast_values = self.forecast_model.per_share_values
        metrics['value_distribution'] = calculate_distribution_comparison(
            market_values, forecast_values)

        # Process each parameter
        for param in parameters:
            param_metrics = self._get_parameter_metrics(
                param, calculate_normal_distribution_metrics)
            if param_metrics:
                metrics[param] = param_metrics

        return metrics

    def _get_common_parameters(
            self, parameters: List[str] = None) -> List[str]:
        """Get common parameters between market and forecast models"""
        if parameters is not None:
            return parameters

        common_params = []
        for param in self.market_model.driver_generator.drivers:
            if param in self.forecast_model.driver_generator.drivers:
                common_params.append(param)

        return common_params

    def _ensure_simulations_run(self, base_revenue: float) -> None:
        """Ensure that both models have been simulated"""
        if not hasattr(
                self.market_model,
                'per_share_values') or len(
                self.market_model.per_share_values) == 0:
            self.market_model.run_simulation(base_revenue)

        if not hasattr(
                self.forecast_model,
                'per_share_values') or len(
                self.forecast_model.per_share_values) == 0:
            self.forecast_model.run_simulation(base_revenue)

    def _get_parameter_metrics(
            self, param: str, metric_calculator: Callable) -> Optional[Dict[str, float]]:
        """Calculate metrics for a specific parameter"""
        # Check if parameter has proper distribution specs
        market_dist_spec = self.market_model.driver_generator.drivers.get(
            param, {}).get('dist_spec')
        forecast_dist_spec = self.forecast_model.driver_generator.drivers.get(
            param, {}).get('dist_spec')

        if not isinstance(
                market_dist_spec,
                DistributionSpec) or not isinstance(
                forecast_dist_spec,
                DistributionSpec):
            return None

        # Only handle normal distributions for now
        if market_dist_spec.dist_type == 'normal' and forecast_dist_spec.dist_type == 'normal':
            market_mean = market_dist_spec.params.get('mean')
            market_std = market_dist_spec.params.get('std')
            forecast_mean = forecast_dist_spec.params.get('mean')
            forecast_std = forecast_dist_spec.params.get('std')

            return metric_calculator(
                market_mean,
                market_std,
                forecast_mean,
                forecast_std)
        else:
            # For non-normal distributions, provide basic metrics
            return {
                'market_type': market_dist_spec.dist_type,
                'forecast_type': forecast_dist_spec.dist_type,
                'comparison': 'Not available for non-normal distributions'
            }

    def _calculate_overlap(
            self,
            dist1: np.ndarray,
            dist2: np.ndarray,
            bins: int = 100) -> float:
        """
        Calculate the overlap coefficient between two empirical distributions

        Args:
            dist1: First distribution samples
            dist2: Second distribution samples
            bins: Number of bins for histogram

        Returns:
            Overlap coefficient (0-1)
        """
        # Calculate histograms with the same bins
        min_val = min(np.min(dist1), np.min(dist2))
        max_val = max(np.max(dist1), np.max(dist2))
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        hist1, _ = np.histogram(dist1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)

        # Normalize if not already normalized
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

        # Calculate overlap (sum of minimums)
        return np.sum(np.minimum(hist1, hist2))

    def _bhattacharyya_distance(
            self,
            mu1: float,
            sigma1: float,
            mu2: float,
            sigma2: float) -> float:
        """
        Calculate Bhattacharyya distance between two normal distributions

        Args:
            mu1: Mean of first distribution
            sigma1: Standard deviation of first distribution
            mu2: Mean of second distribution
            sigma2: Standard deviation of second distribution

        Returns:
            Bhattacharyya distance (higher means more different)
        """
        var1 = sigma1**2
        var2 = sigma2**2

        # Calculate the Bhattacharyya distance
        term1 = 0.25 * np.log(0.25 * (var1/var2 + var2/var1 + 2))
        term2 = 0.25 * ((mu1 - mu2)**2 / (var1 + var2))

        return term1 + term2

    def plot_calibration_comparison(
        self,
        parameters: List[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot comparison between market and forecast distributions

        Args:
            parameters: List of parameters to plot (defaults to key parameters)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        from dcf_lab.visualization.calibration.plots import create_comparison_figure

        # Default to key parameters if none specified
        if parameters is None:
            parameters = self._get_default_parameters()

        # Calculate metrics
        metrics = self.calculate_calibration_metrics(
            base_revenue=1.0, parameters=parameters)

        # Create figure with subplots
        fig, axes = create_comparison_figure(parameters, figsize)

        # Plot each parameter
        try:
            for i, param in enumerate(parameters):
                ax = axes[i]

                if param == 'value_distribution':
                    self._plot_value_distribution(ax, metrics[param])
                else:
                    self._plot_parameter_distribution(ax, param, metrics)

            plt.tight_layout()
            return fig
        except Exception as e:
            # Clean up resources in case of error
            plt.close(fig)
            raise e

    def _get_default_parameters(self) -> List[str]:
        """
        Get default parameters for calibration plots

        Returns:
            List of parameter names
        """
        parameters = ['value_distribution']
        for param in [
            'wacc',
            'revenue_growth',
            'ebit_margin',
                'terminal_growth']:
            if param in self.market_model.driver_generator.drivers and param in self.forecast_model.driver_generator.drivers:
                parameters.append(param)
        return parameters

    def _plot_value_distribution(
            self, ax: plt.Axes, metrics: Dict[str, Any]) -> None:
        """
        Plot value distribution comparison

        Args:
            ax: Matplotlib axes to plot on
            metrics: Metrics dictionary for value distribution
        """
        from dcf_lab.visualization.calibration.plots import plot_value_distribution

        market_values = self.market_model.per_share_values
        forecast_values = self.forecast_model.per_share_values

        plot_value_distribution(
            ax=ax,
            market_values=market_values,
            forecast_values=forecast_values,
            metrics=metrics,
            value_per_share_label=VALUE_PER_SHARE_LABEL
        )

    def _plot_parameter_distribution(
        self,
        ax: plt.Axes,
        param: str,
        metrics: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Plot parameter distribution comparison

        Args:
            ax: Matplotlib axes to plot on
            param: Parameter name
            metrics: Metrics dictionary
        """
        from dcf_lab.visualization.calibration.plots import plot_parameter_distribution

        # Skip parameters without metrics
        if param not in metrics:
            ax.text(0.5, 0.5, f'No metrics for {param}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(param)
            return

        # Get distribution specifications
        market_spec = self.market_model.driver_generator.drivers[param][
            'dist_spec']
        forecast_spec = self.forecast_model.driver_generator.drivers[param][
            'dist_spec']

        # Only handle normal distributions
        if market_spec.dist_type == 'normal' and forecast_spec.dist_type == 'normal':
            plot_parameter_distribution(
                ax=ax,
                market_spec=market_spec,
                forecast_spec=forecast_spec,
                param_name=param,
                metrics=metrics[param]
            )


class ImpliedParameterDistribution:
    """
    Extract implied distributions for multiple parameters simultaneously

    This class can solve for multiple parameter distributions at once,
    particularly useful for determining correlated parameter distributions
    implied by stock price movements.
    """

    def __init__(
        self,
        dcf_model: MonteCarloDCF,
        base_revenue: float,
        current_price: float,
        price_std: float = None,
        price_range: Tuple[float, float] = None
    ):
        """
        Initialize the implied parameter distribution solver

        Args:
            dcf_model: Monte Carlo DCF model
            base_revenue: Base revenue for the model
            current_price: Current stock price
            price_std: Standard deviation of observed prices
            price_range: Tuple of (min_price, max_price) to match with 95% confidence
        """
        self.dcf_model = dcf_model
        self.base_revenue = base_revenue
        self.current_price = current_price

        # Calculate price standard deviation if range is provided
        if price_range is not None and price_std is None:
            # 95% confidence interval is approximately mean ± 2*std
            self.price_std = (price_range[1] - price_range[0]) / 4
        else:
            self.price_std = price_std

    def solve_for_correlated_parameters(
        self,
        parameters: List[str],
        initial_means: Dict[str, float] = None,
        initial_stds: Dict[str, float] = None,
        initial_correlation: float = 0.0,
        bounds: Dict[str, Tuple[float, float]] = None
    ) -> Tuple[Dict[str, DistributionSpec], np.ndarray, Dict]:
        """
        Solve for multiple correlated parameter distributions simultaneously

        This function determines the means, standard deviations, and correlations
        of multiple parameters that would match the observed price mean and volatility.

        Args:
            parameters: List of parameter names to solve for (e.g., ['wacc', 'growth_rate'])
            initial_means: Initial guesses for parameter means
            initial_stds: Initial guesses for parameter standard deviations
            initial_correlation: Initial guess for parameter correlation
            bounds: Parameter bounds as {param_name: (lower, upper)}

        Returns:
            Dictionary of parameter distributions, correlation matrix, and model results
        """
        from dcf_lab.solvers.correlated_parameters import CorrelatedParameterSolver

        # Create and use the correlated parameter solver
        solver = CorrelatedParameterSolver(
            dcf_model=self.dcf_model,
            base_revenue=self.base_revenue,
            current_price=self.current_price,
            price_std=self.price_std
        )

        return solver.solve(
            parameters=parameters,
            initial_means=initial_means,
            initial_stds=initial_stds,
            initial_correlation=initial_correlation,
            bounds=bounds
        )


# Usage example
if __name__ == "__main__":
    # Create market-implied assumptions analyzer
    market_implied = MarketImpliedAssumptions(
        ticker='AAPL',
        base_revenue=400000,  # $400B
        current_price=150,
        shares_outstanding=16000,  # 16B shares
        total_debt=100000,  # $100B
        total_cash=50000,  # $50B
    )

    # Set base assumptions
    market_implied.set_base_assumptions(
        ebit_margin=0.25,
        tax_rate=0.15,
        capex_percent=0.05,
        nwc_percent=0.10,
        da_percent=0.04,
        wacc=0.09,
        terminal_growth=0.025,
        terminal_multiple=15.0
    )

    # Calculate implied growth rate
    implied_growth, _ = market_implied.implied_growth_rate()
    print(f"Implied growth rate: {(implied_growth - 1) * 100:.2f}%")

    # Calculate all implied assumptions
    all_implied = market_implied.calculate_all_implied_assumptions()
    for key, value in all_implied.items():
        # Format all values as percentages
        print(f"{key}: {value * 100:.2f}%")

    # Plot implied vs actual
    historical_growth = [0.03, 0.05, 0.08, 0.04, 0.06]
    historical_margins = [0.21, 0.22, 0.24, 0.23, 0.25]
    fig1 = market_implied.plot_implied_vs_actual(
        actual_growth_history=historical_growth,
        actual_margin_history=historical_margins,
        analyst_growth_forecast=0.07,
        analyst_margin_forecast=0.26
    )

    # Plot scenarios
    fig2 = market_implied.plot_scenarios()

    plt.show()
