"""
Monte Carlo DCF Engine

This module implements a vectorized Monte Carlo simulation engine for DCF valuation,
allowing for probabilistic distributions of key drivers rather than fixed point estimates.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class DistributionSpec:
    """Specification for a distribution to be used in Monte Carlo simulation"""
    dist_type: str  # 'normal', 'uniform', 'triangular', 'custom', 'constant'
    params: Dict[str, Any]  # Parameters specific to the distribution type
    bounds: Tuple[float, float] = None  # Optional bounds for truncation

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate n_samples from the distribution"""
        if self.dist_type == 'normal':
            samples = stats.norm.rvs(
                loc=self.params.get('mean', 0),
                scale=self.params.get('std', 1),
                size=n_samples
            )
        elif self.dist_type == 'uniform':
            samples = stats.uniform.rvs(
                loc=self.params.get('min', 0),
                scale=self.params.get('max', 1) - self.params.get('min', 0),
                size=n_samples
            )
        elif self.dist_type == 'triangular':
            samples = stats.triang.rvs(
                c=(self.params.get('mode', 0.5) - self.params.get('min', 0)) /
                (self.params.get('max', 1) - self.params.get('min', 0)),
                loc=self.params.get('min', 0),
                scale=self.params.get('max', 1) - self.params.get('min', 0),
                size=n_samples
            )
        elif self.dist_type == 'custom':
            if callable(self.params.get('sampler')):
                samples = self.params['sampler'](n_samples)
            else:
                raise ValueError(
                    "Custom distribution requires a 'sampler' function")
        elif self.dist_type == 'constant':
            samples = np.full(n_samples, self.params.get('value', 0))
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")

        # Apply bounds if specified
        if self.bounds is not None:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])

        return samples


class MonteCarloDrivers:
    """Generator for correlated Monte Carlo drivers"""

    def __init__(
        self,
        n_periods: int = 10,
        n_paths: int = 10000,
        correlation_matrix: Optional[np.ndarray] = None,
        driver_names: Optional[List[str]] = None
    ):
        self.n_periods = n_periods
        self.n_paths = n_paths
        self.correlation_matrix = correlation_matrix
        self.driver_names = driver_names or []
        self.drivers = {}

    def add_driver(
        self,
        name: str,
        dist_spec: Union[DistributionSpec, List[DistributionSpec]],
        is_cumulative: bool = False
    ) -> None:
        """
        Add a driver distribution to the simulation

        Args:
            name: Name of the driver
            dist_spec: Single distribution spec for all periods or list of specs per period
            is_cumulative: If True, values are cumulative products of growth rates (1+g)
        """
        if name in self.drivers:
            print(f"Warning: Overwriting existing driver '{name}'")

        if name not in self.driver_names:
            self.driver_names.append(name)

        self.drivers[name] = {
            'dist_spec': dist_spec,
            'is_cumulative': is_cumulative
        }

    def generate_paths(self) -> Dict[str, np.ndarray]:
        """
        Generate correlated Monte Carlo paths for all drivers

        Returns:
            Dictionary mapping driver names to arrays of shape (n_paths, n_periods)
        """
        n_drivers = len(self.driver_names)
        uncorrelated_samples = {}

        # Generate uncorrelated samples for each driver
        for name in self.driver_names:
            dist_spec = self.drivers[name]['dist_spec']
            is_cumulative = self.drivers[name]['is_cumulative']

            # Initialize array for this driver
            samples = np.zeros((self.n_paths, self.n_periods))

            # Handle single vs. per-period distributions
            if isinstance(dist_spec, list):
                for period in range(min(self.n_periods, len(dist_spec))):
                    samples[:, period] = dist_spec[period].sample(self.n_paths)

                # Fill any remaining periods with the last distribution
                if len(dist_spec) < self.n_periods:
                    for period in range(len(dist_spec), self.n_periods):
                        samples[:, period] = dist_spec[-1].sample(self.n_paths)
            else:
                for period in range(self.n_periods):
                    samples[:, period] = dist_spec.sample(self.n_paths)

            # Apply cumulative product if needed
            if is_cumulative:
                # Interpret values as growth factors (1+g)
                samples = np.cumprod(samples, axis=1)

            uncorrelated_samples[name] = samples

        # Apply correlations if specified
        if self.correlation_matrix is not None and n_drivers > 1:
            # Stack all samples into a single array
            all_samples = np.stack([uncorrelated_samples[name].reshape(-1)
                                   for name in self.driver_names], axis=1)

            # Check correlation matrix dimensions
            if self.correlation_matrix.shape != (n_drivers, n_drivers):
                raise ValueError(
                    f"Correlation matrix must have shape ({n_drivers}, {n_drivers})")

            # Apply Cholesky decomposition to get correlated samples
            chol = np.linalg.cholesky(self.correlation_matrix)
            correlated_samples = all_samples @ chol.T

            # Reshape and store the correlated samples
            for i, name in enumerate(self.driver_names):
                original_shape = uncorrelated_samples[name].shape
                uncorrelated_samples[name] = correlated_samples[:, i].reshape(
                    original_shape)

        return uncorrelated_samples


class MonteCarloDCF:
    """Monte Carlo DCF model with probabilistic driver distributions"""

    def __init__(
        self,
        n_periods: int = 10,
        n_paths: int = 10000,
        terminal_period: int = None
    ):
        self.n_periods = n_periods
        self.n_paths = n_paths
        self.terminal_period = terminal_period or n_periods

        # Initialize drivers
        self.driver_generator = MonteCarloDrivers(n_periods, n_paths)

        # Arrays to hold simulation results
        self.revenues = None
        self.ebit_margins = None
        self.ebit = None
        self.tax_rates = None
        self.capex_pcts = None
        self.nwc_pcts = None
        self.da_pcts = None
        self.fcff = None
        self.terminal_values = None
        self.enterprise_values = None
        self.equity_values = None
        self.share_counts = None
        self.per_share_values = None

    def set_revenue_growth(
        self,
        dist_spec: Union[DistributionSpec, List[DistributionSpec]]
    ) -> None:
        """
        Set revenue growth distribution(s)

        Args:
            dist_spec: Single distribution spec or list of specs for each period
        """
        self.driver_generator.add_driver(
            'revenue_growth', dist_spec, is_cumulative=True)

    def set_ebit_margin(
        self,
        dist_spec: Union[DistributionSpec, List[DistributionSpec]]
    ) -> None:
        """Set EBIT margin distribution(s)"""
        self.driver_generator.add_driver('ebit_margin', dist_spec)

    def set_tax_rate(
        self,
        dist_spec: Union[DistributionSpec, List[DistributionSpec]]
    ) -> None:
        """Set tax rate distribution(s)"""
        self.driver_generator.add_driver('tax_rate', dist_spec)

    def set_capex_percent(
        self,
        dist_spec: Union[DistributionSpec, List[DistributionSpec]]
    ) -> None:
        """Set capital expenditure as percentage of revenue"""
        self.driver_generator.add_driver('capex_pct', dist_spec)

    def set_nwc_percent(
        self,
        dist_spec: Union[DistributionSpec, List[DistributionSpec]]
    ) -> None:
        """Set net working capital change as percentage of revenue change"""
        self.driver_generator.add_driver('nwc_pct', dist_spec)

    def set_da_percent(
        self,
        dist_spec: Union[DistributionSpec, List[DistributionSpec]]
    ) -> None:
        """Set depreciation & amortization as percentage of revenue"""
        self.driver_generator.add_driver('da_pct', dist_spec)

    def set_wacc(
        self,
        dist_spec: DistributionSpec
    ) -> None:
        """Set weighted average cost of capital distribution"""
        self.driver_generator.add_driver('wacc', dist_spec)

    def set_terminal_growth(
        self,
        dist_spec: DistributionSpec
    ) -> None:
        """Set terminal growth rate distribution"""
        self.driver_generator.add_driver('terminal_growth', dist_spec)

    def set_terminal_multiple(
        self,
        dist_spec: DistributionSpec
    ) -> None:
        """Set terminal multiple (EV/EBITDA or EV/EBIT) distribution"""
        self.driver_generator.add_driver('terminal_multiple', dist_spec)

    def set_terminal_weights(
        self,
        perpetuity_weight: float = 0.5,
        multiple_weight: float = 0.5
    ) -> None:
        """Set weights for blending terminal value methods"""
        if abs(perpetuity_weight + multiple_weight - 1.0) > 1e-6:
            raise ValueError("Terminal weights must sum to 1.0")

        self.perpetuity_weight = perpetuity_weight
        self.multiple_weight = multiple_weight

    def set_debt(
        self,
        current_debt: float,
        dist_spec: Optional[DistributionSpec] = None
    ) -> None:
        """
        Set debt for enterprise to equity value conversion

        Args:
            current_debt: Current total debt
            dist_spec: Distribution for debt changes over time (optional)
        """
        self.current_debt = current_debt
        if dist_spec is not None:
            self.driver_generator.add_driver('debt_change', dist_spec)

    def set_cash(
        self,
        current_cash: float,
        dist_spec: Optional[DistributionSpec] = None
    ) -> None:
        """
        Set cash for enterprise to equity value conversion

        Args:
            current_cash: Current cash and equivalents
            dist_spec: Distribution for cash changes over time (optional)
        """
        self.current_cash = current_cash
        if dist_spec is not None:
            self.driver_generator.add_driver('cash_change', dist_spec)

    def set_shares_outstanding(
        self,
        current_shares: float,
        dist_spec: Optional[DistributionSpec] = None
    ) -> None:
        """
        Set shares outstanding for per-share value calculation

        Args:
            current_shares: Current shares outstanding
            dist_spec: Distribution for share count changes (optional)
        """
        self.current_shares = current_shares
        if dist_spec is not None:
            self.driver_generator.add_driver(
                'share_change', dist_spec, is_cumulative=True)

    def set_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        driver_names: List[str]
    ) -> None:
        """
        Set correlation matrix for drivers

        Args:
            correlation_matrix: Correlation matrix as numpy array
            driver_names: List of driver names corresponding to matrix rows/columns
        """
        self.driver_generator.correlation_matrix = correlation_matrix
        self.driver_generator.driver_names = driver_names

    def run_simulation(
        self,
        base_revenue: float,
        base_year: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo DCF simulation

        Args:
            base_revenue: Starting revenue value
            base_year: Base year for calculations (e.g., 0 for current year)

        Returns:
            Dictionary with simulation results
        """
        # Generate driver paths
        drivers = self.driver_generator.generate_paths()

        # Calculate revenue paths
        revenue_growth = drivers.get(
            'revenue_growth', np.zeros(
                (self.n_paths, self.n_periods)) + 1)
        self.revenues = np.zeros((self.n_paths, self.n_periods))
        self.revenues[:, 0] = base_revenue * revenue_growth[:, 0]
        for t in range(1, self.n_periods):
            self.revenues[:, t] = self.revenues[:, t-1] * revenue_growth[:, t]

        # Calculate EBIT
        self.ebit_margins = drivers.get(
            'ebit_margin', np.zeros(
                (self.n_paths, self.n_periods)) + 0.15)
        self.ebit = self.revenues * self.ebit_margins

        # Calculate depreciation & amortization
        self.da_pcts = drivers.get(
            'da_pct', np.zeros(
                (self.n_paths, self.n_periods)) + 0.05)
        da = self.revenues * self.da_pcts

        # Calculate taxes
        self.tax_rates = drivers.get(
            'tax_rate', np.zeros(
                (self.n_paths, self.n_periods)) + 0.25)
        self.ebit * self.tax_rates

        # Calculate capex
        self.capex_pcts = drivers.get(
            'capex_pct', np.zeros(
                (self.n_paths, self.n_periods)) + 0.07)
        capex = self.revenues * self.capex_pcts

        # Calculate net working capital change
        self.nwc_pcts = drivers.get(
            'nwc_pct', np.zeros(
                (self.n_paths, self.n_periods)) + 0.1)
        nwc_change = np.zeros((self.n_paths, self.n_periods))
        for t in range(self.n_periods):
            if t == 0:
                # For first period, use change from base_revenue
                revenue_change = self.revenues[:, 0] - base_revenue
            else:
                revenue_change = self.revenues[:, t] - self.revenues[:, t-1]
            nwc_change[:, t] = revenue_change * self.nwc_pcts[:, t]

        # Calculate FCFF
        self.fcff = self.ebit * (1 - self.tax_rates) + da - capex - nwc_change

        # Calculate terminal values
        terminal_period = min(self.terminal_period, self.n_periods - 1)

        # Terminal value using perpetuity method
        wacc = drivers.get('wacc', np.full((self.n_paths, 1), 0.09))[:, 0]
        terminal_growth = drivers.get('terminal_growth',
                                      np.full((self.n_paths, 1), 0.02))[:, 0]

        # Ensure growth < WACC
        valid_growth = np.minimum(terminal_growth, wacc - 0.01)

        perpetuity_tv = self.fcff[:, terminal_period] * \
            (1 + valid_growth) / (wacc - valid_growth)

        # Terminal value using multiple method
        terminal_multiple = drivers.get('terminal_multiple',
                                        np.full((self.n_paths, 1), 12.0))[:, 0]
        multiple_tv = self.ebit[:, terminal_period] * terminal_multiple

        # Blend terminal values
        perpetuity_weight = getattr(self, 'perpetuity_weight', 0.5)
        multiple_weight = getattr(self, 'multiple_weight', 0.5)
        self.terminal_values = perpetuity_tv * \
            perpetuity_weight + multiple_tv * multiple_weight

        # Calculate discount factors
        discount_factors = np.zeros((self.n_paths, self.n_periods))
        for t in range(self.n_periods):
            discount_factors[:, t] = 1 / (1 + wacc) ** (t + 1)

        # Calculate PV of terminal value
        terminal_value_pv = self.terminal_values * \
            discount_factors[:, terminal_period]

        # Calculate PV of FCFFs (excluding terminal period)
        fcff_pv = np.sum(self.fcff[:, :terminal_period] *
                         discount_factors[:, :terminal_period], axis=1)

        # Calculate enterprise value
        self.enterprise_values = fcff_pv + terminal_value_pv

        # Calculate equity value
        debt = getattr(self, 'current_debt', 0)
        cash = getattr(self, 'current_cash', 0)

        # Apply any debt/cash changes if specified
        if 'debt_change' in drivers:
            debt_changes = drivers['debt_change'].sum(axis=1)
            debt = debt + debt_changes

        if 'cash_change' in drivers:
            cash_changes = drivers['cash_change'].sum(axis=1)
            cash = cash + cash_changes

        self.equity_values = self.enterprise_values - debt + cash

        # Calculate per-share value
        shares = getattr(self, 'current_shares', 1.0)

        # Apply share count changes if specified
        if 'share_change' in drivers:
            self.share_counts = shares * drivers['share_change'][:, -1]
        else:
            self.share_counts = np.full(self.n_paths, shares)

        self.per_share_values = self.equity_values / self.share_counts

        # Return results
        return {
            'revenues': self.revenues,
            'ebit': self.ebit,
            'fcf': self.fcff,
            'terminal_values': self.terminal_values,
            'enterprise_values': self.enterprise_values,
            'equity_values': self.equity_values,
            'per_share_values': self.per_share_values
        }

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for the simulation results

        Returns:
            Dictionary with summary statistics for each output variable
        """
        if self.per_share_values is None:
            raise ValueError(
                "Run simulation before getting summary statistics")

        results = {}

        # Calculate statistics for per-share values
        per_share_stats = {
            'mean': float(np.mean(self.per_share_values)),
            'median': float(np.median(self.per_share_values)),
            'std': float(np.std(self.per_share_values)),
            'min': float(np.min(self.per_share_values)),
            'max': float(np.max(self.per_share_values)),
            'p10': float(np.percentile(self.per_share_values, 10)),
            'p25': float(np.percentile(self.per_share_values, 25)),
            'p75': float(np.percentile(self.per_share_values, 75)),
            'p90': float(np.percentile(self.per_share_values, 90)),
        }
        results['per_share_value'] = per_share_stats

        # Calculate statistics for enterprise values
        if self.enterprise_values is not None:
            ev_stats = {
                'mean': float(np.mean(self.enterprise_values)),
                'median': float(np.median(self.enterprise_values)),
                'std': float(np.std(self.enterprise_values)),
                'min': float(np.min(self.enterprise_values)),
                'max': float(np.max(self.enterprise_values)),
                'p10': float(np.percentile(self.enterprise_values, 10)),
                'p25': float(np.percentile(self.enterprise_values, 25)),
                'p75': float(np.percentile(self.enterprise_values, 75)),
                'p90': float(np.percentile(self.enterprise_values, 90)),
            }
            results['enterprise_value'] = ev_stats

        # Calculate statistics for equity values
        if self.equity_values is not None:
            equity_stats = {
                'mean': float(np.mean(self.equity_values)),
                'median': float(np.median(self.equity_values)),
                'std': float(np.std(self.equity_values)),
                'min': float(np.min(self.equity_values)),
                'max': float(np.max(self.equity_values)),
                'p10': float(np.percentile(self.equity_values, 10)),
                'p25': float(np.percentile(self.equity_values, 25)),
                'p75': float(np.percentile(self.equity_values, 75)),
                'p90': float(np.percentile(self.equity_values, 90)),
            }
            results['equity_value'] = equity_stats

        return results

    def plot_distribution(
        self,
        current_price: Optional[float] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot the distribution of per-share values

        Args:
            current_price: Current stock price to compare against
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.per_share_values is None:
            raise ValueError("Run simulation before plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(self.per_share_values, bins=50, alpha=0.7, density=True)

        # Add vertical lines for key statistics
        median = np.median(self.per_share_values)
        p10 = np.percentile(self.per_share_values, 10)
        p90 = np.percentile(self.per_share_values, 90)

        ax.axvline(
            median,
            color='r',
            linestyle='-',
            label=f'Median: ${median:.2f}')
        ax.axvline(
            p10,
            color='orange',
            linestyle='--',
            label=f'P10: ${p10:.2f}')
        ax.axvline(
            p90,
            color='green',
            linestyle='--',
            label=f'P90: ${p90:.2f}')

        # Add current price if provided
        if current_price is not None:
            ax.axvline(current_price, color='blue', linestyle='-',
                       label=f'Current: ${current_price:.2f}')

            # Calculate probability of undervaluation
            prob_undervalued = np.mean(
                self.per_share_values > current_price) * 100
            ax.text(0.05, 0.95, f'P(Undervalued): {prob_undervalued:.1f}%',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')

        ax.set_title('Monte Carlo DCF: Per-Share Value Distribution')
        ax.set_xlabel('Value per Share ($)')
        ax.set_ylabel('Probability Density')
        ax.legend()

        return fig

    def plot_scenarios(
        self,
        scenario_names: List[str],
        scenario_results: List[Dict[str, np.ndarray]],
        current_price: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot comparison of different scenarios

        Args:
            scenario_names: List of scenario names
            scenario_results: List of results from run_simulation for each scenario
            current_price: Current stock price to compare against
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare KDE for each scenario
        for name, result in zip(scenario_names, scenario_results):
            per_share_values = result['per_share_values']
            median = np.median(per_share_values)

            # Plot KDE
            kde = stats.gaussian_kde(per_share_values)
            x = np.linspace(min(per_share_values), max(per_share_values), 1000)
            ax.plot(x, kde(x), label=f'{name} (Median: ${median:.2f})')

            # Add vertical line for median
            ax.axvline(median, linestyle='--', alpha=0.3)

        # Add current price if provided
        if current_price is not None:
            ax.axvline(current_price, color='black', linestyle='-',
                       label=f'Current: ${current_price:.2f}')

        ax.set_title('Monte Carlo DCF: Scenario Comparison')
        ax.set_xlabel('Value per Share ($)')
        ax.set_ylabel('Probability Density')
        ax.legend()

        return fig

    def calculate_irr(
        self,
        current_price: float,
        periods: int = 5
    ) -> np.ndarray:
        """
        Calculate implied IRR for investment at current price

        Args:
            current_price: Current stock price
            periods: Number of periods to consider for IRR calculation

        Returns:
            Array of IRR values for each Monte Carlo path
        """
        if self.per_share_values is None:
            raise ValueError("Run simulation before calculating IRR")

        # Simple IRR calculation: (final_value/initial_value)^(1/periods) - 1
        irr = (self.per_share_values / current_price) ** (1 / periods) - 1

        return irr


# Helper functions for common distributions

def normal_distribution(mean: float,
                        std: float,
                        bounds: Optional[Tuple[float,
                                               float]] = None) -> DistributionSpec:
    """Create a normal distribution specification"""
    return DistributionSpec(
        dist_type='normal',
        params={'mean': mean, 'std': std},
        bounds=bounds
    )


def uniform_distribution(min_val: float, max_val: float) -> DistributionSpec:
    """Create a uniform distribution specification"""
    return DistributionSpec(
        dist_type='uniform',
        params={'min': min_val, 'max': max_val}
    )


def triangular_distribution(
        min_val: float,
        mode: float,
        max_val: float) -> DistributionSpec:
    """Create a triangular distribution specification"""
    return DistributionSpec(
        dist_type='triangular',
        params={'min': min_val, 'mode': mode, 'max': max_val}
    )


def constant_value(value: float) -> DistributionSpec:
    """Create a constant value 'distribution'"""
    return DistributionSpec(
        dist_type='constant',
        params={'value': value}
    )


def custom_distribution(
        sampler_func: Callable[[int], np.ndarray]) -> DistributionSpec:
    """Create a custom distribution with a sampling function"""
    return DistributionSpec(
        dist_type='custom',
        params={'sampler': sampler_func}
    )


# Usage example
if __name__ == "__main__":
    # Create Monte Carlo DCF model
    mc_dcf = MonteCarloDCF(n_periods=10, n_paths=5000)

    # Set distributions for key drivers
    mc_dcf.set_revenue_growth(
        normal_distribution(
            1.06, 0.02, bounds=(
                1.01, 1.15)))
    mc_dcf.set_ebit_margin(triangular_distribution(0.18, 0.22, 0.25))
    mc_dcf.set_tax_rate(constant_value(0.25))
    mc_dcf.set_capex_percent(uniform_distribution(0.05, 0.08))
    mc_dcf.set_nwc_percent(constant_value(0.10))
    mc_dcf.set_wacc(triangular_distribution(0.08, 0.09, 0.11))
    mc_dcf.set_terminal_growth(triangular_distribution(0.015, 0.02, 0.025))
    mc_dcf.set_terminal_multiple(normal_distribution(12.0, 1.0))

    # Set financial information
    mc_dcf.set_debt(10000)
    mc_dcf.set_cash(5000)
    mc_dcf.set_shares_outstanding(1000)

    # Run simulation
    results = mc_dcf.run_simulation(base_revenue=20000)

    # Get summary statistics
    summary_stats = mc_dcf.get_summary_statistics()
    print(summary_stats)

    # Plot distribution
    fig = mc_dcf.plot_distribution(current_price=150)
    plt.show()
