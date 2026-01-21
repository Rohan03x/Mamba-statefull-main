"""
Valuation Ensemble

This module implements a stacked ensemble that blends DCF, residual-income, and relative valuation
methods to produce more robust valuation estimates. It allows for dynamic weighting based on
market conditions and the quality of inputs for each method.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .dcf_mc import MonteCarloDCF
from .rel_mult_ml import RelativeValuationModel


@dataclass
class ValuationMethod:
    """Valuation method with metadata and results"""
    name: str
    description: str
    model: Any
    weight: float = 1.0
    results: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None


class ValuationEnsemble:
    """
    Ensemble of multiple valuation methods

    This class combines various valuation approaches (DCF, residual income, relative)
    and produces a weighted consensus valuation based on method quality and
    current market regime.
    """

    def __init__(self):
        """Initialize the valuation ensemble"""
        self.methods = {}
        self.regime_detector = None
        self.is_calibrated = False
        self.current_regime = None
        self.regime_weights = {}

    def add_method(
        self,
        name: str,
        description: str,
        model: Any,
        weight: float = 1.0
    ) -> None:
        """
        Add a valuation method to the ensemble

        Args:
            name: Name/identifier of the method
            description: Description of the method
            model: The valuation model
            weight: Base weight for the method
        """
        self.methods[name] = ValuationMethod(
            name=name,
            description=description,
            model=model,
            weight=weight
        )

        self.is_calibrated = False

    def set_regime_detector(self, detector: Callable) -> None:
        """
        Set a regime detection function

        Args:
            detector: Function that returns current market regime
        """
        self.regime_detector = detector

    def set_regime_weights(
            self, regime_weights: Dict[str, Dict[str, float]]) -> None:
        """
        Set weights for each method under different regimes

        Args:
            regime_weights: Dictionary mapping regimes to method weights
                Example: {'high_growth': {'dc': 1.5, 'relative': 0.8}, ...}
        """
        self.regime_weights = regime_weights

    def detect_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Detect the current market regime

        Args:
            market_data: Dictionary of market data

        Returns:
            Current market regime
        """
        if self.regime_detector is None:
            return 'normal'

        return self.regime_detector(market_data)

    def calibrate(
        self,
        historical_data: pd.DataFrame,
        target_values: pd.Series,
        regime_data: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Calibrate method weights based on historical performance

        Args:
            historical_data: DataFrame with historical inputs
            target_values: Series with actual realized values
            regime_data: Optional mapping from dates to regimes

        Returns:
            Dictionary of calibrated weights
        """
        # Initialize error tracking
        method_errors = {name: [] for name in self.methods}

        # Evaluate each method on historical data
        self._collect_method_errors(
            historical_data,
            target_values,
            regime_data,
            method_errors)

        # Calculate quality scores and regime-specific weights
        regime_specific_weights = self._calculate_method_weights(method_errors)

        # Normalize weights and update internal state
        self._normalize_weights(regime_specific_weights)

        self.is_calibrated = True

        return {name: method.weight for name, method in self.methods.items()}

    def _collect_method_errors(
            self,
            historical_data,
            target_values,
            regime_data,
            method_errors):
        """Collect prediction errors for each method across historical data"""
        for date in historical_data.index:
            inputs = historical_data.loc[date]
            true_value = target_values.loc[date]

            # Determine regime if provided
            regime = 'normal'
            if regime_data and date in regime_data:
                regime = regime_data[date]

            self._evaluate_methods_for_date(
                inputs, true_value, regime, method_errors)

    def _evaluate_methods_for_date(
            self,
            inputs,
            true_value,
            regime,
            method_errors):
        """Evaluate each method for a specific date"""
        for name, method in self.methods.items():
            pred_value = self._get_method_prediction(method, inputs)
            if pred_value is None:
                continue

            # Calculate error
            error = (pred_value - true_value) / true_value
            method_errors[name].append((error, regime))

    def _get_method_prediction(self, method, inputs):
        """Get prediction from a method model"""
        if hasattr(method.model, 'predict_equity_value'):
            return method.model.predict_equity_value(inputs)
        elif hasattr(method.model, 'value'):
            return method.model.value(inputs)
        else:
            # Skip methods that don't have a standard interface
            return None

    def _calculate_method_weights(self, method_errors):
        """Calculate quality scores and regime-specific weights"""
        regime_specific_weights = defaultdict(dict)

        for name, errors in method_errors.items():
            if not errors:
                continue

            # Calculate overall RMSE
            squared_errors = [e[0]**2 for e in errors]
            rmse = np.sqrt(np.mean(squared_errors))

            # Calculate regime-specific RMSE
            regime_errors = self._group_errors_by_regime(errors)
            regime_rmse = {
                r: np.sqrt(np.mean(errs)) for r, errs in regime_errors.items()
            }

            # Set weights based on RMSE
            self._set_method_weights(
                name, rmse, regime_rmse, regime_specific_weights)

        return regime_specific_weights

    def _group_errors_by_regime(self, errors):
        """Group squared errors by market regime"""
        regime_errors = defaultdict(list)
        for error, regime in errors:
            regime_errors[regime].append(error**2)
        return regime_errors

    def _set_method_weights(
            self,
            name,
            rmse,
            regime_rmse,
            regime_specific_weights):
        """Set method weights based on RMSE values"""
        # Calculate weights as inverse of RMSE
        overall_weight = 1 / (rmse + 1e-6)
        self.methods[name].weight = overall_weight
        self.methods[name].quality_score = 1 / (rmse + 1e-6)

        # Set regime-specific weights
        for regime, r_rmse in regime_rmse.items():
            regime_specific_weights[regime][name] = 1 / (r_rmse + 1e-6)

    def _normalize_weights(self, regime_specific_weights):
        """Normalize all weights to sum to 1.0"""
        # Normalize overall weights
        total_weight = sum(method.weight for method in self.methods.values())
        for name in self.methods:
            self.methods[name].weight /= total_weight

        # Normalize regime-specific weights
        for regime in regime_specific_weights:
            total = sum(regime_specific_weights[regime].values())
            for name in regime_specific_weights[regime]:
                regime_specific_weights[regime][name] /= total

        # Update regime weights
        self.regime_weights = dict(regime_specific_weights)

    def get_method_weights(self, regime: str) -> Dict[str, float]:
        """
        Get weights for each method given the current regime

        Args:
            regime: Current market regime

        Returns:
            Dictionary mapping method names to weights
        """
        # If we have regime-specific weights, use them
        if regime in self.regime_weights:
            weights = {}
            for name in self.methods:
                weights[name] = self.regime_weights[regime].get(
                    name, self.methods[name].weight)
        else:
            # Otherwise use base weights
            weights = {
                name: method.weight for name,
                method in self.methods.items()}

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight

        return weights

    def value(
        self,
        inputs: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Produce ensemble valuation

        Args:
            inputs: Inputs for valuation models
            market_data: Optional market data for regime detection
            regime: Optional explicit regime (overrides detection)

        Returns:
            Dictionary with valuation results
        """
        # Detect regime if not provided
        regime = self._determine_regime(regime, market_data)
        self.current_regime = regime

        # Get method weights for the current regime
        weights = self.get_method_weights(regime)

        # Run each valuation method
        method_results = self._run_valuation_methods(inputs, weights)

        # Extract and combine values from results
        value_collections = self._extract_valuation_components(method_results)

        # Build final ensemble results
        ensemble_results = self._calculate_ensemble_results(
            regime, weights, method_results, value_collections)

        return ensemble_results

    def _determine_regime(self, regime, market_data):
        """Determine market regime from data or default"""
        if regime is None and market_data is not None:
            return self.detect_regime(market_data)
        elif regime is None:
            return 'normal'
        return regime

    def _run_valuation_methods(self, inputs, weights):
        """Run each valuation method and collect results"""
        method_results = {}
        for name, method in self.methods.items():
            try:
                result = self._apply_valuation_method(method, inputs)
                if result is None:
                    continue

                # Store results and weight
                method_results[name] = {
                    'result': result,
                    'weight': weights.get(name, 0)
                }

                # Update method with results
                self.methods[name].results = result

            except Exception as e:
                print(f"Error applying method {name}: {e}")

        return method_results

    def _apply_valuation_method(self, method, inputs):
        """Apply a valuation method to inputs"""
        if hasattr(method.model, 'predict_equity_value'):
            return method.model.predict_equity_value(inputs)
        elif hasattr(method.model, 'value'):
            return method.model.value(inputs)
        else:
            # Skip methods that don't have a standard interface
            return None

    def _extract_valuation_components(self, method_results):
        """Extract valuation components from method results"""
        enterprise_values = []
        equity_values = []
        per_share_values = []

        for name, res in method_results.items():
            weight = res['weight']
            result = res['result']

            if isinstance(result, dict):
                self._extract_from_dict(
                    result,
                    weight,
                    enterprise_values,
                    equity_values,
                    per_share_values)
            elif hasattr(result, 'enterprise_value'):
                self._extract_from_object(
                    result,
                    weight,
                    enterprise_values,
                    equity_values,
                    per_share_values)

        return {
            'enterprise_values': enterprise_values,
            'equity_values': equity_values,
            'per_share_values': per_share_values
        }

    def _extract_from_dict(
            self,
            result,
            weight,
            enterprise_values,
            equity_values,
            per_share_values):
        """Extract values from dictionary result"""
        if 'enterprise_value' in result:
            enterprise_values.append((result['enterprise_value'], weight))

        if 'equity_value' in result:
            equity_values.append((result['equity_value'], weight))

        if 'per_share_value' in result:
            per_share_values.append((result['per_share_value'], weight))

    def _extract_from_object(
            self,
            result,
            weight,
            enterprise_values,
            equity_values,
            per_share_values):
        """Extract values from object attributes"""
        enterprise_values.append((result.enterprise_value, weight))
        equity_values.append((result.equity_value, weight))

        if hasattr(result, 'per_share_value'):
            per_share_values.append((result.per_share_value, weight))

    def _calculate_ensemble_results(
            self,
            regime,
            weights,
            method_results,
            value_collections):
        """Calculate final ensemble results"""
        ensemble_results = {
            'methods': method_results,
            'regime': regime,
            'weights': weights
        }

        # Add weighted average for enterprise value
        enterprise_values = value_collections['enterprise_values']
        if enterprise_values:
            ensemble_results['enterprise_value'] = self._weighted_average(
                enterprise_values)

        # Add weighted average for equity value
        equity_values = value_collections['equity_values']
        if equity_values:
            ensemble_results['equity_value'] = self._weighted_average(
                equity_values)

        # Add weighted average for per share value
        per_share_values = value_collections['per_share_values']
        if per_share_values:
            ensemble_results['per_share_value'] = self._weighted_average(
                per_share_values)

        return ensemble_results

    def _weighted_average(self, values):
        """Calculate weighted average from (value, weight) pairs"""
        total_weight = sum(w for _, w in values)
        return sum(val * w for val, w in values) / \
            total_weight if total_weight > 0 else 0

    def estimate_value_distribution(
        self,
        inputs: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        regime: Optional[str] = None,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Estimate full value distribution from ensemble

        Args:
            inputs: Inputs for valuation models
            market_data: Optional market data for regime detection
            regime: Optional explicit regime (overrides detection)
            n_samples: Number of samples to generate

        Returns:
            Dictionary with distribution results
        """
        # Get point estimate first
        point_results = self.value(inputs, market_data, regime)

        # Collect distributions from individual methods
        distributions = self._collect_method_distributions(inputs, n_samples)

        # Combine distributions if available
        if distributions:
            # Generate samples from the combined distribution
            value_samples = self._generate_ensemble_samples(
                distributions, n_samples)

            # Calculate distribution statistics
            distribution_results = self._calculate_distribution_statistics(
                value_samples)
        else:
            # Fall back to point estimates
            distribution_results = self._create_fallback_distribution(
                point_results, n_samples)

        return distribution_results

    def _collect_method_distributions(self, inputs, n_samples):
        """Collect distributions from individual methods"""
        distributions = {}

        for name, method in self.methods.items():
            if hasattr(method.model, 'estimate_distribution'):
                try:
                    dist = method.model.estimate_distribution(
                        inputs, n_samples)
                    distributions[name] = dist
                except Exception as e:
                    print(f"Error getting distribution for {name}: {e}")

        return distributions

    def _generate_ensemble_samples(self, distributions, n_samples):
        """Generate samples from ensemble of distributions"""
        weights = self.get_method_weights(self.current_regime)

        # Prepare output arrays
        ev_samples = np.zeros(n_samples)
        eq_samples = np.zeros(n_samples)
        ps_samples = np.zeros(n_samples)

        # Calculate total weight for methods with distributions
        total_weight = sum(weights.get(name, 0) for name in distributions)

        # Pre-compute method selection probabilities
        methods = list(distributions.keys())
        probs = [weights.get(name, 0) / total_weight for name in methods]

        # Generate samples
        for i in range(n_samples):
            sample = self._sample_from_distribution(
                methods, probs, distributions)
            if sample:
                self._extract_sample_values(
                    sample, i, ev_samples, eq_samples, ps_samples)

        return {
            'enterprise_value': ev_samples,
            'equity_value': eq_samples,
            'per_share_value': ps_samples
        }

    def _sample_from_distribution(self, methods, probs, distributions):
        """Sample a value from one of the distributions based on weights"""
        # Create a numpy random number generator instance with seed
        rng = np.random.default_rng(seed=42)

        # Use modern Generator API for random choice
        selected = rng.choice(methods, p=probs)

        # Sample from the selected distribution
        dist = distributions[selected]
        if hasattr(dist, 'sample'):
            return dist.sample()
        elif isinstance(dist, dict) and 'samples' in dist:
            # Use modern Generator API for random integer
            idx = rng.integers(len(dist['samples']))
            return dist['samples'][idx]
        else:
            return None

    def _extract_sample_values(
            self,
            sample,
            index,
            ev_samples,
            eq_samples,
            ps_samples):
        """Extract values from a sample and store in output arrays"""
        if isinstance(sample, dict):
            ev_samples[index] = sample.get('enterprise_value', 0)
            eq_samples[index] = sample.get('equity_value', 0)
            ps_samples[index] = sample.get('per_share_value', 0)
        else:
            ev_samples[index] = getattr(sample, 'enterprise_value', 0)
            eq_samples[index] = getattr(sample, 'equity_value', 0)
            ps_samples[index] = getattr(sample, 'per_share_value', 0)

    def _calculate_distribution_statistics(self, value_samples):
        """Calculate statistics for the sample distributions"""
        quantile_points = [0.05, 0.25, 0.5, 0.75, 0.95]
        result = {}

        for value_type, samples in value_samples.items():
            quantiles = np.quantile(samples, quantile_points)
            result[value_type] = {
                'samples': samples,
                'mean': np.mean(samples),
                'std': np.std(samples),
                'p05': quantiles[0],
                'p25': quantiles[1],
                'median': quantiles[2],
                'p75': quantiles[3],
                'p95': quantiles[4]
            }

        return result

    def _create_fallback_distribution(self, point_results, n_samples):
        """Create fallback distribution when no model distributions are available"""
        # Create fallback distributions using point estimates
        ev_value = point_results.get('enterprise_value', 0)
        eq_value = point_results.get('equity_value', 0)
        ps_value = point_results.get('per_share_value', 0)

        distribution_results = {
            'enterprise_value': {
                'samples': np.ones(n_samples) * ev_value,
                'mean': ev_value,
                'std': 0,
                'p05': ev_value,
                'p25': ev_value,
                'median': ev_value,
                'p75': ev_value,
                'p95': ev_value
            },
            'equity_value': {
                'samples': np.ones(n_samples) * eq_value,
                'mean': eq_value,
                'std': 0,
                'p05': eq_value,
                'p25': eq_value,
                'median': eq_value,
                'p75': eq_value,
                'p95': eq_value
            },
            'per_share_value': {
                'samples': np.ones(n_samples) * ps_value,
                'mean': ps_value,
                'std': 0,
                'p05': ps_value,
                'p25': ps_value,
                'median': ps_value,
                'p75': ps_value,
                'p95': ps_value
            }
        }

        # Add distribution to point results
        point_results['distribution'] = distribution_results

        return point_results

    def plot_ensemble_distribution(
        self,
        results: Dict[str, Any],
        current_price: Optional[float] = None,
        title: str = 'Ensemble Valuation Distribution',
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot the ensemble valuation distribution

        Args:
            results: Results from estimate_value_distribution
            current_price: Optional current market price
            title: Plot title
            figsize: Figure size
        """
        if 'distribution' not in results:
            raise ValueError("No distribution data in results")

        dist = results['distribution']

        plt.figure(figsize=figsize)

        # Plot per-share value distribution
        if 'per_share_value' in dist:
            ps_samples = dist['per_share_value']['samples']

            # Plot histogram
            plt.hist(
                ps_samples,
                bins=30,
                alpha=0.5,
                color='blue',
                density=True)

            # Plot kernel density estimate
            x = np.linspace(np.min(ps_samples), np.max(ps_samples), 1000)
            kde = stats.gaussian_kde(ps_samples)
            plt.plot(x, kde(x), 'b-', linewidth=2)

            # Add quantile lines
            for q, label in zip([0.05, 0.5, 0.95], ['P5', 'Median', 'P95']):
                val = np.percentile(ps_samples, q * 100)
                plt.axvline(val, color='red', linestyle='--', alpha=0.7)
                plt.text(
                    val,
                    plt.gca().get_ylim()[1] * 0.9,
                    f' {label}: {
                        val:.2f}',
                    horizontalalignment='left',
                    verticalalignment='top')

            # Add current price if provided
            if current_price:
                plt.axvline(
                    current_price,
                    color='green',
                    linestyle='-',
                    linewidth=2)
                plt.text(
                    current_price,
                    plt.gca().get_ylim()[1] * 0.8,
                    f' Current: {
                        current_price:.2f}',
                    horizontalalignment='left',
                    verticalalignment='top',
                    color='green')

                # Calculate probability of being undervalued
                prob_undervalued = np.mean(ps_samples > current_price)
                plt.text(
                    plt.gca().get_xlim()[1] * 0.95,
                    plt.gca().get_ylim()[1] * 0.95,
                    f'P(Undervalued): {
                        prob_undervalued:.1%}',
                    horizontalalignment='right',
                    verticalalignment='top',
                    bbox={
                        'facecolor': 'white',
                        'alpha': 0.7})

        plt.title(title)
        plt.xlabel('Per-Share Value')
        plt.ylabel('Probability Density')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_method_comparison(
        self,
        results: Dict[str, Any],
        current_price: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot comparison of different valuation methods

        Args:
            results: Results from value() or estimate_value_distribution()
            current_price: Optional current market price
            figsize: Figure size
        """
        if 'methods' not in results:
            raise ValueError("No method results in results")

        method_results = results['methods']
        weights = results['weights']

        # Extract per-share values
        methods = []
        values = []
        method_weights = []

        for name, res in method_results.items():
            method_val = None
            if isinstance(
                    res['result'],
                    dict) and 'per_share_value' in res['result']:
                method_val = res['result']['per_share_value']
            elif hasattr(res['result'], 'per_share_value'):
                method_val = res['result'].per_share_value

            if method_val is not None:
                methods.append(name)
                values.append(method_val)
                method_weights.append(weights.get(name, 0))

        # Sort by weight
        sorted_idx = np.argsort(method_weights)[::-1]
        methods = [methods[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        method_weights = [method_weights[i] for i in sorted_idx]

        # Create plot
        plt.figure(figsize=figsize)

        # Plot ensemble value
        ensemble_val = results.get('per_share_value', None)
        if ensemble_val is not None:
            plt.axvline(ensemble_val, color='blue', linestyle='-', linewidth=2,
                        label=f'Ensemble: {ensemble_val:.2f}')

        # Plot method values
        plt.barh(methods, values, alpha=0.6)

        # Add weight labels
        for i, (val, weight) in enumerate(zip(values, method_weights)):
            plt.text(val, i, f' {val:.2f} ({weight:.1%})', va='center')

        # Add current price if provided
        if current_price:
            plt.axvline(
                current_price,
                color='green',
                linestyle='-',
                linewidth=2,
                label=f'Current: {
                    current_price:.2f}')

        plt.title('Valuation Method Comparison')
        plt.xlabel('Per-Share Value')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


class ResidualIncomeModel:
    """
    Residual Income Valuation Model

    This class implements the Residual Income Valuation method, which values
    a company based on its book value plus the present value of expected
    residual income (earnings in excess of required return on book value).
    """

    def __init__(
        self,
        book_value: float,
        cost_of_equity: float,
        shares_outstanding: Optional[float] = None
    ):
        """
        Initialize the residual income model

        Args:
            book_value: Current book value of equity
            cost_of_equity: Cost of equity capital
            shares_outstanding: Number of shares outstanding
        """
        self.book_value = book_value
        self.cost_of_equity = cost_of_equity
        self.shares_outstanding = shares_outstanding
        self.ri_forecasts = []
        self.terminal_growth = 0.02
        self.is_fitted = False

    def set_residual_income_forecasts(self, ri_forecasts: List[float]) -> None:
        """
        Set forecasted residual income values

        Args:
            ri_forecasts: List of forecasted residual income values
        """
        self.ri_forecasts = ri_forecasts
        self.is_fitted = True

    def calculate_residual_income(
        self,
        net_income: float,
        prior_book_value: float
    ) -> float:
        """
        Calculate residual income

        Args:
            net_income: Net income for the period
            prior_book_value: Book value at the start of the period

        Returns:
            Residual income
        """
        required_return = prior_book_value * self.cost_of_equity
        return net_income - required_return

    def set_forecasts_from_earnings(
        self,
        earnings_forecasts: List[float],
        roe_forecasts: List[float]
    ) -> None:
        """
        Set forecasts from earnings and ROE projections

        Args:
            earnings_forecasts: List of forecasted earnings
            roe_forecasts: List of forecasted return on equity
        """
        if len(earnings_forecasts) != len(roe_forecasts):
            raise ValueError(
                "Earnings and ROE forecasts must have the same length")

        # Initialize book value
        book_values = [self.book_value]

        # Calculate projected book values
        for i in range(len(earnings_forecasts)):
            # Projected book value = previous book value + earnings * retention rate
            # Assuming retention rate = 1 - (earnings / (roe * book value))
            if roe_forecasts[i] > 0:
                retention_rate = 1 - \
                    earnings_forecasts[i] / \
                    (roe_forecasts[i] * book_values[-1])
                retention_rate = max(0, min(1, retention_rate))
            else:
                retention_rate = 0.5  # Default if ROE is not positive

            new_book_value = book_values[-1] + \
                earnings_forecasts[i] * retention_rate
            book_values.append(new_book_value)

        # Calculate residual income
        ri_forecasts = []
        for i in range(len(earnings_forecasts)):
            ri = self.calculate_residual_income(
                earnings_forecasts[i], book_values[i])
            ri_forecasts.append(ri)

        self.set_residual_income_forecasts(ri_forecasts)

    def value(self) -> Dict[str, float]:
        """
        Calculate intrinsic value using residual income model

        Returns:
            Dictionary with valuation results
        """
        if not self.is_fitted:
            raise ValueError(
                "Model is not fitted yet (no residual income forecasts)")

        # Present value of explicit forecast period
        pv_ri = 0
        for i, ri in enumerate(self.ri_forecasts):
            pv_ri += ri / ((1 + self.cost_of_equity) ** (i + 1))

        # Terminal value calculation
        if self.ri_forecasts:
            terminal_ri = self.ri_forecasts[-1] * (1 + self.terminal_growth)
            terminal_value = terminal_ri / \
                (self.cost_of_equity - self.terminal_growth)
            pv_terminal = terminal_value / \
                ((1 + self.cost_of_equity) ** len(self.ri_forecasts))
        else:
            pv_terminal = 0

        # Total equity value
        equity_value = self.book_value + pv_ri + pv_terminal

        # Per-share value
        per_share_value = None
        if self.shares_outstanding and self.shares_outstanding > 0:
            per_share_value = equity_value / self.shares_outstanding

        return {
            'equity_value': equity_value,
            'per_share_value': per_share_value,
            'book_value': self.book_value,
            'pv_residual_income': pv_ri,
            'pv_terminal_value': pv_terminal
        }

    def estimate_distribution(
        self,
        ri_std_devs: Optional[List[float]] = None,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Estimate value distribution using Monte Carlo

        Args:
            ri_std_devs: Standard deviations for residual income forecasts
            n_samples: Number of samples

        Returns:
            Dictionary with distribution samples and statistics
        """
        if not self.is_fitted:
            raise ValueError(
                "Model is not fitted yet (no residual income forecasts)")

        # Default standard deviations if not provided (increasing with time)
        if ri_std_devs is None:
            ri_std_devs = [abs(ri) * 0.2 * (1.2 ** i)
                           for i, ri in enumerate(self.ri_forecasts)]

        # Generate samples
        equity_samples = np.zeros(n_samples)
        rng = np.random.default_rng(seed=42)

        for i in range(n_samples):
            # Sample residual income values
            ri_samples = [
                rng.normal(ri, std)
                for ri, std in zip(self.ri_forecasts, ri_std_devs)
            ]

            # Calculate present value
            pv_ri = 0
            for j, ri in enumerate(ri_samples):
                pv_ri += ri / ((1 + self.cost_of_equity) ** (j + 1))

            # Terminal value calculation
            terminal_ri = ri_samples[-1] * (1 + self.terminal_growth)
            terminal_value = terminal_ri / \
                (self.cost_of_equity - self.terminal_growth)
            pv_terminal = terminal_value / \
                ((1 + self.cost_of_equity) ** len(ri_samples))

            # Total equity value
            equity_samples[i] = self.book_value + pv_ri + pv_terminal

        # Calculate distribution statistics
        quantiles = np.quantile(equity_samples, [0.05, 0.25, 0.5, 0.75, 0.95])

        # Per-share values if available
        per_share_samples = None
        ps_quantiles = None

        if self.shares_outstanding and self.shares_outstanding > 0:
            per_share_samples = equity_samples / self.shares_outstanding
            ps_quantiles = np.quantile(
                per_share_samples, [
                    0.05, 0.25, 0.5, 0.75, 0.95])

        return {
            'equity_samples': equity_samples,
            'equity_mean': np.mean(equity_samples),
            'equity_std': np.std(equity_samples),
            'equity_quantiles': {
                'p05': quantiles[0],
                'p25': quantiles[1],
                'median': quantiles[2],
                'p75': quantiles[3],
                'p95': quantiles[4]},
            'per_share_samples': per_share_samples,
            'per_share_mean': np.mean(per_share_samples) if per_share_samples is not None else None,
            'per_share_std': np.std(per_share_samples) if per_share_samples is not None else None,
            'per_share_quantiles': {
                'p05': ps_quantiles[0],
                'p25': ps_quantiles[1],
                'median': ps_quantiles[2],
                'p75': ps_quantiles[3],
                'p95': ps_quantiles[4]} if ps_quantiles is not None else None}


def create_valuation_ensemble(
    dcf_model: MonteCarloDCF,
    relative_model: RelativeValuationModel,
    residual_income_model: ResidualIncomeModel,
    base_weights: Optional[Dict[str, float]] = None
) -> ValuationEnsemble:
    """
    Create a valuation ensemble with standard components

    Args:
        dcf_model: Monte Carlo DCF model
        relative_model: Relative valuation model
        residual_income_model: Residual income model
        base_weights: Optional base weights for each method

    Returns:
        Configured ValuationEnsemble
    """
    ensemble = ValuationEnsemble()

    # Set base weights
    if base_weights is None:
        base_weights = {
            'dc': 2.0,
            'relative': 1.5,
            'residual_income': 1.0
        }

    # Add valuation methods
    ensemble.add_method(
        name='dc',
        description='Monte Carlo Discounted Cash Flow',
        model=dcf_model,
        weight=base_weights.get('dc', 2.0)
    )

    ensemble.add_method(
        name='relative',
        description='Relative Valuation',
        model=relative_model,
        weight=base_weights.get('relative', 1.5)
    )

    ensemble.add_method(
        name='residual_income',
        description='Residual Income Valuation',
        model=residual_income_model,
        weight=base_weights.get('residual_income', 1.0)
    )

    return ensemble


def simple_regime_detector(market_data: Dict[str, Any]) -> str:
    """
    Simple regime detection based on market conditions

    Args:
        market_data: Dictionary with market indicators

    Returns:
        Detected regime ('growth', 'value', 'defensive', 'normal')
    """
    # Extract indicators
    vix = market_data.get('vix', 20)
    rate_10y = market_data.get('treasury_10y', 0.03)
    rate_change = market_data.get('treasury_10y_1m_change', 0)
    mkt_return_1y = market_data.get('market_return_1y', 0.08)

    # Detect high volatility regime
    if vix > 30:
        return 'defensive'

    # Detect rising rate environment
    if rate_10y > 0.04 and rate_change > 0.005:
        return 'value'

    # Detect growth regime
    if mkt_return_1y > 0.15 and rate_10y < 0.035:
        return 'growth'

    # Default regime
    return 'normal'
