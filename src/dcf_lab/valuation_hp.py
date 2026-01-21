"""
HIGH-PRECISION VALUATION MODULE
===============================

Enhanced valuation calculations with pinpoint mathematical accuracy.
All financial calculations performed using decimal arithmetic for maximum precision.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .forecast_model import ForecastInputs

# Import our high-precision framework
from .precision_math import (
    FinancialMathHP,
    HighPrecisionMath,
    PrecisionConfig,
    high_precision_context,
    setup_global_precision,
)


class HighPrecisionValuation:
    """High-precision DCF valuation with decimal arithmetic"""

    def __init__(self, precision_digits: int = 50):
        """Initialize with specified precision"""
        self.precision = precision_digits
        PrecisionConfig.set_decimal_precision(precision_digits)
        setup_global_precision()

    @staticmethod
    def _to_decimal_dict(data: Dict[str, float]) -> Dict[str, Decimal]:
        """Convert dictionary of floats to decimals"""
        return {key: HighPrecisionMath.to_decimal(
            value) for key, value in data.items()}

    @staticmethod
    def _to_decimal_series(series: pd.Series) -> List[Decimal]:
        """Convert pandas series to list of decimals"""
        return [HighPrecisionMath.to_decimal(value) for value in series.values]

    def discounted_cash_flow_hp(
        self,
        cf_df: pd.DataFrame,
        is_df: pd.DataFrame,
        drivers: Dict[str, float],
        inputs: ForecastInputs,
        marketcap: Optional[float],
        starting: Dict[str, float],
    ) -> Dict[str, Decimal]:
        """
        Compute DCF valuation with maximum precision using decimal arithmetic.

        Returns all values as Decimal objects for pinpoint accuracy.
        """
        with high_precision_context(self.precision + 10):
            # Convert inputs to high-precision decimals
            drivers_hp = self._to_decimal_dict(drivers)
            starting_hp = self._to_decimal_dict(starting)

            # Extract cash flows with high precision
            fcff_values = self._to_decimal_series(cf_df["fcf"])
            last_fcff = fcff_values[-1]

            # Calculate EBITDA with precision
            ebit_values = self._to_decimal_series(is_df["ebit"])
            da_values = self._to_decimal_series(is_df["d_and_a"])
            last_ebitda = ebit_values[-1] + da_values[-1]

            # High-precision input extraction
            cod = (HighPrecisionMath.to_decimal(inputs.cost_of_debt)
                   if inputs.cost_of_debt is not None
                   else drivers_hp.get("cost_of_debt", Decimal('0.05')))

            tax = (HighPrecisionMath.to_decimal(inputs.tax_rate)
                   if inputs.tax_rate is not None
                   else drivers_hp.get("eff_tax", Decimal('0.25')))

            beta = (HighPrecisionMath.to_decimal(inputs.beta)
                    if inputs.beta is not None
                    else Decimal('1.0'))

            # High-precision cost of equity calculation
            rf_rate = HighPrecisionMath.to_decimal(inputs.rf_rate)
            mrp = HighPrecisionMath.to_decimal(inputs.market_risk_premium)
            coe = rf_rate + beta * mrp

            # High-precision market values
            equity_value = (
                HighPrecisionMath.to_decimal(marketcap) if marketcap is not None else max(
                    starting_hp.get(
                        "equity",
                        Decimal('0')),
                    Decimal('0')))

            debt_value = starting_hp.get("debt", Decimal('0'))

            # High-precision WACC calculation
            wacc_rate = FinancialMathHP.wacc(
                equity_value=equity_value,
                debt_value=debt_value,
                cost_of_equity=coe,
                cost_of_debt=cod,
                tax_rate=tax
            )

            # High-precision present value of cash flows
            pv_fcff = Decimal('0')
            for period, cf in enumerate(fcff_values):
                period_decimal = Decimal(period + 1)
                pv_cf = FinancialMathHP.present_value(
                    cf, wacc_rate, period_decimal)
                pv_fcff += pv_cf

            # High-precision terminal value calculation
            years_decimal = Decimal(inputs.years)

            if inputs.terminal_method == "gordon":
                terminal_growth = HighPrecisionMath.to_decimal(
                    inputs.terminal_growth)
                tv = FinancialMathHP.terminal_value_gordon_growth(
                    final_fcf=last_fcff,
                    growth_rate=terminal_growth,
                    discount_rate=wacc_rate
                )
            else:
                exit_multiple = HighPrecisionMath.to_decimal(
                    inputs.exit_multiple)
                tv = last_ebitda * exit_multiple

            # Present value of terminal value
            pv_tv = FinancialMathHP.present_value(tv, wacc_rate, years_decimal)

            # Enterprise value
            enterprise_value = pv_fcff + pv_tv

            # Net debt calculation
            total_debt = (HighPrecisionMath.to_decimal(inputs.total_debt)
                          if inputs.total_debt is not None
                          else starting_hp.get("debt", Decimal('0')))

            cash = (HighPrecisionMath.to_decimal(inputs.cash)
                    if inputs.cash is not None
                    else starting_hp.get("cash", Decimal('0')))

            net_debt = total_debt - cash

            # Other adjustments
            minority_interest = HighPrecisionMath.to_decimal(
                inputs.minority_interest)
            preferred_equity = HighPrecisionMath.to_decimal(
                inputs.preferred_equity)

            # Equity value
            equity_value_model = enterprise_value - \
                net_debt - minority_interest - preferred_equity

            # Per share value
            shares_outstanding = HighPrecisionMath.to_decimal(
                inputs.shares_outstanding)
            per_share = (equity_value_model / shares_outstanding
                         if shares_outstanding > 0
                         else None)

            return {
                "wacc": wacc_rate,
                "cost_of_equity": coe,
                "cost_of_debt": cod,
                "pv_fcf": pv_fcff,
                "pv_terminal": pv_tv,
                "terminal_value": tv,
                "enterprise_value": enterprise_value,
                "net_debt": net_debt,
                "equity_value": equity_value_model,
                "per_share": per_share,
                "last_fcf": last_fcff,
                "last_ebitda": last_ebitda,
            }

    def sensitivity_analysis_hp(
        self,
        cf_df: pd.DataFrame,
        is_df: pd.DataFrame,
        drivers: Dict[str, float],
        inputs: ForecastInputs,
        marketcap: Optional[float],
        starting: Dict[str, float],
        wacc_range: Tuple[float, float] = (0.06, 0.15),
        growth_range: Tuple[float, float] = (0.01, 0.05),
        steps: int = 10
    ) -> pd.DataFrame:
        """
        Perform high-precision sensitivity analysis on WACC and terminal growth.

        Returns DataFrame with enterprise values for different scenario combinations.
        """
        wacc_min, wacc_max = wacc_range
        growth_min, growth_max = growth_range

        # Create high-precision ranges
        wacc_step = Decimal(str((wacc_max - wacc_min) / (steps - 1)))
        growth_step = Decimal(str((growth_max - growth_min) / (steps - 1)))

        wacc_values = [
            Decimal(
                str(wacc_min)) +
            i *
            wacc_step for i in range(steps)]
        growth_values = [
            Decimal(
                str(growth_min)) +
            i *
            growth_step for i in range(steps)]

        results = []

        with high_precision_context(self.precision + 5):
            for wacc_rate in wacc_values:
                for growth_rate in growth_values:
                    # Create modified inputs
                    test_inputs = ForecastInputs(
                        years=inputs.years, terminal_method="gordon",
                        terminal_growth=float(growth_rate),
                        exit_multiple=inputs.exit_multiple,
                        rf_rate=inputs.rf_rate,
                        market_risk_premium=inputs.market_risk_premium,
                        beta=float(
                            (wacc_rate - Decimal(str(inputs.rf_rate))) /
                            Decimal(str(inputs.market_risk_premium))),
                        cost_of_debt=inputs.cost_of_debt,
                        tax_rate=inputs.tax_rate,
                        shares_outstanding=inputs.shares_outstanding,
                        cash=inputs.cash, total_debt=inputs.total_debt,
                        minority_interest=inputs.minority_interest,
                        preferred_equity=inputs.preferred_equity,)

                    try:
                        valuation = self.discounted_cash_flow_hp(
                            cf_df, is_df, drivers, test_inputs, marketcap, starting)

                        results.append({
                            'wacc': float(wacc_rate),
                            'terminal_growth': float(growth_rate),
                            'enterprise_value': float(valuation['enterprise_value']),
                            'equity_value': float(valuation['equity_value']),
                            'per_share': float(valuation['per_share']) if valuation['per_share'] else None
                        })
                    except Exception as e:
                        # Handle edge cases
                        results.append({
                            'wacc': float(wacc_rate),
                            'terminal_growth': float(growth_rate),
                            'enterprise_value': None,
                            'equity_value': None,
                            'per_share': None,
                            'error': str(e)
                        })

        return pd.DataFrame(results)

    def monte_carlo_valuation_hp(
        self,
        cf_df: pd.DataFrame,
        is_df: pd.DataFrame,
        drivers: Dict[str, float],
        inputs: ForecastInputs,
        marketcap: Optional[float],
        starting: Dict[str, float],
        parameter_distributions: Dict[str, Tuple[float, float]],
        num_simulations: int = 10000
    ) -> Dict[str, List[Decimal]]:
        """
        Perform Monte Carlo simulation with high-precision arithmetic.

        parameter_distributions: Dict with keys like 'wacc_std', 'growth_std'
        and values as (mean, std_dev) tuples
        """
        import random
        from decimal import Decimal

        results = {
            'enterprise_values': [],
            'equity_values': [],
            'per_share_values': [],
            'wacc_values': [],
            'growth_values': []
        }

        with high_precision_context(self.precision + 5):
            for _ in range(num_simulations):
                # Sample parameters from distributions
                wacc_mean, wacc_std = parameter_distributions.get(
                    'wacc', (0.10, 0.02))
                growth_mean, growth_std = parameter_distributions.get(
                    'growth', (0.025, 0.01))

                # Generate random values with high precision
                wacc_sample = max(
                    0.01, random.normalvariate(
                        wacc_mean, wacc_std))
                growth_sample = max(
                    0.0, min(
                        0.1, random.normalvariate(
                            growth_mean, growth_std)))

                wacc_decimal = Decimal(str(wacc_sample))
                growth_decimal = Decimal(str(growth_sample))

                # Calculate corresponding beta for this WACC
                rf = Decimal(str(inputs.rf_rate))
                mrp = Decimal(str(inputs.market_risk_premium))
                beta_sample = (wacc_decimal - rf) / \
                    mrp if mrp > 0 else Decimal('1.0')

                # Create test inputs
                test_inputs = ForecastInputs(
                    years=inputs.years,
                    terminal_method="gordon",
                    terminal_growth=float(growth_decimal),
                    exit_multiple=inputs.exit_multiple,
                    rf_rate=inputs.rf_rate,
                    market_risk_premium=inputs.market_risk_premium,
                    beta=float(beta_sample),
                    cost_of_debt=inputs.cost_of_debt,
                    tax_rate=inputs.tax_rate,
                    shares_outstanding=inputs.shares_outstanding,
                    cash=inputs.cash,
                    total_debt=inputs.total_debt,
                    minority_interest=inputs.minority_interest,
                    preferred_equity=inputs.preferred_equity,
                )

                try:
                    valuation = self.discounted_cash_flow_hp(
                        cf_df, is_df, drivers, test_inputs, marketcap, starting
                    )

                    results['enterprise_values'].append(
                        valuation['enterprise_value'])
                    results['equity_values'].append(valuation['equity_value'])
                    results['per_share_values'].append(valuation['per_share'])
                    results['wacc_values'].append(wacc_decimal)
                    results['growth_values'].append(growth_decimal)

                except Exception:
                    # Skip failed simulations
                    continue

        return results

    def internal_rate_of_return_hp(self, cash_flows: List[float]) -> Decimal:
        """Calculate IRR with maximum precision"""
        cf_decimals = [HighPrecisionMath.to_decimal(cf) for cf in cash_flows]
        return FinancialMathHP.internal_rate_of_return(cf_decimals)

    def net_present_value_hp(
            self,
            cash_flows: List[float],
            discount_rate: float) -> Decimal:
        """Calculate NPV with maximum precision"""
        cf_decimals = [HighPrecisionMath.to_decimal(cf) for cf in cash_flows]
        rate_decimal = HighPrecisionMath.to_decimal(discount_rate)
        return FinancialMathHP.net_present_value(cf_decimals, rate_decimal)

    def wacc_hp(
            self,
            equity_value: float,
            debt_value: float,
            cost_of_equity: float,
            cost_of_debt: float,
            tax_rate: float) -> Decimal:
        """Calculate WACC with maximum precision"""
        return FinancialMathHP.wacc(
            equity_value=HighPrecisionMath.to_decimal(equity_value),
            debt_value=HighPrecisionMath.to_decimal(debt_value),
            cost_of_equity=HighPrecisionMath.to_decimal(cost_of_equity),
            cost_of_debt=HighPrecisionMath.to_decimal(cost_of_debt),
            tax_rate=HighPrecisionMath.to_decimal(tax_rate)
        )

    def format_results(
            self, results: Dict[str, Decimal], decimal_places: int = 6) -> Dict[str, str]:
        """Format high-precision results for display"""
        formatted = {}
        for key, value in results.items():
            if value is None:
                formatted[key] = "N/A"
            elif isinstance(value, Decimal):
                # Round to specified decimal places for display
                rounded = round(value, decimal_places)
                formatted[key] = f"{rounded:,.{decimal_places}f}"
            else:
                formatted[key] = str(value)
        return formatted


# Utility functions for backward compatibility
def discounted_cash_flow_precision(
    cf_df: pd.DataFrame,
    is_df: pd.DataFrame,
    drivers: Dict[str, float],
    inputs: ForecastInputs,
    marketcap: Optional[float],
    starting: Dict[str, float],
    precision_digits: int = 50
) -> Dict[str, float]:
    """
    High-precision DCF calculation that returns float results for compatibility.

    This function provides pinpoint accuracy internally but converts results
    to floats for compatibility with existing code.
    """
    hp_valuation = HighPrecisionValuation(precision_digits)
    hp_results = hp_valuation.discounted_cash_flow_hp(
        cf_df, is_df, drivers, inputs, marketcap, starting
    )

    # Convert Decimal results back to floats for compatibility
    float_results = {}
    for key, value in hp_results.items():
        if value is None:
            float_results[key] = None
        elif isinstance(value, Decimal):
            float_results[key] = float(value)
        else:
            float_results[key] = value

    return float_results


if __name__ == "__main__":
    # Example usage of high-precision valuation
    print("🎯 High-Precision Valuation Framework")
    print("=====================================")

    # Initialize high-precision valuation
    hp_val = HighPrecisionValuation(precision_digits=50)

    # Test high-precision financial calculations
    print("\n=== High-Precision Financial Tests ===")

    # Test IRR calculation
    cash_flows = [-1000, 300, 400, 500, 200]
    irr = hp_val.internal_rate_of_return_hp(cash_flows)
    print(f"IRR (50 digits): {irr}")

    # Test NPV calculation
    npv = hp_val.net_present_value_hp(cash_flows, 0.10)
    print(f"NPV (50 digits): {npv}")

    # Test WACC calculation
    wacc = hp_val.wacc_hp(
        equity_value=1000000,
        debt_value=500000,
        cost_of_equity=0.12,
        cost_of_debt=0.06,
        tax_rate=0.25
    )
    print(f"WACC (50 digits): {wacc}")

    print("\n✅ High-precision valuation framework ready for deployment!")
