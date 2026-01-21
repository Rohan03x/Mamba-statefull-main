"""
Guardrails & Fallback Logic for AI Price Forecasting

This module implements robust guardrails to detect and handle edge cases:
- Noisy/monotone forecasts detection
- Insufficient data handling
- Model failure detection
- Automatic fallback to simpler models
- Quality assurance checks

Key Features:
- Forecast quality validators
- Model health monitoring
- Automatic fallback mechanisms
- Simple baseline models
- Error recovery strategies
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastQualityValidator:
    """
    Validates forecast quality and detects problematic forecasts
    """

    def __init__(self):
        self.quality_thresholds = {
            'min_forecast_variance': 1e-6,
            'max_forecast_variance': 1.0,
            'min_forecast_range': 1e-4,
            'max_forecast_range': 0.5,
            'max_monotone_streak': 7,
            'min_reasonable_returns': -0.2,
            'max_reasonable_returns': 0.2,
            'max_volatility_ratio': 10.0,
            'min_quantile_spread': 1e-4
        }

    def validate_forecast_quality(self,
                                  forecast_returns: List[float],
                                  forecast_series: pd.Series = None,
                                  quantile_forecasts: Dict[str,
                                                           Any] = None) -> Dict[str,
                                                                                Any]:
        """
        Comprehensive forecast quality validation

        Args:
            forecast_returns: Predicted log returns
            forecast_series: Predicted price series
            quantile_forecasts: Quantile prediction results

        Returns:
            Dictionary with validation results and quality scores
        """
        validation_results = {
            'is_valid': True,
            'quality_score': 1.0,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }

        try:
            # Validate forecast returns
            if forecast_returns:
                return_validation = self._validate_returns(forecast_returns)
                validation_results['metrics'].update(
                    return_validation['metrics'])
                validation_results['issues'].extend(
                    return_validation['issues'])
                validation_results['warnings'].extend(
                    return_validation['warnings'])

                if not return_validation['is_valid']:
                    validation_results['is_valid'] = False

            # Validate price series
            if forecast_series is not None and not forecast_series.empty:
                series_validation = self._validate_series(forecast_series)
                validation_results['metrics'].update(
                    series_validation['metrics'])
                validation_results['issues'].extend(
                    series_validation['issues'])
                validation_results['warnings'].extend(
                    series_validation['warnings'])

                if not series_validation['is_valid']:
                    validation_results['is_valid'] = False

            # Validate quantile forecasts
            if quantile_forecasts:
                quantile_validation = self._validate_quantiles(
                    quantile_forecasts)
                validation_results['metrics'].update(
                    quantile_validation['metrics'])
                validation_results['issues'].extend(
                    quantile_validation['issues'])
                validation_results['warnings'].extend(
                    quantile_validation['warnings'])

                if not quantile_validation['is_valid']:
                    validation_results['is_valid'] = False

            # Calculate overall quality score
            validation_results['quality_score'] = self._calculate_quality_score(
                validation_results)

        except Exception as e:
            logger.error(f"Forecast validation failed: {e}")
            validation_results['is_valid'] = False
            validation_results['quality_score'] = 0.0
            validation_results['issues'].append(f"Validation error: {str(e)}")

        return validation_results

    def _validate_returns(self, returns: List[float]) -> Dict[str, Any]:
        """Validate forecast returns for reasonableness"""
        result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metrics': {}}

        if not returns or len(returns) == 0:
            result['is_valid'] = False
            result['issues'].append("Empty forecast returns")
            return result

        returns_array = np.array(returns)

        # Check for NaN or infinite values
        if np.any(np.isnan(returns_array)) or np.any(np.isinf(returns_array)):
            result['is_valid'] = False
            result['issues'].append(
                "NaN or infinite values in forecast returns")

        # Check return magnitude
        if np.any(returns_array < self.quality_thresholds
                  ['min_reasonable_returns']):
            result['warnings'].append("Extremely negative returns detected")

        if np.any(returns_array > self.quality_thresholds
                  ['max_reasonable_returns']):
            result['warnings'].append("Extremely positive returns detected")

        # Check variance
        returns_var = np.var(returns_array)
        result['metrics']['returns_variance'] = returns_var

        if returns_var < self.quality_thresholds['min_forecast_variance']:
            result['issues'].append(
                "Forecast variance too low (potentially flat/constant)")
            result['is_valid'] = False
        elif returns_var > self.quality_thresholds['max_forecast_variance']:
            result['warnings'].append(
                "Forecast variance very high (potentially noisy)")

        # Check for monotone behavior
        monotone_streak = self._check_monotone_streak(returns_array)
        result['metrics']['max_monotone_streak'] = monotone_streak

        if monotone_streak > self.quality_thresholds['max_monotone_streak']:
            result['issues'].append(
                f"Monotone streak detected ({monotone_streak} periods)")
            result['is_valid'] = False

        return result

    def _validate_series(self, series: pd.Series) -> Dict[str, Any]:
        """Validate forecast price series"""
        result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metrics': {}}

        if series.empty:
            result['is_valid'] = False
            result['issues'].append("Empty forecast series")
            return result

        # Check for NaN values
        if series.isna().any():
            result['is_valid'] = False
            result['issues'].append("NaN values in forecast series")

        # Check for negative prices
        if (series <= 0).any():
            result['is_valid'] = False
            result['issues'].append("Non-positive prices in forecast")

        # Check forecast range
        price_range = (series.max() - series.min()) / series.iloc[0]
        result['metrics']['relative_price_range'] = price_range

        if price_range < self.quality_thresholds['min_forecast_range']:
            result['issues'].append(
                "Forecast range too small (potentially flat)")
            result['is_valid'] = False
        elif price_range > self.quality_thresholds['max_forecast_range']:
            result['warnings'].append(
                "Forecast range very large (potentially volatile)")

        return result

    def _validate_quantiles(
            self, quantile_forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quantile forecast consistency"""
        result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metrics': {}}

        # Extract quantile values for first horizon
        quantiles = self._extract_quantile_values(quantile_forecasts)

        if not quantiles:
            result['warnings'].append("No valid quantile forecasts found")
            return result

        # Validate quantile ordering and spread
        self._check_quantile_ordering(quantiles, result)
        self._check_quantile_spread(quantiles, result)

        return result

    def _extract_quantile_values(
            self, quantile_forecasts: Dict[str, Any]) -> Dict[float, float]:
        """Extract quantile values from forecast data"""
        quantiles = {}
        for key, values in quantile_forecasts.items():
            if key.startswith('q_') and values:
                try:
                    q_level = int(key[2:]) / 100.0
                    if isinstance(values[0], list) and values[0]:
                        # First horizon, first value
                        quantiles[q_level] = values[0][0]
                    elif isinstance(values[0], (int, float)):
                        quantiles[q_level] = values[0]
                except (ValueError, IndexError):
                    continue
        return quantiles

    def _check_quantile_ordering(
            self, quantiles: Dict[float, float],
            result: Dict[str, Any]) -> None:
        """Check if quantiles are properly ordered"""
        sorted_levels = sorted(quantiles.keys())
        sorted_values = [quantiles[level] for level in sorted_levels]

        if not all(sorted_values[i] <= sorted_values[i+1]
                   for i in range(len(sorted_values)-1)):
            result['issues'].append("Quantile forecasts not properly ordered")
            result['is_valid'] = False

    def _check_quantile_spread(
            self, quantiles: Dict[float, float],
            result: Dict[str, Any]) -> None:
        """Check if quantile spread is reasonable"""
        sorted_levels = sorted(quantiles.keys())
        sorted_values = [quantiles[level] for level in sorted_levels]

        if len(sorted_values) >= 2:
            quantile_spread = sorted_values[-1] - sorted_values[0]
            result['metrics']['quantile_spread'] = quantile_spread

            if quantile_spread < self.quality_thresholds['min_quantile_spread']:
                result['issues'].append(
                    "Quantile spread too small (uncertain estimates)")
                result['is_valid'] = False

    def _check_monotone_streak(self, values: np.ndarray) -> int:
        """Check for maximum monotone streak in forecast"""
        if len(values) < 2:
            return 0

        max_streak = 1
        current_streak = 1

        for i in range(1, len(values)):
            if values[i] >= values[i-1]:  # Non-decreasing
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        # Also check for non-increasing streaks
        current_streak = 1
        for i in range(1, len(values)):
            if values[i] <= values[i-1]:  # Non-increasing
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        return max_streak

    def _calculate_quality_score(
            self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)"""
        if not validation_results['is_valid']:
            return 0.0

        score = 1.0

        # Penalize warnings
        warning_penalty = len(validation_results['warnings']) * 0.1
        score -= min(warning_penalty, 0.3)  # Cap at 30% penalty

        # Bonus for good metrics
        metrics = validation_results['metrics']

        # Reward appropriate variance
        if 'returns_variance' in metrics:
            var = metrics['returns_variance']
            if 0.0001 <= var <= 0.01:  # Good variance range
                score += 0.1

        # Reward low monotone streaks
        if 'max_monotone_streak' in metrics:
            streak = metrics['max_monotone_streak']
            if streak <= 3:
                score += 0.1

        return max(0.0, min(1.0, score))


class SimpleBaselineModels:
    """
    Simple baseline models for fallback scenarios
    """

    @staticmethod
    def random_walk_forecast(last_price: float,
                             horizon: int,
                             volatility: float = 0.02) -> Tuple[pd.Series,
                                                                Dict]:
        """
        Simple random walk forecast

        Args:
            last_price: Last observed price
            horizon: Forecast horizon in days
            volatility: Daily volatility estimate

        Returns:
            Tuple of (forecast series, results dict)
        """
        rng = np.random.default_rng(42)

        # Generate random walk
        returns = rng.normal(0, volatility, horizon)
        prices = [last_price]

        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        # Create forecast series
        forecast_dates = pd.date_range(
            start=datetime.now(), periods=horizon, freq='D'
        )
        forecast_series = pd.Series(prices[1:], index=forecast_dates)

        results = {
            'model_type': 'random_walk_baseline',
            'forecast_returns': returns.tolist(),
            'last_price': last_price,
            'volatility_used': volatility,
            'is_fallback': True
        }

        return forecast_series, results

    @staticmethod
    def moving_average_forecast(
            prices: pd.Series, horizon: int) -> Tuple[pd.Series, Dict]:
        """
        Simple moving average forecast

        Args:
            prices: Historical price series
            horizon: Forecast horizon in days

        Returns:
            Tuple of (forecast series, results dict)
        """
        if len(prices) < 5:
            # Not enough data, use last price
            last_price = prices.iloc[-1] if not prices.empty else 100.0
            forecast_values = [last_price] * horizon
        else:
            # Use simple moving average
            ma_period = min(len(prices), 20)
            ma_value = prices.tail(ma_period).mean()

            # Add small random walk component
            rng = np.random.default_rng(42)
            volatility = prices.pct_change().std() if len(prices) > 1 else 0.02
            returns = rng.normal(
                0, volatility * 0.5, horizon)  # Reduced volatility

            forecast_values = [ma_value]
            for ret in returns:
                forecast_values.append(forecast_values[-1] * (1 + ret))
            forecast_values = forecast_values[1:]

        # Create forecast series
        forecast_dates = pd.date_range(
            start=datetime.now(), periods=horizon, freq='D'
        )
        forecast_series = pd.Series(forecast_values, index=forecast_dates)

        results = {
            'model_type': 'moving_average_baseline',
            'forecast_returns': [0.0] * horizon,  # MA is trend-following
            'last_price': prices.iloc[-1] if not prices.empty else 100.0,
            'is_fallback': True
        }

        return forecast_series, results


class ModelHealthMonitor:
    """
    Monitors model health and detects failures
    """

    def __init__(self):
        self.failure_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.health_history = []

    def record_model_attempt(self, model_name: str, success: bool,
                             error_message: str = None) -> None:
        """
        Record model execution attempt

        Args:
            model_name: Name of the model
            success: Whether the attempt was successful
            error_message: Error message if failed
        """
        if success:
            self.success_counts[model_name] += 1
        else:
            self.failure_counts[model_name] += 1

        self.health_history.append({
            'timestamp': datetime.now(),
            'model_name': model_name,
            'success': success,
            'error_message': error_message
        })

        # Keep only recent history
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]

    def get_model_health_score(self, model_name: str) -> float:
        """
        Get health score for a specific model (0-1)

        Args:
            model_name: Name of the model

        Returns:
            Health score between 0 and 1
        """
        total_attempts = self.success_counts[model_name] + \
            self.failure_counts[model_name]

        if total_attempts == 0:
            return 1.0  # No history, assume healthy

        success_rate = self.success_counts[model_name] / total_attempts

        # Recent performance weight
        recent_history = [h for h in self.health_history[-20:]
                          if h['model_name'] == model_name]
        if recent_history:
            recent_success_rate = sum(
                1 for h in recent_history if h['success']) / len(recent_history)
            # Weight recent performance more heavily
            health_score = 0.3 * success_rate + 0.7 * recent_success_rate
        else:
            health_score = success_rate

        return health_score

    def should_use_model(
            self,
            model_name: str,
            threshold: float = 0.7) -> bool:
        """
        Determine if model should be used based on health score

        Args:
            model_name: Name of the model
            threshold: Minimum health score required

        Returns:
            True if model should be used
        """
        return self.get_model_health_score(model_name) >= threshold


class GuardrailSystem:
    """
    Comprehensive guardrail system for robust forecasting
    """

    def __init__(self):
        self.validator = ForecastQualityValidator()
        self.health_monitor = ModelHealthMonitor()
        self.baseline_models = SimpleBaselineModels()

    def safe_forecast_execution(self,
                                forecast_func: Callable,
                                fallback_data: Dict[str,
                                                    Any],
                                model_name: str = "unknown") -> Tuple[pd.Series,
                                                                      Dict]:
        """
        Execute forecast with guardrails and fallback protection

        Args:
            forecast_func: Function that generates the forecast
            fallback_data: Data needed for fallback models (prices, last_price, etc.)
            model_name: Name of the model for health monitoring

        Returns:
            Tuple of (forecast series, results dict)
        """
        try:
            # Attempt primary forecast
            logger.info(f"Attempting forecast with {model_name}")
            forecast_series, results = forecast_func()

            # Validate forecast quality
            validation = self.validator.validate_forecast_quality(
                forecast_returns=results.get('forecast_returns', []),
                forecast_series=forecast_series,
                quantile_forecasts=results.get('quantile_forecasts')
            )

            # Check if forecast passes quality checks
            if validation['is_valid'] and validation['quality_score'] >= 0.5:
                self.health_monitor.record_model_attempt(model_name, True)
                results['validation'] = validation
                results['guardrails_triggered'] = False
                logger.info(
                    f"Forecast validation passed (score: {
                        validation['quality_score']:.3f})")
                return forecast_series, results
            else:
                logger.warning(
                    f"Forecast failed validation: {
                        validation['issues']}")
                raise ValueError(
                    f"Quality validation failed: {
                        validation['issues']}")

        except Exception as e:
            logger.error(f"Primary forecast failed: {e}")
            self.health_monitor.record_model_attempt(model_name, False, str(e))

            # Apply fallback strategy
            return self._apply_fallback_strategy(fallback_data, str(e))

    def _apply_fallback_strategy(self, fallback_data: Dict[str, Any],
                                 error_message: str) -> Tuple[pd.Series, Dict]:
        """
        Apply fallback strategy when primary model fails

        Args:
            fallback_data: Data for fallback models
            error_message: Error from primary model

        Returns:
            Tuple of (fallback forecast series, results dict)
        """
        logger.info("Applying fallback strategy")

        horizon = fallback_data.get('horizon', 10)

        # Try moving average fallback first
        if 'prices' in fallback_data and not fallback_data['prices'].empty:
            try:
                forecast_series, results = self.baseline_models.moving_average_forecast(
                    fallback_data['prices'], horizon)
                results['fallback_reason'] = error_message
                results['guardrails_triggered'] = True
                logger.info("Using moving average fallback")
                return forecast_series, results
            except Exception as e:
                logger.warning(f"Moving average fallback failed: {e}")

        # Last resort: random walk
        try:
            last_price = fallback_data.get('last_price', 100.0)
            volatility = fallback_data.get('volatility', 0.02)

            forecast_series, results = self.baseline_models.random_walk_forecast(
                last_price, horizon, volatility)
            results['fallback_reason'] = error_message
            results['guardrails_triggered'] = True
            logger.info("Using random walk fallback")
            return forecast_series, results

        except Exception as e:
            logger.error(f"All fallback strategies failed: {e}")

            # Ultimate fallback: flat forecast
            forecast_dates = pd.date_range(
                start=datetime.now(), periods=horizon, freq='D')
            last_price = fallback_data.get('last_price', 100.0)
            forecast_series = pd.Series(
                [last_price] * horizon, index=forecast_dates)

            results = {
                'model_type': 'flat_fallback',
                'forecast_returns': [0.0] * horizon,
                'last_price': last_price,
                'fallback_reason': error_message,
                'guardrails_triggered': True,
                'is_fallback': True
            }

            logger.warning("Using flat forecast as ultimate fallback")
            return forecast_series, results

    def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate system health report

        Returns:
            Dictionary with health metrics and recommendations
        """
        report = {
            'model_health': {},
            'recent_failures': [],
            'recommendations': [],
            'overall_health': 'good'
        }

        # Model health scores
        all_models = set(
            self.health_monitor.success_counts.keys()) | set(
            self.health_monitor.failure_counts.keys())

        for model in all_models:
            health_score = self.health_monitor.get_model_health_score(model)
            report['model_health'][model] = {
                'health_score': health_score,
                'success_count': self.health_monitor.success_counts[model],
                'failure_count': self.health_monitor.failure_counts[model]
            }

            if health_score < 0.5:
                report['recommendations'].append(
                    f"Consider disabling {model} (health: {health_score:.3f})")

        # Recent failures
        recent_failures = [
            h for h in self.health_monitor.health_history[-10:]
            if not h['success']]
        report['recent_failures'] = recent_failures

        # Overall health assessment
        if len(recent_failures) > 5:
            report['overall_health'] = 'poor'
        elif len(recent_failures) > 2:
            report['overall_health'] = 'fair'

        return report


def test_guardrail_system():
    """Test the guardrail system functionality"""
    print("=== TESTING GUARDRAIL SYSTEM ===")

    guardrails = GuardrailSystem()

    # Test 1: Successful forecast
    def good_forecast():
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 10)
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        forecast_dates = pd.date_range(
            start=datetime.now(), periods=10, freq='D')
        forecast_series = pd.Series(prices[1:], index=forecast_dates)

        return forecast_series, {
            'forecast_returns': returns.tolist(),
            'model_type': 'test_good'
        }

    # Test good forecast
    fallback_data = {'horizon': 10, 'last_price': 100.0, 'volatility': 0.02}
    _, results = guardrails.safe_forecast_execution(
        good_forecast, fallback_data, "good_model"
    )

    print(
        f"Good forecast - Guardrails triggered: {results.get('guardrails_triggered', False)}")
    print(
        f"Quality score: {
            results.get(
                'validation',
                {}).get(
                'quality_score',
                'N/A')}")

    # Test 2: Failing forecast
    def bad_forecast():
        raise ValueError("Model training failed")

    _, results = guardrails.safe_forecast_execution(
        bad_forecast, fallback_data, "bad_model"
    )

    print(
        f"Bad forecast - Guardrails triggered: {results.get('guardrails_triggered', False)}")
    print(f"Fallback model: {results.get('model_type', 'unknown')}")
    print(f"Fallback reason: {results.get('fallback_reason', 'N/A')}")

    # Test 3: Invalid forecast (monotone)
    def monotone_forecast():
        returns = [0.01] * 10  # Monotone increasing
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        forecast_dates = pd.date_range(
            start=datetime.now(), periods=10, freq='D')
        forecast_series = pd.Series(prices[1:], index=forecast_dates)

        return forecast_series, {
            'forecast_returns': returns,
            'model_type': 'test_monotone'
        }

    _, results = guardrails.safe_forecast_execution(
        monotone_forecast, fallback_data, "monotone_model"
    )

    print(
        f"Monotone forecast - Guardrails triggered: {
            results.get(
                'guardrails_triggered',
                False)}")
    print(f"Quality issues: {results.get('validation', {}).get('issues', [])}")

    # Generate health report
    health_report = guardrails.generate_health_report()
    print("\nHealth Report:")
    print(f"Overall health: {health_report['overall_health']}")
    print(f"Models tracked: {len(health_report['model_health'])}")
    print(f"Recent failures: {len(health_report['recent_failures'])}")

    return guardrails


if __name__ == "__main__":
    test_guardrail_system()
