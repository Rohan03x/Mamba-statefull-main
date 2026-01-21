"""
HIGH-PRECISION MACHINE LEARNING MODULE
======================================

Enhanced ML algorithms with pinpoint mathematical accuracy.
All optimization, loss functions, and statistical calculations
performed with maximum precision using decimal arithmetic.
"""

from decimal import Decimal
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

# Import our high-precision framework
from .precision_math import (
    HighPrecisionMath,
    PrecisionConfig,
    StatisticalMathHP,
    high_precision_context,
)

# Constants
LENGTH_MISMATCH_ERROR = "y_true and y_pred must have same length"
EMPTY_DATA_ERROR = "Cannot process empty data arrays"
INVALID_ALPHA_ERROR = "Alpha must be between 0 and 1"


class HighPrecisionLossFunctions:
    """High-precision loss functions for ML models"""

    @staticmethod
    def mean_squared_error_hp(
            y_true: List[Decimal],
            y_pred: List[Decimal]) -> Decimal:
        """MSE with maximum precision"""
        if len(y_true) != len(y_pred):
            raise ValueError(LENGTH_MISMATCH_ERROR)

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            squared_errors = [
                (true - pred) ** 2 for true,
                pred in zip(
                    y_true,
                    y_pred)]
            return sum(squared_errors) / Decimal(len(squared_errors))

    @staticmethod
    def mean_absolute_error_hp(
            y_true: List[Decimal],
            y_pred: List[Decimal]) -> Decimal:
        """MAE with maximum precision"""
        if len(y_true) != len(y_pred):
            raise ValueError(LENGTH_MISMATCH_ERROR)

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            absolute_errors = [abs(true - pred)
                               for true, pred in zip(y_true, y_pred)]
            return sum(absolute_errors) / Decimal(len(absolute_errors))

    @staticmethod
    def huber_loss_hp(y_true: List[Decimal], y_pred: List[Decimal],
                      delta: Decimal = Decimal('1.0')) -> Decimal:
        """Huber loss with maximum precision"""
        if len(y_true) != len(y_pred):
            raise ValueError(LENGTH_MISMATCH_ERROR)

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            losses = []
            for true, pred in zip(y_true, y_pred):
                error = abs(true - pred)
                if error <= delta:
                    loss = (error ** 2) / 2
                else:
                    loss = delta * error - (delta ** 2) / 2
                losses.append(loss)
            return sum(losses) / Decimal(len(losses))

    @staticmethod
    def log_cosh_loss_hp(
            y_true: List[Decimal],
            y_pred: List[Decimal]) -> Decimal:
        """Log-Cosh loss with maximum precision"""
        if len(y_true) != len(y_pred):
            raise ValueError(LENGTH_MISMATCH_ERROR)

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 10):
            losses = []
            for true, pred in zip(y_true, y_pred):
                error = true - pred
                exp_pos = HighPrecisionMath.safe_exp(error)
                exp_neg = HighPrecisionMath.safe_exp(-error)
                cosh_val = (exp_pos + exp_neg) / Decimal('2')
                loss = HighPrecisionMath.safe_log(cosh_val)
                losses.append(loss)
            return sum(losses) / Decimal(len(losses))

    @staticmethod
    def quantile_loss_hp(y_true: List[Decimal], y_pred: List[Decimal],
                         quantile: Decimal = Decimal('0.5')) -> Decimal:
        """Quantile (Pinball) loss with maximum precision"""
        if len(y_true) != len(y_pred):
            raise ValueError(LENGTH_MISMATCH_ERROR)

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            losses = []
            for true, pred in zip(y_true, y_pred):
                error = true - pred
                if error >= 0:
                    loss = quantile * error
                else:
                    loss = (quantile - Decimal('1')) * error
                losses.append(loss)
            return sum(losses) / Decimal(len(losses))


class HighPrecisionOptimizers:
    """High-precision optimization algorithms"""

    @staticmethod
    def gradient_descent_hp(
        objective_func: Callable[[Decimal], Decimal],
        gradient_func: Callable[[Decimal], Decimal],
        initial_x: Decimal,
        learning_rate: Decimal = Decimal('0.01'),
        max_iterations: int = 1000,
        tolerance: Decimal = PrecisionConfig.TOLERANCE_STRICT
    ) -> Tuple[Decimal, List[Decimal]]:
        """Gradient descent with maximum precision"""

        x = initial_x
        history = [x]

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 10):
            for _ in range(max_iterations):
                gradient = gradient_func(x)

                if abs(gradient) < tolerance:
                    break

                x_new = x - learning_rate * gradient

                if abs(x_new - x) < tolerance:
                    break

                x = x_new
                history.append(x)

        return x, history

    @staticmethod
    def adam_optimizer_hp(
        parameters: List[Decimal],
        gradients: List[Decimal],
        learning_rate: Decimal = Decimal('0.001'),
        beta1: Decimal = Decimal('0.9'),
        beta2: Decimal = Decimal('0.999'),
        epsilon: Decimal = Decimal('1e-8'),
        m: Optional[List[Decimal]] = None,
        v: Optional[List[Decimal]] = None,
        t: int = 1
    ) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Adam optimizer with maximum precision"""

        if m is None:
            m = [Decimal('0')] * len(parameters)
        if v is None:
            v = [Decimal('0')] * len(parameters)

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            # Update biased first moment estimate
            m = [beta1 * m_i + (Decimal('1') - beta1) * g_i
                 for m_i, g_i in zip(m, gradients)]

            # Update biased second raw moment estimate
            v = [beta2 * v_i + (Decimal('1') - beta2) * g_i ** 2
                 for v_i, g_i in zip(v, gradients)]

            # Compute bias-corrected first moment estimate
            beta1_t = beta1 ** t
            beta2_t = beta2 ** t

            m_hat = [m_i / (Decimal('1') - beta1_t) for m_i in m]
            v_hat = [v_i / (Decimal('1') - beta2_t) for v_i in v]

            # Update parameters
            updated_params = []
            for param, m_h, v_h in zip(parameters, m_hat, v_hat):
                denominator = HighPrecisionMath.safe_sqrt(v_h) + epsilon
                update = learning_rate * m_h / denominator
                updated_params.append(param - update)

        return updated_params, m, v


class HighPrecisionMetrics:
    """High-precision performance metrics"""

    @staticmethod
    def r_squared_hp(y_true: List[Decimal], y_pred: List[Decimal]) -> Decimal:
        """R-squared with maximum precision"""
        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            y_mean = StatisticalMathHP.mean(y_true)

            ss_res = sum(
                (true - pred) ** 2 for true,
                pred in zip(
                    y_true,
                    y_pred))
            ss_tot = sum((true - y_mean) ** 2 for true in y_true)

            if ss_tot == 0:
                return Decimal('1') if ss_res == 0 else Decimal('0')

            return Decimal('1') - ss_res / ss_tot

    @staticmethod
    def adjusted_r_squared_hp(y_true: List[Decimal], y_pred: List[Decimal],
                              num_features: int) -> Decimal:
        """Adjusted R-squared with maximum precision"""
        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            n = Decimal(len(y_true))
            k = Decimal(num_features)

            r_squared = HighPrecisionMetrics.r_squared_hp(y_true, y_pred)

            if n <= k + 1:
                return r_squared  # Cannot adjust

            adjustment = (n - Decimal('1')) / (n - k - Decimal('1'))
            return Decimal('1') - (Decimal('1') - r_squared) * adjustment

    @staticmethod
    def correlation_hp(
            y_true: List[Decimal],
            y_pred: List[Decimal]) -> Decimal:
        """Correlation coefficient with maximum precision"""
        return StatisticalMathHP.correlation(y_true, y_pred)

    @staticmethod
    def mape_hp(y_true: List[Decimal], y_pred: List[Decimal]) -> Decimal:
        """Mean Absolute Percentage Error with maximum precision"""
        if len(y_true) != len(y_pred):
            raise ValueError(LENGTH_MISMATCH_ERROR)

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            percentage_errors = []
            for true, pred in zip(y_true, y_pred):
                if true == 0:
                    continue  # Skip zero values to avoid division by zero
                error = abs((true - pred) / true) * Decimal('100')
                percentage_errors.append(error)

            if not percentage_errors:
                return Decimal('0')  # All true values were zero

            return sum(percentage_errors) / Decimal(len(percentage_errors))

    @staticmethod
    def directional_accuracy_hp(
            y_true: List[Decimal],
            y_pred: List[Decimal]) -> Decimal:
        """Directional accuracy with maximum precision"""
        if len(y_true) < 2 or len(y_pred) < 2:
            raise ValueError(
                "Need at least 2 observations for directional accuracy")

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            correct_directions = 0
            total_predictions = 0

            for i in range(1, len(y_true)):
                true_direction = y_true[i] - y_true[i-1]
                pred_direction = y_pred[i] - y_pred[i-1]

                # Check if directions match (same sign)
                if (true_direction >= 0 and pred_direction >= 0) or \
                   (true_direction < 0 and pred_direction < 0):
                    correct_directions += 1

                total_predictions += 1

            if total_predictions == 0:
                return Decimal('0')

            return Decimal(correct_directions) / \
                Decimal(total_predictions) * Decimal('100')


class HighPrecisionTechnicalIndicators:
    """High-precision technical analysis indicators"""

    @staticmethod
    def simple_moving_average_hp(
            prices: List[Decimal],
            window: int) -> List[Decimal]:
        """SMA with maximum precision"""
        if window <= 0 or window > len(prices):
            raise ValueError("Invalid window size")

        sma_values = []
        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            for i in range(window - 1, len(prices)):
                window_prices = prices[i - window + 1:i + 1]
                sma = StatisticalMathHP.mean(window_prices)
                sma_values.append(sma)

        return sma_values

    @staticmethod
    def exponential_moving_average_hp(
            prices: List[Decimal],
            window: int) -> List[Decimal]:
        """EMA with maximum precision"""
        if window <= 0:
            raise ValueError("Window must be positive")

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            alpha = Decimal('2') / (Decimal(window) + Decimal('1'))
            ema_values = [prices[0]]  # Initialize with first price

            for price in prices[1:]:
                ema = alpha * price + (Decimal('1') - alpha) * ema_values[-1]
                ema_values.append(ema)

        return ema_values

    @staticmethod
    def relative_strength_index_hp(
            prices: List[Decimal],
            window: int = 14) -> List[Decimal]:
        """RSI with maximum precision"""
        if len(prices) <= window:
            raise ValueError("Need more prices than window size")

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            # Calculate price changes
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

            # Separate gains and losses
            gains = [max(change, Decimal('0')) for change in changes]
            losses = [max(-change, Decimal('0')) for change in changes]

            rsi_values = []

            # Calculate initial averages
            avg_gain = StatisticalMathHP.mean(gains[:window])
            avg_loss = StatisticalMathHP.mean(losses[:window])

            # First RSI value
            if avg_loss == 0:
                rsi_values.append(Decimal('100'))
            else:
                rs = avg_gain / avg_loss
                rsi = Decimal('100') - Decimal('100') / (Decimal('1') + rs)
                rsi_values.append(rsi)

            # Calculate subsequent RSI values using smoothed averages
            alpha = Decimal('1') / Decimal(window)

            for i in range(window, len(changes)):
                # Update smoothed averages
                avg_gain = (Decimal('1') - alpha) * avg_gain + alpha * gains[i]
                avg_loss = (Decimal('1') - alpha) * \
                    avg_loss + alpha * losses[i]

                if avg_loss == 0:
                    rsi_values.append(Decimal('100'))
                else:
                    rs = avg_gain / avg_loss
                    rsi = Decimal('100') - Decimal('100') / (Decimal('1') + rs)
                    rsi_values.append(rsi)

        return rsi_values

    @staticmethod
    def bollinger_bands_hp(prices: List[Decimal], window: int = 20, num_std: Decimal = Decimal(
            '2')) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Bollinger Bands with maximum precision"""
        if window <= 0 or window > len(prices):
            raise ValueError("Invalid window size")

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            middle_band = []
            upper_band = []
            lower_band = []

            for i in range(window - 1, len(prices)):
                window_prices = prices[i - window + 1:i + 1]

                # Calculate moving average (middle band)
                ma = StatisticalMathHP.mean(window_prices)
                middle_band.append(ma)

                # Calculate standard deviation
                std_dev = StatisticalMathHP.standard_deviation(window_prices)

                # Calculate bands
                upper = ma + num_std * std_dev
                lower = ma - num_std * std_dev

                upper_band.append(upper)
                lower_band.append(lower)

        return upper_band, middle_band, lower_band


class HighPrecisionVolatility:
    """High-precision volatility calculations"""

    @staticmethod
    def historical_volatility_hp(
            prices: List[Decimal],
            window: int = 252) -> Decimal:
        """Historical volatility with maximum precision"""
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices")

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 10):
            # Calculate log returns
            log_returns = []
            for i in range(1, len(prices)):
                if prices[i-1] <= 0 or prices[i] <= 0:
                    raise ValueError("Prices must be positive for log returns")
                log_return = HighPrecisionMath.safe_log(
                    prices[i] / prices[i-1])
                log_returns.append(log_return)

            # Calculate standard deviation of returns
            std_dev = StatisticalMathHP.standard_deviation(log_returns)

            # Annualize volatility
            trading_days = Decimal(window)
            annual_vol = std_dev * HighPrecisionMath.safe_sqrt(trading_days)

            return annual_vol

    @staticmethod
    def ewma_volatility_hp(
            returns: List[Decimal],
            lambda_param: Decimal = Decimal('0.94')) -> List[Decimal]:
        """EWMA volatility with maximum precision"""
        if not returns:
            raise ValueError("Returns list cannot be empty")

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 10):
            volatilities = []

            # Initialize with first return squared
            var_ewma = returns[0] ** 2
            volatilities.append(HighPrecisionMath.safe_sqrt(var_ewma))

            # Calculate EWMA volatility
            for return_val in returns[1:]:
                var_ewma = lambda_param * var_ewma + \
                    (Decimal('1') - lambda_param) * (return_val ** 2)
                vol_ewma = HighPrecisionMath.safe_sqrt(var_ewma)
                volatilities.append(vol_ewma)

        return volatilities


class HighPrecisionEnsemble:
    """High-precision ensemble methods"""

    @staticmethod
    def weighted_average_hp(predictions: List[List[Decimal]],
                            weights: List[Decimal]) -> List[Decimal]:
        """Weighted average ensemble with maximum precision"""
        if len(predictions) != len(weights):
            raise ValueError(
                "Number of prediction sets must equal number of weights")

        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        # Normalize weights
        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 5):
            weight_sum = sum(weights)
            if weight_sum == 0:
                raise ValueError("Sum of weights cannot be zero")

            normalized_weights = [w / weight_sum for w in weights]

            # Calculate weighted average for each time step
            ensemble_predictions = []
            num_timesteps = len(predictions[0])

            for t in range(num_timesteps):
                weighted_sum = Decimal('0')
                for i, pred_set in enumerate(predictions):
                    if t < len(pred_set):
                        weighted_sum += normalized_weights[i] * pred_set[t]

                ensemble_predictions.append(weighted_sum)

        return ensemble_predictions

    @staticmethod
    def inverse_variance_weighting_hp(
            predictions: List[List[Decimal]],
            errors: List[List[Decimal]]) -> List[Decimal]:
        """Inverse variance weighting with maximum precision"""
        if len(predictions) != len(errors):
            raise ValueError(
                "Number of prediction and error sets must be equal")

        with high_precision_context(PrecisionConfig.DECIMAL_PRECISION + 10):
            # Calculate variances for each model
            variances = []
            for error_set in errors:
                variance = StatisticalMathHP.variance(error_set)
                variances.append(variance)

            # Calculate inverse variance weights
            weights = []
            for var in variances:
                if var == 0:
                    # Very high weight for zero variance
                    weight = Decimal('1e10')
                else:
                    weight = Decimal('1') / var
                weights.append(weight)

            # Apply weighted average
            return HighPrecisionEnsemble.weighted_average_hp(
                predictions, weights)


# PyTorch integration for high-precision training
class HighPrecisionTraining:
    """High-precision neural network training utilities"""

    @staticmethod
    def setup_high_precision_model(model: nn.Module) -> nn.Module:
        """Configure model for high-precision training"""
        # Convert model to double precision
        model = model.double()

        # Set high precision as default
        torch.set_default_dtype(PrecisionConfig.TORCH_PRECISION)

        return model

    @staticmethod
    def high_precision_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                            loss_type: str = 'mse') -> torch.Tensor:
        """Calculate loss with high precision"""
        # Ensure tensors are in high precision
        y_true = y_true.to(PrecisionConfig.TORCH_PRECISION)
        y_pred = y_pred.to(PrecisionConfig.TORCH_PRECISION)

        if loss_type == 'mse':
            return torch.mean((y_true - y_pred) ** 2)
        elif loss_type == 'mae':
            return torch.mean(torch.abs(y_true - y_pred))
        elif loss_type == 'huber':
            delta = 1.0
            error = torch.abs(y_true - y_pred)
            quadratic = torch.clamp(error, max=delta)
            linear = error - quadratic
            return torch.mean(0.5 * quadratic ** 2 + delta * linear)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test high-precision ML components
    print("🎯 High-Precision Machine Learning Framework")
    print("============================================")

    # Test data
    y_true = [Decimal('1.1'), Decimal('2.2'), Decimal('3.3'), Decimal('4.4')]
    y_pred = [
        Decimal('1.05'),
        Decimal('2.15'),
        Decimal('3.35'),
        Decimal('4.5')]

    # Test loss functions
    print("\n=== High-Precision Loss Functions ===")
    mse = HighPrecisionLossFunctions.mean_squared_error_hp(y_true, y_pred)
    mae = HighPrecisionLossFunctions.mean_absolute_error_hp(y_true, y_pred)
    huber = HighPrecisionLossFunctions.huber_loss_hp(y_true, y_pred)

    print(f"MSE (50 digits): {mse}")
    print(f"MAE (50 digits): {mae}")
    print(f"Huber Loss (50 digits): {huber}")

    # Test metrics
    print("\n=== High-Precision Metrics ===")
    r2 = HighPrecisionMetrics.r_squared_hp(y_true, y_pred)
    corr = HighPrecisionMetrics.correlation_hp(y_true, y_pred)
    mape = HighPrecisionMetrics.mape_hp(y_true, y_pred)

    print(f"R-squared (50 digits): {r2}")
    print(f"Correlation (50 digits): {corr}")
    print(f"MAPE (50 digits): {mape}")

    # Test technical indicators
    print("\n=== High-Precision Technical Indicators ===")
    prices = [
        Decimal('100'),
        Decimal('101'),
        Decimal('102'),
        Decimal('103'),
        Decimal('104')]
    sma = HighPrecisionTechnicalIndicators.simple_moving_average_hp(prices, 3)
    ema = HighPrecisionTechnicalIndicators.exponential_moving_average_hp(
        prices, 3)

    print(f"SMA (50 digits): {sma}")
    print(f"EMA (50 digits): {ema}")

    print("\n✅ High-precision ML framework ready for deployment!")
