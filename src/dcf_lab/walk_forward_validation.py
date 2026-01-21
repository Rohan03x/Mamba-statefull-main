"""
Walk-Forward Validation System for AI Price Forecasting

This module implements comprehensive walk-forward backtesting to validate forecasting
models on out-of-sample data with realistic market conditions.

Key Features:
- Rolling window walk-forward validation
- Multiple performance metrics (MAPE, RMSE, Directional Accuracy, etc.)
- Quantile forecast evaluation (Pinball Loss, CRPS)
- Statistical significance testing
- Performance degradation detection
- Regime-specific performance analysis

        # Financial performance metrics (strategy returns using predicted direction)
        strategy_returns = actual_backend * predicted_direction
        financial_metrics = self._compute_financial_metrics(strategy_returns)
        metrics.sharpe_ratio = financial_metrics['sharpe_ratio']
        metrics.sortino_ratio = financial_metrics['sortino_ratio']
        metrics.max_drawdown = financial_metrics['max_drawdown']
        metrics.calmar_ratio = financial_metrics['calmar_ratio']
- Real-time validation monitoring

Validation Methodology:
- Anchored walk-forward: Fixed training start, expanding window
- Rolling walk-forward: Fixed training window size, rolling forward
- Blocked cross-validation: Respect temporal dependencies
- Out-of-sample forecast evaluation with proper data alignment
"""

import json
import logging
import pickle
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Parallel = None
    delayed = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for walk-forward validation"""
    # Training window parameters
    min_training_days: int = 252  # Minimum 1 year of training data
    # Maximum 5 years (None for expanding)
    max_training_days: Optional[int] = 1260
    validation_step_days: int = 21  # Validate every 3 weeks
    forecast_horizon: int = 5  # Days to forecast ahead

    # Validation periods
    start_date: Optional[str] = None  # Auto-detect if None
    end_date: Optional[str] = None    # Use latest if None
    min_validation_periods: int = 20  # Minimum validation windows

    # Performance metrics
    quantiles: List[float] = field(
        default_factory=lambda: [
            0.1, 0.25, 0.5, 0.75, 0.9])
    target_column: str = 'log_return_5d'
    price_column: str = 'close'

    # Statistical testing
    confidence_level: float = 0.95
    benchmark_model: str = 'random_walk'  # Benchmark for comparison

    # Output options
    save_results: bool = True
    generate_plots: bool = True
    results_dir: str = 'validation_results'

    # GPU/parallelism options
    fold_batch_size: int = 1  # Number of validation windows to process together
    use_cuda_streams: bool = False  # Enable CUDA stream overlap when available
    enable_gpu_metrics: bool = False  # Compute metrics with GPU backends (CuPy/cuDF)
    gpu_metrics_backend: str = 'cupy'  # Backend for GPU metrics ('cupy' or 'torch')
    cpu_workers: int = 1  # Joblib worker count for CPU parallel folds
    joblib_backend: str = 'threading'  # Joblib backend ('threading', 'loky', etc.)
    data_backend: str = 'pandas'  # Data manipulation backend ('pandas' or 'cudf')


@dataclass
class ValidationMetrics:
    """Container for validation performance metrics"""
    # Basic forecast accuracy
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0

    # Directional accuracy
    direction_accuracy: float = 0.0
    hit_rate: float = 0.0

    # Classification-style metrics on directional signals
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Financial performance
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Quantile forecast metrics
    pinball_losses: Dict[float, float] = field(default_factory=dict)
    crps: float = 0.0
    quantile_coverage: Dict[float, float] = field(default_factory=dict)

    # Risk metrics
    max_error: float = 0.0
    error_volatility: float = 0.0

    # Statistical significance
    p_value_vs_benchmark: float = 1.0
    is_significant: bool = False

    # Temporal stability
    performance_trend: float = 0.0  # Slope of performance over time
    performance_stability: float = 0.0  # 1 - CV of performance

    # Sample info
    n_predictions: int = 0
    validation_period: str = ""


class PerformanceCalculator:
    """Calculate comprehensive performance metrics for forecasts"""

    def __init__(self, quantiles: List[float] = None,
                 use_gpu_metrics: bool = False,
                 gpu_backend: str = 'cupy'):
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.use_gpu_metrics = use_gpu_metrics
        self.gpu_backend = gpu_backend.lower() if gpu_backend else 'cupy'
        self._xp = np
        self._cp = None

        if self.use_gpu_metrics:
            if self.gpu_backend == 'cupy':
                try:
                    import cupy as cp  # type: ignore

                    self._xp = cp
                    self._cp = cp
                    logger.info("Using CuPy backend for GPU metrics calculation")
                except Exception as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "CuPy backend unavailable (%s). Falling back to NumPy for metrics.",
                        exc)
                    self.use_gpu_metrics = False
                    self._xp = np
            else:
                logger.warning(
                    "GPU metrics backend '%s' not supported yet. Using NumPy.",
                    self.gpu_backend)
                self.use_gpu_metrics = False
                self._xp = np

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_backend_array(self, array: np.ndarray):
        """Convert array to the configured backend representation."""
        if self.use_gpu_metrics and self._cp is not None:
            return self._cp.asarray(array)
        return np.asarray(array)

    def _to_numpy(self, array):
        """Convert backend arrays back to NumPy."""
        if self.use_gpu_metrics and self._cp is not None:
            return self._cp.asnumpy(array)
        return np.asarray(array)

    def _value_to_float(self, value: Any) -> float:
        """Convert backend scalar to Python float."""
        if self.use_gpu_metrics and self._cp is not None:
            return float(self._cp.asnumpy(value))
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

    def calculate_metrics(self,
                          actual: np.ndarray,
                          predicted: np.ndarray,
                          quantile_forecasts: Optional[Dict[str,
                                                            np.ndarray]] = None) -> ValidationMetrics:
        """
        Calculate comprehensive validation metrics

        Args:
            actual: Actual target values (returns)
            predicted: Predicted target values (returns)
            quantile_forecasts: Dict of quantile predictions
            prices_actual: Actual prices (for additional metrics)
            prices_predicted: Predicted prices (for additional metrics)

        Returns:
            ValidationMetrics object with all calculated metrics
        """
        metrics = ValidationMetrics()

        # Convert to backend arrays (CuPy if enabled)
        actual_backend = self._to_backend_array(actual)
        predicted_backend = self._to_backend_array(predicted)

        # Basic accuracy metrics
        abs_errors = self._xp.abs(actual_backend - predicted_backend)
        metrics.mae = self._value_to_float(self._xp.mean(abs_errors))
        mse = self._xp.mean((actual_backend - predicted_backend) ** 2)
        metrics.rmse = self._value_to_float(self._xp.sqrt(mse))

        # MAPE (handle division by zero)
        denom = self._xp.where(self._xp.abs(actual_backend) > 1e-8,
                               actual_backend, 1e-8)
        mape_values = self._xp.abs((actual_backend - predicted_backend) / denom)
        metrics.mape = self._value_to_float(self._xp.mean(mape_values) * 100.0)

        # Directional accuracy
        actual_direction = self._xp.sign(actual_backend)
        predicted_direction = self._xp.sign(predicted_backend)
        direction_match = actual_direction == predicted_direction
        metrics.direction_accuracy = self._value_to_float(
            self._xp.mean(direction_match))

        # Hit rate (within reasonable bounds)
        error_threshold = self._value_to_float(self._xp.std(actual_backend) * 0.5)
        within_threshold = self._xp.abs(actual_backend - predicted_backend) <= error_threshold
        metrics.hit_rate = self._value_to_float(self._xp.mean(within_threshold))

        # Directional precision/recall treating positive returns as the target class
        actual_positive = actual_backend > 0
        predicted_positive = predicted_backend > 0
        true_positive = self._xp.sum(
            self._xp.logical_and(predicted_positive, actual_positive)
        )
        false_positive = self._xp.sum(
            self._xp.logical_and(predicted_positive, self._xp.logical_not(actual_positive))
        )
        false_negative = self._xp.sum(
            self._xp.logical_and(self._xp.logical_not(predicted_positive), actual_positive)
        )

        precision_backend = true_positive / (true_positive + false_positive + 1e-8)
        recall_backend = true_positive / (true_positive + false_negative + 1e-8)
        f1_backend = (2.0 * precision_backend * recall_backend) / (
            precision_backend + recall_backend + 1e-8
        )

        metrics.precision = self._value_to_float(precision_backend)
        metrics.recall = self._value_to_float(recall_backend)
        metrics.f1_score = self._value_to_float(f1_backend)

        # Financial performance metrics on strategy-aligned returns
        strategy_returns_backend = actual_backend * predicted_direction
        fin_metrics = self._compute_financial_metrics(strategy_returns_backend)
        metrics.sharpe_ratio = fin_metrics['sharpe_ratio']
        metrics.sortino_ratio = fin_metrics['sortino_ratio']
        metrics.max_drawdown = fin_metrics['max_drawdown']
        metrics.calmar_ratio = fin_metrics['calmar_ratio']

        # Risk metrics remain on the active backend
        errors_backend = actual_backend - predicted_backend
        metrics.max_error = self._value_to_float(
            self._xp.max(self._xp.abs(errors_backend))
        )
        metrics.error_volatility = self._value_to_float(
            self._xp.std(errors_backend)
        )

        # Quantile forecast evaluation (requires NumPy representations)
        if quantile_forecasts:
            actual_np = self._to_numpy(actual_backend)
            metrics.pinball_losses = self._calculate_pinball_losses(
                actual_np, quantile_forecasts)
            metrics.crps = self._calculate_crps(actual_np, quantile_forecasts)
            metrics.quantile_coverage = self._calculate_coverage(
                actual_np, quantile_forecasts)

        # Sample info
        metrics.n_predictions = int(actual_backend.size)

        return metrics

    def _compute_financial_metrics(self, returns_array) -> Dict[str, float]:
        """Compute Sharpe, Sortino, and drawdown metrics on backend arrays."""
        if returns_array is None:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            }

        xp = self._cp if (self.use_gpu_metrics and self._cp is not None) else np

        if xp is np:
            returns_backend = np.asarray(returns_array, dtype=float)
        else:
            if isinstance(returns_array, xp.ndarray):
                returns_backend = returns_array
            else:
                returns_backend = xp.asarray(returns_array)

        if returns_backend.size == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            }

        mean_return = xp.mean(returns_backend)
        std_return = xp.std(returns_backend)
        downside_returns = xp.clip(returns_backend, None, 0.0)
        downside = xp.std(downside_returns)

        sqrt_252 = xp.sqrt(252.0)
        sharpe_backend = (mean_return / (std_return + 1e-8)) * sqrt_252
        sortino_backend = (mean_return / (downside + 1e-8)) * sqrt_252

        cumulative = xp.cumprod(1.0 + returns_backend)
        running_max = xp.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = self._value_to_float(xp.min(drawdowns))

        annual_return = self._value_to_float((1.0 + mean_return) ** 252 - 1.0)
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

        return {
            'sharpe_ratio': self._value_to_float(sharpe_backend),
            'sortino_ratio': self._value_to_float(sortino_backend),
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar
        }

    def _calculate_pinball_losses(self,
                                  actual: np.ndarray,
                                  quantile_forecasts: Dict[str,
                                                           np.ndarray]) -> Dict[float,
                                                                                float]:
        """Calculate pinball loss for each quantile"""
        pinball_losses = {}

        for q in self.quantiles:
            q_key = f'q_{int(q*100)}'
            if q_key in quantile_forecasts:
                q_pred = quantile_forecasts[q_key]
                if len(q_pred.shape) > 1:
                    q_pred = q_pred[:,
                                    0] if q_pred.shape[1] > 0 else q_pred.flatten()

                # Ensure same length
                min_len = min(len(actual), len(q_pred))
                actual_subset = actual[:min_len]
                q_pred_subset = q_pred[:min_len]

                # Pinball loss
                errors = actual_subset - q_pred_subset
                loss = np.where(errors >= 0, q * errors, (q - 1) * errors)
                pinball_losses[q] = np.mean(loss)

        return pinball_losses

    def _calculate_crps(self, actual: np.ndarray,
                        quantile_forecasts: Dict[str, np.ndarray]) -> float:
        """Calculate Continuous Ranked Probability Score"""
        try:
            # Get quantile predictions in order
            quantile_data = self._extract_quantile_data(
                actual, quantile_forecasts)
            if not quantile_data:
                return float('inf')

            quantile_levels, quantile_values, min_len = quantile_data
            actual_subset = actual[:min_len]

            # Calculate CRPS for each observation
            crps_values = []
            for i in range(min_len):
                obs = actual_subset[i]
                q_preds = [qv[i] for qv in quantile_values]

                # Approximate CRPS using quantiles
                crps_i = sum(((1.0 if obs <= q_pred else 0.0) - q_level) ** 2
                             for q_level, q_pred in zip(quantile_levels, q_preds))
                crps_values.append(crps_i)

            return np.mean(crps_values)

        except Exception as e:
            logger.warning(f"CRPS calculation failed: {e}")
            return float('inf')

    def _extract_quantile_data(self,
                               actual: np.ndarray,
                               quantile_forecasts: Dict[str,
                                                        np.ndarray]) -> Optional[Tuple]:
        """Extract and prepare quantile data for CRPS calculation"""
        quantile_values = []
        quantile_levels = []

        for q in sorted(self.quantiles):
            q_key = f'q_{int(q*100)}'
            if q_key in quantile_forecasts:
                q_pred = quantile_forecasts[q_key]
                if len(q_pred.shape) > 1:
                    q_pred = q_pred[:,
                                    0] if q_pred.shape[1] > 0 else q_pred.flatten()

                quantile_values.append(q_pred)
                quantile_levels.append(q)

        if not quantile_values:
            return None

        # Ensure all arrays have same length
        min_len = min(len(actual), min(len(qv) for qv in quantile_values))
        return quantile_levels, quantile_values, min_len

    def _calculate_coverage(self,
                            actual: np.ndarray,
                            quantile_forecasts: Dict[str,
                                                     np.ndarray]) -> Dict[float,
                                                                          float]:
        """Calculate empirical coverage for prediction intervals"""
        coverage = {}
        coverage_levels = [0.8, 0.6, 0.4]  # 80%, 60%, 40% intervals

        for level in coverage_levels:
            empirical_coverage = self._calculate_single_coverage(
                actual, quantile_forecasts, level)
            if empirical_coverage is not None:
                coverage[level] = empirical_coverage

        return coverage

    def _calculate_single_coverage(self, actual: np.ndarray,
                                   quantile_forecasts: Dict[str, np.ndarray],
                                   level: float) -> Optional[float]:
        """Calculate coverage for a single confidence level"""
        lower_q = (1 - level) / 2
        upper_q = 1 - lower_q

        lower_key = f'q_{int(lower_q*100)}'
        upper_key = f'q_{int(upper_q*100)}'

        if lower_key not in quantile_forecasts or upper_key not in quantile_forecasts:
            return None

        # Extract and prepare predictions
        lower_pred = self._prepare_prediction_array(
            quantile_forecasts[lower_key])
        upper_pred = self._prepare_prediction_array(
            quantile_forecasts[upper_key])

        min_len = min(len(actual), len(lower_pred), len(upper_pred))
        actual_subset = actual[:min_len]
        lower_subset = lower_pred[:min_len]
        upper_subset = upper_pred[:min_len]

        # Calculate empirical coverage
        within_interval = (
            actual_subset >= lower_subset) & (
            actual_subset <= upper_subset)
        return np.mean(within_interval)

    def _prepare_prediction_array(self, pred_array: np.ndarray) -> np.ndarray:
        """Prepare prediction array by flattening if needed"""
        if len(pred_array.shape) > 1:
            return pred_array[:,
                              0] if pred_array.shape[1] > 0 else pred_array.flatten()
        return pred_array


class WalkForwardValidator:
    """Main walk-forward validation system"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.performance_calc = PerformanceCalculator(
            config.quantiles,
            use_gpu_metrics=config.enable_gpu_metrics,
            gpu_backend=config.gpu_metrics_backend
        )
        backend = (config.data_backend or 'pandas').lower()
        self._data_backend = backend if backend in {'pandas', 'cudf'} else 'pandas'
        self._cudf_available = False
        if self._data_backend == 'cudf':
            try:
                self._cudf_available = True
                logger.info("Walk-forward validator using cuDF data backend")
            except Exception as exc:
                logger.warning("cuDF backend requested but unavailable (%s); reverting to pandas", exc)
                self._data_backend = 'pandas'
        self.validation_results = []
        self.summary_stats = {}
        # Optional extra diagnostics (provenance/telemetry) provided by caller
        self._extra_diagnostics: Dict[str, Any] = {'provenance': None, 'telemetry': None}

        # Create results directory
        if config.save_results:
            Path(config.results_dir).mkdir(exist_ok=True)

    def validate_model(self,
                       data: pd.DataFrame,
                       model_factory: Callable,
                       feature_columns: List[str],
                       provenance: Optional[Dict[str, Any]] = None,
                       telemetry: Optional[Dict[str, Any]] = None,
                       **model_kwargs) -> Dict[str, Any]:
        """
        Run walk-forward validation on a model

        Args:
            data: DataFrame with features and targets
            model_factory: Function that creates and returns a model instance
            feature_columns: List of feature column names
            **model_kwargs: Additional arguments for model creation

        Returns:
            Comprehensive validation results
        """
        logger.info("Starting walk-forward validation...")
        # Capture optional diagnostics from caller for persistence
        if provenance is not None:
            self._extra_diagnostics['provenance'] = provenance
        if telemetry is not None:
            self._extra_diagnostics['telemetry'] = telemetry

        # Prepare data
        data_clean = self._prepare_data(data)
        validation_windows = self._generate_validation_windows(data_clean)

        logger.info(f"Generated {len(validation_windows)} validation windows")

        # Run validation for each window
        if self.config.fold_batch_size <= 1:
            window_results = self._run_windows_sequential(
                data_clean,
                validation_windows,
                model_factory,
                feature_columns,
                **model_kwargs
            )
        else:
            window_results = self._run_windows_batched(
                data_clean,
                validation_windows,
                model_factory,
                feature_columns,
                **model_kwargs
            )

        # Aggregate results
        aggregated_results = self._aggregate_results(window_results)

        # Save results if requested
        if self.config.save_results:
            self._save_results(aggregated_results, window_results)

        # Generate plots if requested
        if self.config.generate_plots:
            self._generate_plots(window_results)

        logger.info("Walk-forward validation completed")
        return aggregated_results

    def _run_windows_sequential(self,
                                data: pd.DataFrame,
                                windows: List[Tuple[datetime, datetime, datetime, datetime]],
                                model_factory: Callable,
                                feature_columns: List[str],
                                **model_kwargs) -> List[Dict[str, Any]]:
        """Execute validation windows one-by-one (original behaviour)."""
        results: List[Dict[str, Any]] = []
        total = len(windows)

        use_joblib = self._should_use_joblib()

        if use_joblib:
            return self._run_windows_joblib(
                data,
                windows,
                model_factory,
                feature_columns,
                total,
                **model_kwargs
            )

        if getattr(self.config, 'cpu_workers', 1) > 1 and not use_joblib:
            logger.warning(
                "cpu_workers=%s requested but joblib is unavailable; falling back to sequential execution.",
                getattr(self.config, 'cpu_workers', 1)
            )

        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            logger.info(
                "Validation window %s/%s: %s to %s -> %s to %s",
                i + 1,
                total,
                train_start.date(),
                train_end.date(),
                val_start.date(),
                val_end.date()
            )

            try:
                result = self._validate_single_window(
                    data,
                    train_start,
                    train_end,
                    val_start,
                    val_end,
                    model_factory,
                    feature_columns,
                    runtime_overrides=None,
                    fold_idx=i,
                    **model_kwargs
                )
                results.append(result)
            except Exception as exc:  # pragma: no cover - logging only
                logger.error("Validation window %s failed: %s", i + 1, exc)

        return results

    def _should_use_joblib(self) -> bool:
        """Determine whether joblib-based execution is available and requested."""
        if Parallel is None or delayed is None:
            return False
        workers = getattr(self.config, 'cpu_workers', 1)
        return workers is not None and int(workers) > 1

    def _run_windows_joblib(self,
                             data: pd.DataFrame,
                             windows: List[Tuple[datetime, datetime, datetime, datetime]],
                             model_factory: Callable,
                             feature_columns: List[str],
                             total: int,
                             **model_kwargs) -> List[Dict[str, Any]]:
        """Execute validation windows in parallel using joblib."""
        backend = getattr(self.config, 'joblib_backend', 'threading') or 'threading'
        n_jobs = int(getattr(self.config, 'cpu_workers', 1))
        logger.info(
            "Executing validation windows in parallel with joblib (n_jobs=%s, backend=%s)",
            n_jobs,
            backend
        )

        def _joblib_validate(idx: int, window_tuple: Tuple[datetime, datetime, datetime, datetime]):
            train_start, train_end, val_start, val_end = window_tuple
            logger.info(
                "Validation window %s/%s: %s to %s -> %s to %s",
                idx + 1,
                total,
                train_start.date(),
                train_end.date(),
                val_start.date(),
                val_end.date()
            )
            try:
                result = self._validate_single_window(
                    data,
                    train_start,
                    train_end,
                    val_start,
                    val_end,
                    model_factory,
                    feature_columns,
                    runtime_overrides=None,
                    fold_idx=idx,
                    **model_kwargs
                )
                return idx, result, None
            except Exception as exc:  # pragma: no cover - logging only
                return idx, None, exc

        job_outputs = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_joblib_validate)(idx, window_tuple)
            for idx, window_tuple in enumerate(windows)
        )

        results: List[Dict[str, Any]] = []
        for idx, result, exc in job_outputs:
            if exc is not None:
                logger.error("Validation window %s failed: %s", idx + 1, exc)
            elif result is not None:
                results.append(result)

        return results

    def _run_windows_batched(self,
                              data: pd.DataFrame,
                              windows: List[Tuple[datetime, datetime, datetime, datetime]],
                              model_factory: Callable,
                              feature_columns: List[str],
                              **model_kwargs) -> List[Dict[str, Any]]:
        """Execute validation windows in batches with optional GPU overlap."""
        results: List[Dict[str, Any]] = []
        batch_size = max(1, self.config.fold_batch_size)
        total_batches = math.ceil(len(windows) / batch_size) if windows else 0

        for batch_idx, batch_windows in enumerate(self._batch_windows(windows)):
            logger.info(
                "Processing batch %s/%s with %s folds",
                batch_idx + 1,
                total_batches,
                len(batch_windows)
            )

            batch_results = self._validate_window_batch(
                data,
                batch_windows,
                model_factory,
                feature_columns,
                **model_kwargs
            )
            results.extend(batch_results)

        return results

    def _batch_windows(self, windows: List[Tuple[datetime, datetime, datetime, datetime]]):
        """Yield windows in batches respecting the configured batch size."""
        batch_size = max(1, self.config.fold_batch_size)
        for start in range(0, len(windows), batch_size):
            yield windows[start:start + batch_size]

    def _validate_window_batch(self,
                               data: pd.DataFrame,
                               batch_windows: List[Tuple[datetime, datetime, datetime, datetime]],
                               model_factory: Callable,
                               feature_columns: List[str],
                               **model_kwargs) -> List[Dict[str, Any]]:
        """Validate a batch of windows using GPU streams or CPU threads."""
        if not batch_windows:
            return []

        if self.config.use_cuda_streams:
            gpu_results = self._validate_windows_with_cuda_streams(
                data,
                batch_windows,
                model_factory,
                feature_columns,
                **model_kwargs
            )
            if gpu_results is not None:
                return gpu_results

        return self._validate_windows_threadpool(
            data,
            batch_windows,
            model_factory,
            feature_columns,
            **model_kwargs
        )

    def _validate_windows_with_cuda_streams(self,
                                            data: pd.DataFrame,
                                            batch_windows: List[Tuple[datetime, datetime, datetime, datetime]],
                                            model_factory: Callable,
                                            feature_columns: List[str],
                                            **model_kwargs) -> Optional[List[Dict[str, Any]]]:
        """Attempt to validate folds concurrently using CUDA streams."""
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("PyTorch not available for CUDA streams (%s)", exc)
            return None

        if not torch.cuda.is_available():
            logger.warning("CUDA device not detected. Falling back to CPU execution.")
            return None

        results: List[Optional[Dict[str, Any]]] = [None] * len(batch_windows)

        def _execute_fold(index: int,
                          window_tuple: Tuple[datetime, datetime, datetime, datetime],
                          stream: "torch.cuda.Stream") -> Optional[Dict[str, Any]]:
            train_start, train_end, val_start, val_end = window_tuple
            device = torch.device('cuda')

            runtime_overrides = {
                'device': device,
                'cuda_stream': stream
            }

            with torch.cuda.device(device):
                with torch.cuda.stream(stream):
                    return self._validate_single_window(
                        data,
                        train_start,
                        train_end,
                        val_start,
                        val_end,
                        model_factory,
                        feature_columns,
                        runtime_overrides=runtime_overrides,
                        fold_idx=index,
                        **model_kwargs
                    )

        futures = {}
        with ThreadPoolExecutor(max_workers=len(batch_windows)) as executor:
            for idx, window_tuple in enumerate(batch_windows):
                stream = torch.cuda.Stream()
                future = executor.submit(_execute_fold, idx, window_tuple, stream)
                futures[future] = (idx, stream)

            for future in as_completed(futures):
                idx, stream = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # pragma: no cover - logging only
                    logger.error("CUDA stream fold %s failed: %s", idx, exc)
                finally:
                    stream.synchronize()

        torch.cuda.synchronize()

        return [result for result in results if result is not None]

    def _validate_windows_threadpool(self,
                                      data: pd.DataFrame,
                                      batch_windows: List[Tuple[datetime, datetime, datetime, datetime]],
                                      model_factory: Callable,
                                      feature_columns: List[str],
                                      **model_kwargs) -> List[Dict[str, Any]]:
        """Fallback execution using a thread pool on CPU."""
        results: List[Optional[Dict[str, Any]]] = [None] * len(batch_windows)

        with ThreadPoolExecutor(max_workers=len(batch_windows)) as executor:
            future_map = {}
            for idx, (train_start, train_end, val_start, val_end) in enumerate(batch_windows):
                future = executor.submit(
                    self._validate_single_window,
                    data,
                    train_start,
                    train_end,
                    val_start,
                    val_end,
                    model_factory,
                    feature_columns,
                    runtime_overrides=None,
                    fold_idx=idx,
                    **model_kwargs
                )
                future_map[future] = idx

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # pragma: no cover - logging only
                    logger.error("Validation fold %s failed in thread pool: %s", idx, exc)

        return [result for result in results if result is not None]

    # ------------------------------------------------------------------
    # Single Fold Helpers
    # ------------------------------------------------------------------
    def _prepare_model_runtime(self, model: Any, runtime_overrides: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Configure model/device context and return kwargs for fit/predict."""
        device = runtime_overrides.get('device')
        cuda_stream = runtime_overrides.get('cuda_stream')

        fit_kwargs = dict(runtime_overrides.get('fit_kwargs', {}))
        predict_kwargs = dict(runtime_overrides.get('predict_kwargs', {}))

        if device is not None and hasattr(model, 'to'):
            try:
                model.to(device)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Model.to(%s) failed: %s", device, exc)

        if device is not None:
            fit_kwargs.setdefault('device', device)
            predict_kwargs.setdefault('device', device)

        if cuda_stream is not None:
            fit_kwargs.setdefault('cuda_stream', cuda_stream)
            predict_kwargs.setdefault('cuda_stream', cuda_stream)

        if hasattr(model, 'configure_runtime'):
            try:
                model.configure_runtime(runtime_overrides)
            except Exception as exc:  # pragma: no cover - optional hook
                logger.debug("Model runtime configuration failed: %s", exc)

        return fit_kwargs, predict_kwargs

    def _fit_model(self, model: Any, train_data: pd.DataFrame,
                   feature_columns: List[str], fit_kwargs: Dict[str, Any]):
        """Fit model with graceful fallback for legacy signatures."""
        if not hasattr(model, 'fit'):
            raise ValueError("Model must have a 'fit' method")

        try:
            if fit_kwargs:
                model.fit(train_data, feature_columns, **fit_kwargs)
            else:
                model.fit(train_data, feature_columns)
        except TypeError:
            model.fit(train_data, feature_columns)

    def _predict_model(self, model: Any, val_data: pd.DataFrame,
                       feature_columns: List[str], predict_kwargs: Dict[str, Any]):
        """Generate predictions while preserving backward compatibility."""
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")

        try:
            if predict_kwargs:
                return model.predict(val_data.iloc[[-1]], feature_columns, **predict_kwargs)
            return model.predict(val_data.iloc[[-1]], feature_columns)
        except TypeError:
            return model.predict(val_data.iloc[[-1]], feature_columns)

    def _ensure_numpy_array(self, array_like: Any):
        """Convert model outputs to NumPy arrays regardless of backend."""
        if array_like is None:
            return None

        if isinstance(array_like, np.ndarray):
            return array_like

        if self.performance_calc.use_gpu_metrics and self.performance_calc._cp is not None:
            if isinstance(array_like, self.performance_calc._cp.ndarray):
                return self.performance_calc._cp.asnumpy(array_like)

        try:
            import torch  # type: ignore
            if torch.is_tensor(array_like):
                return array_like.detach().cpu().numpy()
        except Exception:
            pass

        if hasattr(array_like, 'to_numpy'):
            try:
                return array_like.to_numpy()
            except Exception:
                pass

        if isinstance(array_like, (list, tuple)):
            return np.asarray(array_like)

        return np.array(array_like)

    def _normalize_predictions(self, predictions: Any) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Normalize various prediction formats into NumPy arrays."""
        if isinstance(predictions, dict):
            normalized = {
                key: self._ensure_numpy_array(value)
                for key, value in predictions.items()
            }

            median_key = 'q_50' if 'q_50' in normalized else next(iter(normalized))
            predicted_values = normalized[median_key]
            if hasattr(predicted_values, 'flatten'):
                predicted_values = predicted_values.flatten()
            else:
                predicted_values = np.asarray(predicted_values)

            return predicted_values, normalized

        predicted_array = self._ensure_numpy_array(predictions)
        if hasattr(predicted_array, 'flatten'):
            predicted_array = predicted_array.flatten()
        else:
            predicted_array = np.asarray(predicted_array)

        return predicted_array, None

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data for validation"""
        data_clean = data.copy()

        # Ensure datetime index
        if not isinstance(data_clean.index, pd.DatetimeIndex):
            if 'date' in data_clean.columns:
                data_clean['date'] = pd.to_datetime(data_clean['date'])
                data_clean.set_index('date', inplace=True)
            else:
                raise ValueError(
                    "Data must have datetime index or 'date' column")

        # Sort by date
        data_clean.sort_index(inplace=True)

        # Remove any future data leakage
        data_clean = data_clean[data_clean.index <= datetime.now()]

        # Basic cleaning
        data_clean = data_clean.dropna(subset=[self.config.target_column])

        return data_clean

    def _generate_validation_windows(
            self, data: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate training/validation window pairs"""
        windows = []

        # Determine date range
        # Ensure parsed dates are timezone-naive to match data index
        start_date = pd.to_datetime(self.config.start_date).tz_localize(None) if self.config.start_date else data.index[0]
        end_date = pd.to_datetime(self.config.end_date).tz_localize(None) if self.config.end_date else data.index[-1]

        # Find first valid training start
        first_valid_train_start = start_date
        first_valid_train_end = first_valid_train_start + \
            timedelta(days=self.config.min_training_days)

        # Start validation from first possible date
        current_val_start = first_valid_train_end + timedelta(days=1)

        while current_val_start < end_date:
            # Define validation window
            val_end = min(
                current_val_start +
                timedelta(
                    days=self.config.forecast_horizon -
                    1),
                end_date)

            # Define training window
            if self.config.max_training_days:
                # Rolling window
                train_start = max(
                    current_val_start -
                    timedelta(
                        days=self.config.max_training_days),
                    first_valid_train_start)
            else:
                # Expanding window
                train_start = first_valid_train_start

            train_end = current_val_start - timedelta(days=1)

            # Ensure minimum training period
            if (train_end - train_start).days >= self.config.min_training_days:
                # Check data availability
                train_data = data[train_start:train_end]
                val_data = data[current_val_start:val_end]

                if len(
                        train_data) >= self.config.min_training_days // 2 and len(val_data) >= 1:
                    windows.append(
                        (train_start, train_end, current_val_start, val_end))

            # Move to next validation period
            current_val_start += timedelta(
                days=self.config.validation_step_days)

        # Ensure minimum number of validation periods
        if len(windows) < self.config.min_validation_periods:
            logger.warning(
                f"Only {
                    len(windows)} validation windows generated, " f"minimum {
                    self.config.min_validation_periods} required")

        return windows

    def _validate_single_window(self,
                                data: pd.DataFrame,
                                train_start: datetime,
                                train_end: datetime,
                                val_start: datetime,
                                val_end: datetime,
                                model_factory: Callable,
                                feature_columns: List[str],
                                runtime_overrides: Optional[Dict[str, Any]] = None,
                                fold_idx: int = 0,
                                **model_kwargs) -> Dict[str, Any]:
        """Validate model on a single time window"""

        # Debug: Confirm fold_idx is being passed
        print(f"🔍 DEBUG: _validate_single_window called with fold_idx={fold_idx}", flush=True)
        logger.info(f"🔍 DEBUG: _validate_single_window called with fold_idx={fold_idx}")

        # Set per-fold random seed for model diversity
        from .ai_price_forecast import set_fold_seed
        fold_seed = set_fold_seed(base_seed=42, fold_idx=fold_idx)
        print(f"✅ Fold seed set to {fold_seed} for fold {fold_idx}", flush=True)

        runtime_overrides = runtime_overrides or {}

        # Split data
        train_data = data[train_start:train_end].copy()
        val_data = data[val_start:val_end].copy()

        # Create and train model
        model = model_factory(**model_kwargs)
        fit_kwargs, predict_kwargs = self._prepare_model_runtime(model, runtime_overrides)

        self._fit_model(model, train_data, feature_columns, fit_kwargs)

        predictions = self._predict_model(
            model,
            val_data,
            feature_columns,
            predict_kwargs
        )

        # Extract actual values
        actual_values = val_data[self.config.target_column].values

        # Handle different prediction formats
        predicted_values, quantile_forecasts = self._normalize_predictions(predictions)
        
        # Apply threshold-based trading logic if configured
        if hasattr(model, '_threshold_config') and model._threshold_config:
            try:
                from src.dcf_lab.decision_logic import apply_threshold_overrides
                
                # Try to get symbol/horizon from various sources
                symbol = None
                horizon = None
                
                # 1. Check model attributes
                symbol = getattr(model, 'symbol', None) or getattr(model, '_current_symbol', None)
                horizon = getattr(model, 'horizon', None) or getattr(model, '_current_horizon', None)
                
                # 2. Check validation config if available
                if not symbol and hasattr(self, 'config'):
                    symbol = getattr(self.config, 'symbol', None)
                if not horizon and hasattr(self, 'config'):
                    horizon = getattr(self.config, 'horizon', None)
                
                # 3. Use defaults as fallback
                symbol = symbol or 'DEFAULT'
                horizon = horizon or 63
                
                # Apply thresholds to convert probabilities to positions {+1, 0, -1}
                predicted_values = apply_threshold_overrides(
                    predicted_values,
                    symbol=str(symbol),
                    horizon=int(horizon),
                    threshold_config=model._threshold_config
                )
                logger.info(f"Applied thresholds for {symbol} h={horizon}")
            except Exception as e:
                logger.warning(f"Threshold application failed: {e}, using raw predictions")

        # Align lengths
        min_len = min(len(actual_values), len(predicted_values))
        actual_aligned = actual_values[:min_len]
        predicted_aligned = predicted_values[:min_len]

        # Calculate metrics
        metrics = self.performance_calc.calculate_metrics(
            actual_aligned, predicted_aligned, quantile_forecasts
        )

        # Add temporal info
        metrics.validation_period = f"{val_start.date()} to {val_end.date()}"

        return {
            'metrics': metrics,
            'train_period': (train_start, train_end),
            'val_period': (val_start, val_end),
            'predictions': predicted_aligned,
            'actual': actual_aligned,
            'quantile_forecasts': quantile_forecasts,
            'train_size': len(train_data),
            'val_size': len(val_data)
        }

    def _aggregate_results(
            self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all validation windows"""
        if not window_results:
            return {'error': 'No successful validation windows'}

        # Extract metrics from all windows
        all_metrics = [result['metrics'] for result in window_results]

        # Calculate aggregate statistics
        aggregate_metrics = ValidationMetrics()

        # Basic metrics
        aggregate_metrics.mae = np.mean([m.mae for m in all_metrics])
        aggregate_metrics.rmse = np.mean([m.rmse for m in all_metrics])
        aggregate_metrics.mape = np.mean([m.mape for m in all_metrics])
        aggregate_metrics.direction_accuracy = np.mean(
            [m.direction_accuracy for m in all_metrics])
        aggregate_metrics.hit_rate = np.mean([m.hit_rate for m in all_metrics])
        aggregate_metrics.precision = np.mean([m.precision for m in all_metrics])
        aggregate_metrics.recall = np.mean([m.recall for m in all_metrics])
        aggregate_metrics.f1_score = np.mean([m.f1_score for m in all_metrics])
        aggregate_metrics.sharpe_ratio = np.mean([m.sharpe_ratio for m in all_metrics])
        aggregate_metrics.sortino_ratio = np.mean([m.sortino_ratio for m in all_metrics])
        aggregate_metrics.max_drawdown = np.mean([m.max_drawdown for m in all_metrics])
        aggregate_metrics.calmar_ratio = np.mean([m.calmar_ratio for m in all_metrics])

        # Quantile metrics
        if all_metrics[0].pinball_losses:
            for q in self.config.quantiles:
                losses = [
                    m.pinball_losses.get(q, np.inf) for m in all_metrics
                    if q in m.pinball_losses]
                if losses:
                    aggregate_metrics.pinball_losses[q] = np.mean(losses)

        crps_values = [m.crps for m in all_metrics if not np.isinf(m.crps)]
        if crps_values:
            aggregate_metrics.crps = np.mean(crps_values)

        # Coverage metrics
        for level in [0.8, 0.6, 0.4]:
            coverages = [
                m.quantile_coverage.get(level, np.nan) for m in all_metrics
                if level in m.quantile_coverage]
            if coverages:
                aggregate_metrics.quantile_coverage[level] = np.mean(coverages)

        # Risk metrics
        aggregate_metrics.max_error = np.max(
            [m.max_error for m in all_metrics])
        aggregate_metrics.error_volatility = np.mean(
            [m.error_volatility for m in all_metrics])

        # Temporal stability
        mae_values = [m.mae for m in all_metrics]
        if len(mae_values) > 1:
            # Performance trend (slope of MAE over time)
            x = np.arange(len(mae_values))
            slope, _, _, _, _ = stats.linregress(x, mae_values)
            aggregate_metrics.performance_trend = slope

            # Performance stability (1 - coefficient of variation)
            cv = np.std(mae_values) / \
                np.mean(mae_values) if np.mean(mae_values) > 0 else 1
            aggregate_metrics.performance_stability = max(0, 1 - cv)

        # Sample info
        aggregate_metrics.n_predictions = sum(
            [m.n_predictions for m in all_metrics])

        # Build comprehensive results
        results = {
            'aggregate_metrics': aggregate_metrics,
            'individual_metrics': all_metrics,
            'summary_statistics': {
                'n_validation_windows': len(window_results),
                'total_predictions': aggregate_metrics.n_predictions,
                'validation_period': f"{window_results[0]['val_period'][0].date()} to {window_results[-1]['val_period'][1].date()}",
                'avg_training_size': np.mean([r['train_size'] for r in window_results]),
                'avg_validation_size': np.mean([r['val_size'] for r in window_results])
            },
            'performance_over_time': {
                'mae_trend': [m.mae for m in all_metrics],
                'direction_accuracy_trend': [m.direction_accuracy for m in all_metrics],
                'validation_dates': [r['val_period'][0] for r in window_results]
            },
            'window_details': window_results
        }

        return results

    def _save_results(
            self, aggregated_results: Dict[str, Any],
            window_results: List[Dict[str, Any]]):
        """Save validation results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save aggregate results as JSON
        json_results = self._prepare_json_results(aggregated_results)
        json_path = Path(self.config.results_dir) / \
            f"validation_results_{timestamp}.json"

        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        # Save detailed results as pickle
        pickle_path = Path(self.config.results_dir) / \
            f"validation_detailed_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({'aggregated': aggregated_results,
                        'windows': window_results}, f)

        logger.info(f"Results saved to {json_path} and {pickle_path}")

    def _prepare_json_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization"""
        json_results = {}

        # Extract aggregate metrics
        metrics = results['aggregate_metrics']
        json_results['aggregate_metrics'] = {
            'mae': float(metrics.mae),
            'rmse': float(metrics.rmse),
            'mape': float(metrics.mape),
            'direction_accuracy': float(metrics.direction_accuracy),
            'hit_rate': float(metrics.hit_rate),
            'precision': float(metrics.precision),
            'recall': float(metrics.recall),
            'f1_score': float(metrics.f1_score),
            'f1': float(metrics.f1_score),
                'sharpe_ratio': float(metrics.sharpe_ratio),
                'sortino_ratio': float(metrics.sortino_ratio),
                'max_drawdown': float(metrics.max_drawdown),
                'calmar_ratio': float(metrics.calmar_ratio),
            'crps': float(metrics.crps)
            if not np.isinf(metrics.crps) else None,
            'max_error': float(metrics.max_error),
            'error_volatility': float(metrics.error_volatility),
            'performance_trend': float(metrics.performance_trend),
            'performance_stability': float(metrics.performance_stability),
            'n_predictions': int(metrics.n_predictions)}

        # Add pinball losses
        if metrics.pinball_losses:
            json_results['aggregate_metrics']['pinball_losses'] = {
                str(k): float(v) for k, v in metrics.pinball_losses.items()
            }

        # Add coverage
        if metrics.quantile_coverage:
            json_results['aggregate_metrics']['quantile_coverage'] = {
                str(k): float(v) for k, v in metrics.quantile_coverage.items()
            }

    # Add summary statistics
        json_results['summary_statistics'] = results['summary_statistics']

        # Add performance trends
        json_results['performance_trends'] = {
            'mae_over_time': [
                float(x) for x in results['performance_over_time']['mae_trend']],
            'direction_accuracy_over_time': [
                float(x) for x in results['performance_over_time']['direction_accuracy_trend']],
            'validation_dates': [
                d.isoformat() for d in results['performance_over_time']['validation_dates']]}

        # Add diagnostics: provenance and telemetry (including Alpha Vantage rate/backoff if available)
        diagnostics: Dict[str, Any] = {}
        if self._extra_diagnostics.get('provenance') is not None:
            diagnostics['provenance'] = self._extra_diagnostics['provenance']
        # Start with caller-provided telemetry and augment with rate limiter stats if present
        tel: Dict[str, Any] = {}
        if isinstance(self._extra_diagnostics.get('telemetry'), dict):
            tel.update(self._extra_diagnostics['telemetry'])
        try:
            # Avoid hard dependency if utils not present
            from .utils import rate as rate_utils  # type: ignore
            if hasattr(rate_utils, 'get_stats'):
                av_stats = rate_utils.get_stats()
                # nest under alpha_vantage key to avoid collisions
                tel.setdefault('alpha_vantage_rate', av_stats)
        except Exception:
            pass
        if tel:
            diagnostics['telemetry'] = tel
        if diagnostics:
            json_results['diagnostics'] = diagnostics

        return json_results

    def _generate_plots(self, window_results: List[Dict[str, Any]]):
        """Generate validation plots"""
        if not window_results:
            return

        # Set up the plotting style
        plt.style.use('default')
        _, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data for plotting
        val_dates = [r['val_period'][0] for r in window_results]
        mae_values = [r['metrics'].mae for r in window_results]
        direction_acc = [
            r['metrics'].direction_accuracy for r in window_results]

        # Plot 1: MAE over time
        axes[0, 0].plot(val_dates, mae_values, 'b-o', markersize=4)
        axes[0, 0].set_title('Mean Absolute Error Over Time')
        axes[0, 0].set_xlabel('Validation Date')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Direction accuracy over time
        axes[0, 1].plot(val_dates, direction_acc, 'g-o', markersize=4)
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--',
                           alpha=0.7, label='Random Baseline')
        axes[0, 1].set_title('Directional Accuracy Over Time')
        axes[0, 1].set_xlabel('Validation Date')
        axes[0, 1].set_ylabel('Direction Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Error distribution
        all_errors = []
        for result in window_results:
            if 'actual' in result and 'predictions' in result:
                errors = result['actual'] - result['predictions']
                all_errors.extend(errors)

        if all_errors:
            axes[1, 0].hist(all_errors, bins=30, alpha=0.7,
                            color='skyblue', edgecolor='black')
            axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Forecast Error Distribution')
            axes[1, 0].set_xlabel('Forecast Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Performance metrics summary
        metrics_names = ['MAE', 'Direction Acc', 'Hit Rate']
        aggregate_values = [
            np.mean(mae_values),
            np.mean(direction_acc),
            np.mean([r['metrics'].hit_rate for r in window_results])
        ]

        bars = axes[1, 1].bar(metrics_names, aggregate_values, color=[
                              'blue', 'green', 'orange'])
        axes[1, 1].set_title('Average Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, aggregate_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(self.config.results_dir) / \
            f"validation_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Validation plots saved to {plot_path}")


def create_simple_model_factory():
    """Create a simple model factory for testing purposes"""
    from .ensemble_forecaster import EnsembleForecaster

    def model_factory(**kwargs):
        return EnsembleForecaster(
            forecast_horizon=kwargs.get('forecast_horizon', 5),
            quantiles=kwargs.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
        )

    return model_factory


def run_validation_example():
    """Example of how to use the walk-forward validation system"""

    # Create configuration
    config = ValidationConfig(
        min_training_days=252,
        validation_step_days=21,
        forecast_horizon=5,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        save_results=True,
        generate_plots=True
    )

    # Create validator
    validator = WalkForwardValidator(config)

    # Generate sample data for testing
    rng = np.random.default_rng(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)

    # Generate synthetic financial data
    returns = rng.normal(0.0002, 0.01, n_days)  # Daily returns
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]

    # Create features
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'log_return_1d': np.log(np.array(prices[1:]) / np.array(prices[:-1])),
        'volatility_20d': rng.normal(0.15, 0.05, n_days),
        'momentum_5d': rng.normal(0, 0.01, n_days),
        'volume': rng.exponential(1000000, n_days)
    }).set_index('date')

    # Create 5-day forward returns
    data['log_return_5d'] = data['log_return_1d'].rolling(5).sum().shift(-5)
    data = data.dropna()

    # Define feature columns
    feature_columns = ['volatility_20d', 'momentum_5d', 'volume']

    # Create model factory
    model_factory = create_simple_model_factory()

    # Run validation
    results = validator.validate_model(
        data=data,
        model_factory=model_factory,
        feature_columns=feature_columns,
        forecast_horizon=5,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )

    print("\n=== WALK-FORWARD VALIDATION RESULTS ===")
    print(
        f"Number of validation windows: {
            results['summary_statistics']['n_validation_windows']}")
    print(
        f"Total predictions: {
            results['summary_statistics']['total_predictions']}")
    print(
        f"Validation period: {
            results['summary_statistics']['validation_period']}")

    metrics = results['aggregate_metrics']
    print("\nAggregate Performance:")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"RMSE: {metrics.rmse:.6f}")
    print(f"MAPE: {metrics.mape:.2f}%")
    print(f"Direction Accuracy: {metrics.direction_accuracy:.3f}")
    print(f"Hit Rate: {metrics.hit_rate:.3f}")

    if not np.isinf(metrics.crps):
        print(f"CRPS: {metrics.crps:.6f}")

    print(f"Performance Stability: {metrics.performance_stability:.3f}")

    return results


if __name__ == "__main__":
    run_validation_example()
