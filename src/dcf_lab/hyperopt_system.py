"""
Hyperparameter Optimization System for AI Price Forecasting

This module implements systematic hyperparameter tuning using Optuna and Bayesian optimization
to optimize ensemble weights, model parameters, forecasting horizons, and other configuration
parameters for maximum forecasting performance.

Key Features:
- Optuna-based Bayesian optimization
- Multi-objective optimization (accuracy, stability, robustness)
- Automated parameter space definition
- Cross-validation integration
- Ensemble weight optimization
- Model architecture tuning
- Performance-based early stopping
- Hyperparameter persistence and caching

Author: AI Assistant
Created: 2024
"""

import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn(
        "optuna not available - hyperparameter optimization disabled")

# Statistical and ML imports
try:
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - some optimization methods disabled")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available - cross-validation limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HyperoptConfig:
    """Configuration for hyperparameter optimization"""

    # Optimization strategy
    optimization_method: str = 'optuna'  # 'optuna', 'bayesian', 'grid', 'random'
    # 'minimize' for error, 'maximize' for accuracy
    optimization_direction: str = 'minimize'
    n_trials: int = 100
    n_jobs: int = 1  # Parallel trials (if supported)
    timeout_seconds: Optional[int] = 3600  # 1 hour timeout

    # Objective function settings
    primary_metric: str = 'mae'  # 'mae', 'rmse', 'mape', 'direction_accuracy'
    include_stability_penalty: bool = True
    stability_weight: float = 0.1  # Weight for stability in multi-objective
    include_robustness_penalty: bool = True
    robustness_weight: float = 0.05  # Weight for robustness

    # Cross-validation settings
    cv_method: str = 'time_series'  # 'time_series', 'blocked', 'purged'
    cv_folds: int = 5
    cv_test_size: int = 50  # Days for each test period
    cv_gap: int = 5  # Gap between train/test to avoid look-ahead
    min_train_size: int = 252  # Minimum training days

    # Parameter space settings
    optimize_ensemble_weights: bool = True
    optimize_forecast_horizon: bool = True
    optimize_quantiles: bool = True
    optimize_model_params: bool = True
    optimize_feature_selection: bool = False

    # Ensemble weight optimization
    ensemble_weight_min: float = 0.0
    ensemble_weight_max: float = 1.0
    ensemble_normalize_weights: bool = True

    # Horizon optimization
    horizon_min: int = 1
    horizon_max: int = 30
    horizon_step: int = 1

    # Quantile optimization
    quantile_levels: List[float] = field(
        default_factory=lambda: [
            0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    optimize_quantile_selection: bool = True
    min_quantiles: int = 3
    max_quantiles: int = 7

    # Model-specific parameters
    lstm_hidden_sizes: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256])
    lstm_learning_rates: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.1])
    lstm_dropout_rates: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5])

    rf_n_estimators: List[int] = field(
        default_factory=lambda: [
            50, 100, 200, 500])
    rf_max_depths: List[int] = field(default_factory=lambda: [5, 10, 15, None])
    rf_min_samples_splits: List[int] = field(
        default_factory=lambda: [2, 5, 10])

    # Early stopping and pruning
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10  # Trials without improvement
    early_stopping_threshold: float = 0.001  # Minimum improvement threshold
    enable_pruning: bool = True
    pruning_warmup_steps: int = 5  # Steps before pruning can start

    # Caching and persistence
    cache_results: bool = True
    cache_directory: str = "hyperopt_cache"
    save_best_params: bool = True
    load_previous_study: bool = True
    study_name: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization"""

    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    n_trials: int
    optimization_time: float
    cv_scores: List[float]
    stability_score: float
    robustness_score: float

    # Performance breakdown
    train_score: float
    validation_score: float
    test_score: Optional[float] = None

    # Parameter importance (if available)
    param_importance: Optional[Dict[str, float]] = None

    # Cross-validation details
    cv_details: Optional[Dict[str, Any]] = None


class ObjectiveFunction(ABC):
    """Abstract base class for optimization objective functions"""

    @abstractmethod
    def __call__(self, params: Dict[str, Any], data: pd.DataFrame,
                 config: HyperoptConfig) -> float:
        """Evaluate objective function with given parameters"""

    @abstractmethod
    def get_cv_score(self, params: Dict[str, Any], data: pd.DataFrame,
                     config: HyperoptConfig) -> Tuple[float, List[float]]:
        """Get cross-validation score and individual fold scores"""


class ForecastingObjective(ObjectiveFunction):
    """Objective function for forecasting model optimization"""

    def __init__(self, model_factory: Callable, feature_columns: List[str]):
        self.model_factory = model_factory
        self.feature_columns = feature_columns

    def __call__(self, params: Dict[str, Any], data: pd.DataFrame,
                 config: HyperoptConfig) -> float:
        """Evaluate forecasting performance with given parameters"""

        # Get cross-validation score
        cv_score, fold_scores = self.get_cv_score(params, data, config)

        # Add stability penalty if enabled
        if config.include_stability_penalty and len(fold_scores) > 1:
            stability_penalty = np.std(fold_scores) * config.stability_weight
            cv_score += stability_penalty

        # Add robustness penalty if enabled
        if config.include_robustness_penalty:
            robustness_penalty = self._compute_robustness_penalty(
                params, data, config
            )
            cv_score += robustness_penalty * config.robustness_weight

        return cv_score

    def _prepare_cv_data(self,
                         params: Dict[str,
                                      Any],
                         data: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                      pd.Series]:
        """Prepare features and target for cross-validation."""
        target_col = 'log_return_5d'  # Default target
        if 'forecast_horizon' in params:
            target_col = f'log_return_{params["forecast_horizon"]}d'

        if target_col not in data.columns:
            # Fallback to first available target
            target_cols = [col for col in data.columns if 'log_return' in col]
            if target_cols:
                target_col = target_cols[0]
            else:
                raise ValueError("No target variable found in data")

        # Prepare features and target
        X = data[self.feature_columns].dropna()
        y = data[target_col].loc[X.index]

        # Remove any remaining NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]

        return X, y

    def _create_cv_splitter(self, config: HyperoptConfig):
        """Create cross-validation splitter based on configuration."""
        if config.cv_method == 'time_series':
            return TimeSeriesSplit(
                n_splits=config.cv_folds,
                test_size=config.cv_test_size,
                gap=config.cv_gap
            )
        else:
            return TimeSeriesSplit(n_splits=config.cv_folds)

    def _train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series):
        """Train model with appropriate method."""
        if hasattr(model, 'fit'):
            if hasattr(model, 'partial_fit'):
                # Online learning model
                for i in range(len(X_train)):
                    model.partial_fit(
                        X_train.iloc[i:i+1].values,
                        [y_train.iloc[i]]
                    )
            else:
                # Batch learning model
                model.fit(X_train.values, y_train.values)
        else:
            # Custom model interface
            model.train(X_train.values, y_train.values)

    def _make_predictions(self, model, X_test: pd.DataFrame):
        """Make predictions using the trained model."""
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test.values)
        else:
            y_pred = model.forecast(X_test.values)

        # Handle quantile predictions
        if isinstance(y_pred, dict):
            # Use median prediction for scoring
            y_pred = y_pred.get(0.5, list(y_pred.values())[0])
        elif hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
            # Multi-output, use first output
            y_pred = y_pred[:, 0]

        return y_pred

    def _calculate_fold_score(
            self,
            y_test: pd.Series,
            y_pred,
            config: HyperoptConfig) -> float:
        """Calculate score for a single fold."""
        if config.primary_metric == 'mae':
            return mean_absolute_error(y_test, y_pred)
        elif config.primary_metric == 'rmse':
            return np.sqrt(mean_squared_error(y_test, y_pred))
        elif config.primary_metric == 'mape':
            return np.mean(np.abs((y_test - y_pred) /
                           np.maximum(np.abs(y_test), 1e-8))) * 100
        elif config.primary_metric == 'direction_accuracy':
            if len(y_test) > 1:
                y_test_dir = np.diff(y_test) > 0
                y_pred_dir = np.diff(y_pred) > 0
                # 1 - accuracy for minimization
                return 1 - np.mean(y_test_dir == y_pred_dir)
            else:
                return 0.5  # Random baseline
        else:
            return mean_absolute_error(y_test, y_pred)

    def get_cv_score(self, params: Dict[str, Any], data: pd.DataFrame,
                     config: HyperoptConfig) -> Tuple[float, List[float]]:
        """Get cross-validation score using time series split"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for cross-validation")

        # Prepare data and validate
        X, y = self._prepare_cv_data(params, data)
        if not self._validate_cv_data_size(X, config):
            return float('inf'), []

        # Execute cross-validation
        fold_scores = self._execute_cv_folds(X, y, params, config)
        
        # Calculate final score
        return self._calculate_final_cv_score(fold_scores)

    def _validate_cv_data_size(self, X, config):
        """Validate that data is sufficient for cross-validation"""
        min_required = config.min_train_size + config.cv_test_size
        return len(X) >= min_required

    def _execute_cv_folds(self, X, y, params, config):
        """Execute all cross-validation folds"""
        cv = self._create_cv_splitter(config)
        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            score = self._evaluate_single_fold(
                X, y, params, config, train_idx, test_idx, fold)
            if score is not None:
                fold_scores.append(score)

        return fold_scores

    def _evaluate_single_fold(self, X, y, params, config, 
                             train_idx, test_idx, fold):
        """Evaluate a single cross-validation fold"""
        try:
            # Ensure minimum training size
            if len(train_idx) < config.min_train_size:
                return None

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train and evaluate model
            model = self.model_factory(**params)
            self._train_model(model, X_train, y_train)
            y_pred = self._make_predictions(model, X_test)
            
            return self._calculate_fold_score(y_test, y_pred, config)

        except Exception as e:
            logger.warning(f"CV fold {fold} failed: {e}")
            return float('inf')

    def _calculate_final_cv_score(self, fold_scores):
        """Calculate final cross-validation score from fold results"""
        if not fold_scores:
            return float('inf'), []

        # Remove infinite scores
        valid_scores = [s for s in fold_scores if not np.isinf(s)]
        if not valid_scores:
            return float('inf'), fold_scores

        return np.mean(valid_scores), fold_scores

    def _compute_robustness_penalty(self, params: Dict[str, Any],
                                    data: pd.DataFrame = None,
                                    config: HyperoptConfig = None) -> float:
        """Compute robustness penalty based on parameter sensitivity

        Args:
            params: Parameters to evaluate
            data: Training data (reserved for future use)
            config: Configuration (reserved for future use)
        """
        # Note: data and config parameters reserved for future robustness
        # metrics

        # Simple robustness measure: preference for simpler models
        penalty = 0.0

        # Penalize complex ensemble weights (prefer uniform)
        if 'ensemble_weights' in params:
            weights = np.array(list(params['ensemble_weights'].values()))
            uniform_weights = np.ones_like(weights) / len(weights)
            penalty += np.sum(np.abs(weights - uniform_weights))

        # Penalize extreme horizons
        if 'forecast_horizon' in params:
            horizon = params['forecast_horizon']
            optimal_horizon = 5  # Assume 5-day is optimal
            penalty += abs(horizon - optimal_horizon) * 0.01

        # Penalize too many quantiles
        if 'quantiles' in params:
            n_quantiles = len(params['quantiles'])
            if n_quantiles > 5:
                penalty += (n_quantiles - 5) * 0.1

        return penalty


class HyperparameterOptimizer:
    """Main hyperparameter optimization system"""

    def __init__(self, config: HyperoptConfig):
        self.config = config
        self.study = None
        self.optimization_history = []
        self.best_result = None

        # Setup cache directory
        if config.cache_results:
            os.makedirs(config.cache_directory, exist_ok=True)

        logger.info(
            "Initialized hyperparameter optimizer with %s",
            config.optimization_method,
        )

    def optimize(self,
                 objective: ObjectiveFunction,
                 data: pd.DataFrame,
                 parameter_space: Optional[Dict[str,
                                                Any]] = None) -> OptimizationResult:
        """Run hyperparameter optimization"""

        if parameter_space is None:
            parameter_space = self._create_default_parameter_space()

        start_time = datetime.now()

        if self.config.optimization_method == 'optuna':
            result = self._optimize_with_optuna(
                objective, data, parameter_space)
        elif self.config.optimization_method == 'grid':
            result = self._optimize_with_grid_search(
                objective, data, parameter_space)
        elif self.config.optimization_method == 'random':
            result = self._optimize_with_random_search(
                objective, data, parameter_space)
        elif self.config.optimization_method == 'bayesian':
            result = self._optimize_with_bayesian(
                objective, data, parameter_space)
        else:
            raise ValueError(
                f"Unknown optimization method: {self.config.optimization_method}"
            )

        optimization_time = (datetime.now() - start_time).total_seconds()
        result.optimization_time = optimization_time

        self.best_result = result

        # Save results if configured
        if self.config.save_best_params:
            self._save_optimization_result(result)

        logger.info(f"Optimization completed in {optimization_time:.2f}s. "
                    f"Best score: {result.best_score:.6f}")

        return result

    def _create_or_load_study(self, study_name: str):
        """Create or load Optuna study based on configuration."""
        if self.config.load_previous_study:
            try:
                # Try to load existing study
                storage = f"sqlite:///{self.config.cache_directory}/optuna_studies.db"
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage
                )
                logger.info(
                    f"Loaded existing study with {len(study.trials)} trials")
                return study
            except Exception:
                # Create new study if loading fails
                pass

        # Create new study
        return optuna.create_study(
            direction=self.config.optimization_direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner() if self.config.enable_pruning else None,
            study_name=study_name,
            storage=(
                f"sqlite:///{self.config.cache_directory}/optuna_studies.db"
                if self.config.cache_results
                else None
            )
        )

    def _sample_parameter(self, trial, param_name: str, param_config: Dict):
        """Sample a single parameter using Optuna trial."""
        if param_config['type'] == 'float':
            return trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high'],
                log=param_config.get('log', False)
            )
        elif param_config['type'] == 'int':
            return trial.suggest_int(
                param_name,
                param_config['low'],
                param_config['high'],
                step=param_config.get('step', 1)
            )
        elif param_config['type'] == 'categorical':
            return trial.suggest_categorical(
                param_name,
                param_config['choices']
            )
        elif param_config['type'] == 'discrete_uniform':
            return trial.suggest_discrete_uniform(
                param_name,
                param_config['low'],
                param_config['high'],
                param_config['q']
            )

    def _sample_ensemble_weights(self, trial, parameter_space: Dict):
        """Sample ensemble weights and normalize if configured."""
        if not (
                self.config.optimize_ensemble_weights and 'ensemble_models' in parameter_space):
            return {}

        ensemble_weights = {}
        for model_name in parameter_space['ensemble_models']['choices']:
            weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
            ensemble_weights[model_name] = weight

        # Normalize weights if configured
        if self.config.ensemble_normalize_weights:
            total_weight = sum(ensemble_weights.values())
            if total_weight > 0:
                ensemble_weights = {
                    k: v/total_weight for k,
                    v in ensemble_weights.items()}

        return ensemble_weights

    def _create_optuna_objective(
            self,
            objective: ObjectiveFunction,
            data: pd.DataFrame,
            parameter_space: Dict):
        """Create objective function for Optuna optimization."""
        def optuna_objective(trial):
            params = {}

            # Sample parameters from space
            for param_name, param_config in parameter_space.items():
                if param_name != 'ensemble_models':  # Handle ensemble models separately
                    params[param_name] = self._sample_parameter(
                        trial, param_name, param_config)

            # Handle ensemble weights
            ensemble_weights = self._sample_ensemble_weights(
                trial, parameter_space)
            if ensemble_weights:
                params['ensemble_weights'] = ensemble_weights

            # Evaluate objective
            try:
                score = objective(params, data, self.config)

                # Report for pruning if enabled
                if self.config.enable_pruning and hasattr(trial, 'report'):
                    trial.report(score, step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')

        return optuna_objective

    def _optimize_with_optuna(
            self, objective: ObjectiveFunction, data: pd.DataFrame,
            parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize using Optuna"""

        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna required for Bayesian optimization")

        # Create or load study
        study_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = self.config.study_name or f"forecasting_study_{study_suffix}"
        self.study = self._create_or_load_study(study_name)

        # Create objective function

        # Run optimization
        optuna_objective_func = self._create_optuna_objective(
            objective, data, parameter_space)
        self.study.optimize(
            optuna_objective_func,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs if self.config.n_jobs > 1 else None
        )

        # Get best results
        best_params = self.study.best_params
        best_score = self.study.best_value

        # Get cross-validation details
        cv_score, cv_scores = objective.get_cv_score(
            best_params, data, self.config)

        # Calculate stability and robustness
        stability_score = np.std(cv_scores) if len(cv_scores) > 1 else 0.0
        robustness_score = objective._compute_robustness_penalty(
            best_params, data, self.config) if hasattr(
            objective, '_compute_robustness_penalty') else 0.0

        # Get parameter importance
        param_importance = None
        try:
            param_importance = optuna.importance.get_param_importances(
                self.study)
        except Exception:
            pass

        # Create optimization history
        optimization_history = []
        for trial in self.study.trials:
            optimization_history.append(
                {'trial_number': trial.number, 'params': trial.params,
                 'value': trial.value, 'state': trial.state.name,
                 'duration': trial.duration.total_seconds()
                 if trial.duration else None})

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            n_trials=len(self.study.trials),
            optimization_time=0.0,  # Will be set by caller
            cv_scores=cv_scores,
            stability_score=stability_score,
            robustness_score=robustness_score,
            train_score=cv_score,
            validation_score=best_score,
            param_importance=param_importance,
            cv_details={
                'cv_method': self.config.cv_method,
                'cv_folds': self.config.cv_folds}
        )

    def _convert_param_to_grid_format(self, param_config):
        """Convert parameter configuration to grid search format"""
        if param_config['type'] in ['float', 'int']:
            if 'choices' in param_config:
                return param_config['choices']
            else:
                # Create discrete values
                low, high = param_config['low'], param_config['high']
                if param_config['type'] == 'int':
                    return list(
                        range(
                            low,
                            high + 1,
                            param_config.get(
                                'step',
                                1)))
                else:
                    n_values = min(
                        10, self.config.n_trials // 10)  # Limit grid size
                    return np.linspace(low, high, n_values).tolist()
        elif param_config['type'] == 'categorical':
            return param_config['choices']

    def _prepare_grid_combinations(self, parameter_space):
        """Prepare parameter combinations for grid search"""
        param_grid = {}
        for param_name, param_config in parameter_space.items():
            param_grid[param_name] = self._convert_param_to_grid_format(
                param_config)

        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        # Limit to n_trials
        if len(param_combinations) > self.config.n_trials:
            rng = np.random.default_rng(42)
            rng.shuffle(param_combinations)
            param_combinations = param_combinations[:self.config.n_trials]

        return param_combinations

    def _evaluate_grid_combination(
            self,
            params,
            objective,
            data,
            trial_number):
        """Evaluate a single parameter combination"""
        try:
            score = objective(params, data, self.config)
            return {
                'trial_number': trial_number,
                'params': params,
                'value': score,
                'state': 'COMPLETE'
            }
        except Exception as e:
            logger.warning(f"Grid search trial {trial_number} failed: {e}")
            return {
                'trial_number': trial_number,
                'params': params,
                'value': float('in'),
                'state': 'FAILED'
            }

    def _optimize_with_grid_search(
            self, objective: ObjectiveFunction, data: pd.DataFrame,
            parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize using grid search"""

        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for grid search")

        # Prepare parameter combinations
        param_combinations = self._prepare_grid_combinations(parameter_space)

        best_score = float('in')
        best_params = None
        optimization_history = []

        for i, params in enumerate(param_combinations):
            result = self._evaluate_grid_combination(
                params, objective, data, i)
            optimization_history.append(result)

            if result['value'] < best_score:
                best_score = result['value']
                best_params = params

        # Get cross-validation details for best params
        cv_score, cv_scores = objective.get_cv_score(
            best_params, data, self.config)
        stability_score = np.std(cv_scores) if len(cv_scores) > 1 else 0.0

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            n_trials=len(param_combinations),
            optimization_time=0.0,
            cv_scores=cv_scores,
            stability_score=stability_score,
            robustness_score=0.0,
            train_score=cv_score,
            validation_score=best_score
        )

    def _sample_random_param(self, param_config, rng):
        """Sample a random parameter value"""
        if param_config['type'] == 'float':
            if param_config.get('log', False):
                log_value = rng.uniform(
                    np.log(param_config['low']),
                    np.log(param_config['high'])
                )
                return np.exp(log_value)
            else:
                return rng.uniform(param_config['low'], param_config['high'])
        elif param_config['type'] == 'int':
            return rng.integers(param_config['low'], param_config['high'] + 1)
        elif param_config['type'] == 'categorical':
            return rng.choice(param_config['choices'])

    def _sample_random_params(self, parameter_space, rng):
        """Sample random parameters for all parameters"""
        params = {}
        for param_name, param_config in parameter_space.items():
            params[param_name] = self._sample_random_param(param_config, rng)
        return params

    def _evaluate_random_trial(self, params, objective, data, trial_number):
        """Evaluate a single random trial"""
        try:
            score = objective(params, data, self.config)
            return {
                'trial_number': trial_number,
                'params': params,
                'value': score,
                'state': 'COMPLETE'
            }
        except Exception as e:
            logger.warning(f"Random search trial {trial_number} failed: {e}")
            return {
                'trial_number': trial_number,
                'params': params,
                'value': float('in'),
                'state': 'FAILED'
            }

    def _optimize_with_random_search(
            self, objective: ObjectiveFunction, data: pd.DataFrame,
            parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize using random search"""

        # Initialize random number generator
        rng = np.random.default_rng(42)

        best_score = float('in')
        best_params = None
        optimization_history = []

        for i in range(self.config.n_trials):
            # Sample random parameters
            params = self._sample_random_params(parameter_space, rng)

            # Evaluate trial
            result = self._evaluate_random_trial(params, objective, data, i)
            optimization_history.append(result)

            if result['value'] < best_score:
                best_score = result['value']
                best_params = params

        # Get cross-validation details for best params
        cv_score, cv_scores = objective.get_cv_score(
            best_params, data, self.config)
        stability_score = np.std(cv_scores) if len(cv_scores) > 1 else 0.0

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            n_trials=self.config.n_trials,
            optimization_time=0.0,
            cv_scores=cv_scores,
            stability_score=stability_score,
            robustness_score=0.0,
            train_score=cv_score,
            validation_score=best_score
        )

    def _get_random_baseline(self, objective, data, parameter_space):
        """Get baseline using random search"""
        n_random = min(10, self.config.n_trials // 2)
        random_config = HyperoptConfig()
        random_config.n_trials = n_random
        random_config.optimization_method = 'random'

        random_optimizer = HyperparameterOptimizer(random_config)
        return random_optimizer._optimize_with_random_search(
            objective, data, parameter_space)

    def _perturb_float_param(self, current_value, param_config, rng):
        """Perturb a float parameter"""
        noise_scale = (param_config['high'] - param_config['low']) * 0.1
        perturbed_value = current_value + rng.normal(0, noise_scale)
        return np.clip(
            perturbed_value,
            param_config['low'],
            param_config['high'])

    def _perturb_int_param(self, current_value, param_config, rng):
        """Perturb an integer parameter"""
        perturbation = rng.integers(-2, 3)
        perturbed_value = current_value + perturbation
        return np.clip(
            perturbed_value,
            param_config['low'],
            param_config['high'])

    def _perturb_parameters(self, best_params, parameter_space, rng):
        """Perturb parameters around best known values"""
        perturbed_params = {}
        for param_name, param_config in parameter_space.items():
            if param_name in best_params:
                current_value = best_params[param_name]

                if param_config['type'] == 'float':
                    perturbed_params[param_name] = self._perturb_float_param(
                        current_value, param_config, rng)
                elif param_config['type'] == 'int':
                    perturbed_params[param_name] = self._perturb_int_param(
                        current_value, param_config, rng)
                else:
                    perturbed_params[param_name] = current_value
            else:
                # Sample randomly for new parameters
                if param_config['type'] == 'categorical':
                    perturbed_params[param_name] = rng.choice(
                        param_config['choices'])
                # Add other types as needed
        return perturbed_params

    def _evaluate_bayesian_trial(
            self,
            perturbed_params,
            objective,
            data,
            trial_number):
        """Evaluate a single Bayesian optimization trial"""
        try:
            score = objective(perturbed_params, data, self.config)
            return {
                'trial_number': trial_number,
                'params': perturbed_params,
                'value': score,
                'state': 'COMPLETE'
            }
        except Exception as e:
            logger.warning(
                f"Bayesian optimization trial {trial_number} failed: {e}")
            return None

    def _optimize_with_bayesian(
            self, objective: ObjectiveFunction, data: pd.DataFrame,
            parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize using Bayesian optimization (fallback to random if scipy unavailable)"""

        if not SCIPY_AVAILABLE:
            logger.warning(
                "scipy not available, falling back to random search")
            return self._optimize_with_random_search(
                objective, data, parameter_space)

        # Initialize random number generator
        rng = np.random.default_rng(42)

        # Get random baseline
        random_result = self._get_random_baseline(
            objective, data, parameter_space)

        # Use best random result as starting point
        best_params = random_result.best_params
        best_score = random_result.best_score
        optimization_history = random_result.optimization_history

        # Continue with local optimization around best point
        n_random = min(10, self.config.n_trials // 2)
        remaining_trials = self.config.n_trials - n_random

        for i in range(remaining_trials):
            # Perturb best parameters
            perturbed_params = self._perturb_parameters(
                best_params, parameter_space, rng)

            # Evaluate trial
            result = self._evaluate_bayesian_trial(
                perturbed_params, objective, data, n_random + i)
            if result:
                optimization_history.append(result)

                if result['value'] < best_score:
                    best_score = result['value']
                    best_params = perturbed_params

        # Get cross-validation details for best params
        cv_score, cv_scores = objective.get_cv_score(
            best_params, data, self.config)
        stability_score = np.std(cv_scores) if len(cv_scores) > 1 else 0.0

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            n_trials=self.config.n_trials,
            optimization_time=0.0,
            cv_scores=cv_scores,
            stability_score=stability_score,
            robustness_score=0.0,
            train_score=cv_score,
            validation_score=best_score
        )

    def _create_default_parameter_space(self) -> Dict[str, Any]:
        """Create default parameter space for optimization"""

        space = {}

        # Forecast horizon
        if self.config.optimize_forecast_horizon:
            space['forecast_horizon'] = {
                'type': 'int',
                'low': self.config.horizon_min,
                'high': self.config.horizon_max,
                'step': self.config.horizon_step
            }

        # Quantile selection
        if self.config.optimize_quantiles:
            # For simplicity, optimize number of quantiles
            space['n_quantiles'] = {
                'type': 'int',
                'low': self.config.min_quantiles,
                'high': self.config.max_quantiles
            }

        # Model-specific parameters
        if self.config.optimize_model_params:
            # LSTM parameters
            space['lstm_hidden_size'] = {
                'type': 'categorical',
                'choices': self.config.lstm_hidden_sizes
            }
            space['lstm_learning_rate'] = {
                'type': 'categorical',
                'choices': self.config.lstm_learning_rates
            }
            space['lstm_dropout_rate'] = {
                'type': 'categorical',
                'choices': self.config.lstm_dropout_rates
            }

            # Random Forest parameters
            space['rf_n_estimators'] = {
                'type': 'categorical',
                'choices': self.config.rf_n_estimators
            }
            space['rf_max_depth'] = {
                'type': 'categorical',
                'choices': self.config.rf_max_depths
            }

        # Ensemble models (for weight optimization)
        if self.config.optimize_ensemble_weights:
            space['ensemble_models'] = {
                'type': 'categorical',
                'choices': [
                    'lstm',
                    'random_forest',
                    'linear_ar',
                    'garch',
                    'transformer']}

        return space

    def _save_optimization_result(self, result: OptimizationResult):
        """Save optimization results to cache"""

        if not self.config.cache_results:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(
            self.config.cache_directory,
            f'optimization_result_{timestamp}.json'
        )

        # Convert result to serializable format
        result_dict = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'n_trials': result.n_trials,
            'optimization_time': result.optimization_time,
            'cv_scores': result.cv_scores,
            'stability_score': result.stability_score,
            'robustness_score': result.robustness_score,
            'train_score': result.train_score,
            'validation_score': result.validation_score,
            'param_importance': result.param_importance,
            'config': {
                'optimization_method': self.config.optimization_method,
                'primary_metric': self.config.primary_metric,
                'n_trials': self.config.n_trials,
                'cv_method': self.config.cv_method,
                'cv_folds': self.config.cv_folds
            },
            'timestamp': timestamp
        }

        try:
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            logger.info(f"Saved optimization results to {result_file}")
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")

    def load_optimization_result(
            self, filepath: str) -> Optional[OptimizationResult]:
        """Load optimization results from file"""

        try:
            with open(filepath, 'r') as f:
                result_dict = json.load(f)

            result = OptimizationResult(
                best_params=result_dict['best_params'],
                best_score=result_dict['best_score'],
                optimization_history=[],  # Not saved in simple format
                n_trials=result_dict['n_trials'],
                optimization_time=result_dict['optimization_time'],
                cv_scores=result_dict['cv_scores'],
                stability_score=result_dict['stability_score'],
                robustness_score=result_dict['robustness_score'],
                train_score=result_dict['train_score'],
                validation_score=result_dict['validation_score'],
                param_importance=result_dict.get('param_importance')
            )

            logger.info(f"Loaded optimization results from {filepath}")
            return result

        except Exception as e:
            logger.error(f"Failed to load optimization results: {e}")
            return None


# Factory functions and utilities
def create_hyperopt_system(
    optimization_method: str = 'optuna',
    n_trials: int = 100,
    primary_metric: str = 'mae',
    **kwargs
) -> HyperparameterOptimizer:
    """Factory function to create hyperparameter optimization system"""

    config = HyperoptConfig(
        optimization_method=optimization_method,
        n_trials=n_trials,
        primary_metric=primary_metric,
        **kwargs
    )

    return HyperparameterOptimizer(config)


def optimize_forecasting_model(
    model_factory: Callable,
    data: pd.DataFrame,
    feature_columns: List[str],
    config: Optional[HyperoptConfig] = None,
    parameter_space: Optional[Dict[str, Any]] = None
) -> OptimizationResult:
    """Convenient function to optimize a forecasting model"""

    if config is None:
        config = HyperoptConfig()

    optimizer = HyperparameterOptimizer(config)
    objective = ForecastingObjective(model_factory, feature_columns)

    return optimizer.optimize(objective, data, parameter_space)


if __name__ == "__main__":
    # Example usage
    print("🎯 Hyperparameter Optimization System for AI Price Forecasting")
    print("=" * 70)

    # Create sample data
    rng = np.random.default_rng(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Create synthetic financial data
    returns = rng.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': rng.exponential(1000000, n_samples),
        'volatility': np.abs(rng.normal(0.02, 0.01, n_samples)),
        'momentum': rng.normal(0, 0.1, n_samples),
        'rsi': rng.uniform(20, 80, n_samples)
    })

    # Add target variable
    data['log_return_5d'] = np.log(data['close'] / data['close'].shift(5))
    data = data.dropna()

    feature_columns = ['volatility', 'momentum', 'rsi']

    print(
        f"Created synthetic dataset: {len(data)} samples, {len(feature_columns)} features"
    )

    # Define simple model factory for testing
    def simple_model_factory(**params):
        from sklearn.ensemble import RandomForestRegressor

        # Extract parameters with defaults
        n_estimators = params.get('rf_n_estimators', 100)
        max_depth = params.get('rf_max_depth', 10)

        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    # Create optimization config
    config = HyperoptConfig(
        optimization_method='random',
        # Use random for demo (no optuna dependency)
        n_trials=10,  # Small number for demo
        primary_metric='mae',
        optimize_model_params=True,
        cv_folds=3
    )

    print("Running hyperparameter optimization...")
    print(f"  Method: {config.optimization_method}")
    print(f"  Trials: {config.n_trials}")
    print(f"  Metric: {config.primary_metric}")

    try:
        # Run optimization
        result = optimize_forecasting_model(
            model_factory=simple_model_factory,
            data=data,
            feature_columns=feature_columns,
            config=config
        )

        print("\n🎯 Optimization Results:")
        print(f"  Best Score: {result.best_score:.6f}")
        print(f"  Trials Completed: {result.n_trials}")
        print(f"  Optimization Time: {result.optimization_time:.2f}s")
        print(f"  Stability Score: {result.stability_score:.6f}")

        print("\n📊 Best Parameters:")
        for param, value in result.best_params.items():
            print(f"  {param}: {value}")

        print("\n📈 Cross-Validation Scores:")
        for i, score in enumerate(result.cv_scores):
            print(f"  Fold {i+1}: {score:.6f}")
        mean_score = np.mean(result.cv_scores)
        std_score = np.std(result.cv_scores)
        print(f"  Mean: {mean_score:.6f} ± {std_score:.6f}")

        print("\n✅ Hyperparameter optimization completed successfully!")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
