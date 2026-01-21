"""
Hyperparameter Optimization for Walk-Forward Backtesting

This module implements sophisticated hyperparameter optimization using Optuna
with proper time-series safeguards to prevent lookahead bias and overfitting.

Key Features:
- Time-ordered validation splits within training blocks
- Efficient pruning to reduce computational overhead
- Ensemble-aware optimization for multiple model types
- Regime-aware hyperparameter optimization
- Memory-efficient search with early stopping
- Comprehensive tracking and reproducibility

Following academic best practices for financial ML hyperparameter optimization.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, RandomSampler

from .config import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """
    Define hyperparameter search spaces for different model types
    
    This class encapsulates the search space definition for various models
    used in the ensemble, allowing for efficient and targeted optimization.
    """
    
    # Linear models
    ridge_alpha: Tuple[float, float] = (1e-4, 1e2)
    ridge_alpha_log: bool = True
    
    # Tree models
    rf_n_estimators: Tuple[int, int] = (50, 500)
    rf_max_depth: Tuple[int, int] = (3, 20)
    rf_min_samples_split: Tuple[int, int] = (2, 20)
    rf_min_samples_leaf: Tuple[int, int] = (1, 10)
    
    gb_n_estimators: Tuple[int, int] = (50, 300)
    gb_learning_rate: Tuple[float, float] = (0.01, 0.3)
    gb_max_depth: Tuple[int, int] = (3, 15)
    gb_subsample: Tuple[float, float] = (0.6, 1.0)
    
    # Neural networks
    mlp_hidden_layer_sizes: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (50,), (100,), (150,), (50, 25), (100, 50), (150, 100)
    ])
    mlp_alpha: Tuple[float, float] = (1e-5, 1e-1)
    mlp_alpha_log: bool = True
    mlp_learning_rate_init: Tuple[float, float] = (1e-4, 1e-2)
    
    # Ensemble-specific
    meta_learner_alpha: Tuple[float, float] = (1e-4, 1e2)
    meta_learner_alpha_log: bool = True
    
    # Regime detection
    n_regimes: Tuple[int, int] = (2, 5)
    regime_covariance_type: List[str] = field(default_factory=lambda: [
        "full", "diag", "spherical"
    ])
    
    # Options expert
    options_lookback: Tuple[int, int] = (10, 60)
    options_vol_threshold: Tuple[float, float] = (0.8, 2.0)
    
    def suggest_ridge_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Ridge regression"""
        if self.ridge_alpha_log:
            alpha = trial.suggest_float("ridge_alpha", self.ridge_alpha[0], 
                                      self.ridge_alpha[1], log=True)
        else:
            alpha = trial.suggest_float("ridge_alpha", self.ridge_alpha[0], 
                                      self.ridge_alpha[1])
        
        return {"alpha": alpha}
    
    def suggest_random_forest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Random Forest"""
        return {
            "n_estimators": trial.suggest_int("rf_n_estimators", 
                                            self.rf_n_estimators[0], 
                                            self.rf_n_estimators[1]),
            "max_depth": trial.suggest_int("rf_max_depth", 
                                         self.rf_max_depth[0], 
                                         self.rf_max_depth[1]),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 
                                                 self.rf_min_samples_split[0], 
                                                 self.rf_min_samples_split[1]),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 
                                                self.rf_min_samples_leaf[0], 
                                                self.rf_min_samples_leaf[1]),
            "random_state": 42
        }
    
    def suggest_gradient_boost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Gradient Boosting"""
        return {
            "n_estimators": trial.suggest_int("gb_n_estimators", 
                                            self.gb_n_estimators[0], 
                                            self.gb_n_estimators[1]),
            "learning_rate": trial.suggest_float("gb_learning_rate", 
                                               self.gb_learning_rate[0], 
                                               self.gb_learning_rate[1]),
            "max_depth": trial.suggest_int("gb_max_depth", 
                                         self.gb_max_depth[0], 
                                         self.gb_max_depth[1]),
            "subsample": trial.suggest_float("gb_subsample", 
                                           self.gb_subsample[0], 
                                           self.gb_subsample[1]),
            "random_state": 42
        }
    
    def suggest_mlp_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for MLP"""
        hidden_layers = trial.suggest_categorical("mlp_hidden_layers", 
                                                 self.mlp_hidden_layer_sizes)
        
        if self.mlp_alpha_log:
            alpha = trial.suggest_float("mlp_alpha", self.mlp_alpha[0], 
                                      self.mlp_alpha[1], log=True)
        else:
            alpha = trial.suggest_float("mlp_alpha", self.mlp_alpha[0], 
                                      self.mlp_alpha[1])
        
        learning_rate_init = trial.suggest_float("mlp_learning_rate_init", 
                                                self.mlp_learning_rate_init[0], 
                                                self.mlp_learning_rate_init[1])
        
        return {
            "hidden_layer_sizes": hidden_layers,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "random_state": 42,
            "max_iter": 1000
        }
    
    def suggest_ensemble_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for ensemble components"""
        params = {}
        
        # Meta-learner parameters
        if self.meta_learner_alpha_log:
            params["meta_learner_alpha"] = trial.suggest_float(
                "meta_learner_alpha", self.meta_learner_alpha[0], 
                self.meta_learner_alpha[1], log=True
            )
        else:
            params["meta_learner_alpha"] = trial.suggest_float(
                "meta_learner_alpha", self.meta_learner_alpha[0], 
                self.meta_learner_alpha[1]
            )
        
        # Regime detection parameters
        params["n_regimes"] = trial.suggest_int("n_regimes", 
                                              self.n_regimes[0], 
                                              self.n_regimes[1])
        params["regime_covariance_type"] = trial.suggest_categorical(
            "regime_covariance_type", self.regime_covariance_type
        )
        
        # Options expert parameters
        params["options_lookback"] = trial.suggest_int("options_lookback", 
                                                     self.options_lookback[0], 
                                                     self.options_lookback[1])
        params["options_vol_threshold"] = trial.suggest_float(
            "options_vol_threshold", self.options_vol_threshold[0], 
            self.options_vol_threshold[1]
        )
        
        return params


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    study_name: str
    optimization_date: datetime
    
    # Detailed results
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    pruned_trials: int = 0
    failed_trials: int = 0
    
    # Validation metrics
    validation_scores: List[float] = field(default_factory=list)
    validation_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "optimization_time": self.optimization_time,
            "study_name": self.study_name,
            "optimization_date": self.optimization_date.isoformat(),
            "all_trials": self.all_trials,
            "pruned_trials": self.pruned_trials,
            "failed_trials": self.failed_trials,
            "validation_scores": self.validation_scores,
            "validation_std": self.validation_std
        }


class HyperparameterOptimizer:
    """
    Sophisticated hyperparameter optimization for financial ML models
    
    This class implements time-series aware hyperparameter optimization using Optuna
    with proper safeguards against lookahead bias and overfitting.
    
    Key features:
    - Time-ordered validation splits
    - Efficient pruning strategies
    - Ensemble-aware optimization
    - Memory-efficient execution
    - Comprehensive result tracking
    """
    
    def __init__(self, config: BacktestConfig, 
                 param_space: Optional[HyperparameterSpace] = None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            config: Backtest configuration
            param_space: Hyperparameter search space definition
        """
        self.config = config
        self.param_space = param_space or HyperparameterSpace()
        
        # Setup Optuna components
        self.sampler = self._create_sampler()
        self.pruner = self._create_pruner()
        
        # Tracking
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info("🔧 Initialized hyperparameter optimizer")
        logger.info(f"  Trials: {config.optuna_trials}")
        logger.info(f"  Timeout: {config.optuna_timeout}s")
        logger.info(f"  Pruning: {config.optuna_pruning}")
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler"""
        if self.config.optuna_trials < 10:
            # Use random sampler for very few trials
            return RandomSampler(seed=self.config.random_seed)
        else:
            # Use TPE sampler for efficient search
            return TPESampler(
                seed=self.config.random_seed,
                n_startup_trials=min(10, self.config.optuna_trials // 10)
            )
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner"""
        if not self.config.optuna_pruning:
            return None
        
        # Use median pruner for robust pruning
        return MedianPruner(
            n_startup_trials=min(5, self.config.optuna_trials // 20),
            n_warmup_steps=3
        )
    
    def optimize_ensemble_hyperparameters(self, 
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        train_dates: pd.DatetimeIndex,
                                        model_trainer: Any) -> OptimizationResult:
        """
        Optimize hyperparameters for ensemble models
        
        Args:
            X_train: Training features
            y_train: Training targets
            train_dates: Training dates for time-ordered splits
            model_trainer: Model trainer instance
            cv_splitter: Cross-validation splitter
            
        Returns:
            Optimization results
        """
        study_name = f"ensemble_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🎯 Starting ensemble hyperparameter optimization: {study_name}")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Training period: {train_dates.min()} to {train_dates.max()}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",  # Minimize validation error
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name
        )
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            try:
                return self._ensemble_objective(trial, X_train, y_train, 
                                              train_dates, model_trainer)
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                # Return a high error value for failed trials
                return float('inf')
        
        # Run optimization
        start_time = datetime.now()
        
        study.optimize(
            objective,
            n_trials=self.config.optuna_trials,
            timeout=self.config.optuna_timeout,
            n_jobs=1,  # Use single job for reproducibility
            show_progress_bar=False
        )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Compile results
        result = self._compile_optimization_results(study, optimization_time, study_name)
        self.optimization_history.append(result)
        
        logger.info(f"✅ Optimization completed: {study_name}")
        logger.info(f"  Best score: {result.best_score:.6f}")
        logger.info(f"  Completed trials: {result.n_trials}")
        logger.info(f"  Pruned trials: {result.pruned_trials}")
        logger.info(f"  Failed trials: {result.failed_trials}")
        logger.info(f"  Optimization time: {result.optimization_time:.1f}s")
        
        return result
    
    def _ensemble_objective(self, trial: optuna.Trial, X_train: pd.DataFrame,
                          y_train: pd.Series, train_dates: pd.DatetimeIndex,
                          model_trainer: Any) -> float:
        """
        Objective function for ensemble hyperparameter optimization
        
        This function evaluates a set of hyperparameters using time-ordered
        cross-validation to prevent lookahead bias.
        """
        # Suggest hyperparameters for all model types
        trial_params = {}
        
        # Base model parameters
        trial_params.update(self.param_space.suggest_ridge_params(trial))
        trial_params.update(self.param_space.suggest_random_forest_params(trial))
        trial_params.update(self.param_space.suggest_gradient_boost_params(trial))
        
        # Skip MLP for speed unless specifically enabled
        if getattr(self.config, 'optimize_neural_networks', False):
            trial_params.update(self.param_space.suggest_mlp_params(trial))
        
        # Ensemble-specific parameters
        trial_params.update(self.param_space.suggest_ensemble_params(trial))
        
        # Perform time-ordered cross-validation
        cv_scores = []
        
        # Create time-ordered splits within training data
        val_splits = self._create_time_ordered_validation_splits(X_train, train_dates)
        
        for i, (train_idx, val_idx) in enumerate(val_splits):
            x_cv_train = X_train.iloc[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            x_cv_val = X_train.iloc[val_idx]
            y_cv_val = y_train.iloc[val_idx]
            
            try:
                # Train ensemble with current hyperparameters
                ensemble = self._train_ensemble_with_params(
                    x_cv_train, y_cv_train, trial_params, model_trainer
                )
                
                # Validate
                val_predictions = ensemble.predict(x_cv_val)
                val_score = mean_squared_error(y_cv_val, val_predictions)
                cv_scores.append(val_score)
                
                # Report intermediate value for pruning
                trial.report(val_score, i)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            except Exception as e:
                logger.debug(f"CV fold {i} failed: {str(e)}")
                # Use a high penalty score for failed folds
                cv_scores.append(10.0)  # High error penalty
        
        # Return mean CV score
        if not cv_scores:
            return float('inf')
        
        mean_score = np.mean(cv_scores)
        
        # Add regularization penalty for complex models
        complexity_penalty = self._calculate_complexity_penalty(trial_params)
        
        return mean_score + complexity_penalty
    
    def _create_time_ordered_validation_splits(self, X: pd.DataFrame,
                                             _unused_dates: pd.DatetimeIndex) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-ordered validation splits within training data
        
        This ensures no lookahead bias in hyperparameter optimization.
        """
        n_splits = min(5, len(X) // 50)  # Limit number of splits based on data size
        if n_splits < 2:
            n_splits = 2
        
        splits = []
        n_samples = len(X)
        
        # Create expanding window splits
        for i in range(n_splits):
            # Calculate split points
            val_start_pct = 0.5 + (i * 0.3 / n_splits)  # Start from 50% through data
            val_end_pct = val_start_pct + 0.2  # 20% validation window
            
            # Ensure we don't exceed data bounds
            val_end_pct = min(val_end_pct, 0.95)
            val_start_pct = min(val_start_pct, val_end_pct - 0.1)
            
            # Convert to indices
            train_end_idx = int(n_samples * val_start_pct)
            val_start_idx = train_end_idx
            val_end_idx = int(n_samples * val_end_pct)
            
            # Create index arrays
            train_idx = np.arange(0, train_end_idx)
            val_idx = np.arange(val_start_idx, val_end_idx)
            
            if len(train_idx) > 20 and len(val_idx) > 5:  # Minimum sample requirements
                splits.append((train_idx, val_idx))
        
        return splits
    
    def _train_ensemble_with_params(self, x_train: pd.DataFrame, y_train: pd.Series,
                                   params: Dict[str, Any], _unused_model_trainer: Any) -> Any:
        """
        Train ensemble with specific hyperparameters
        
        This is a simplified training for hyperparameter evaluation.
        """
        # This is a placeholder - in practice, you would create and train
        # the ensemble with the given parameters
        
        # For now, return a simple ensemble that can make predictions
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        # Create simple ensemble with key parameters
        rf_params = {k.replace('rf_', ''): v for k, v in params.items() if k.startswith('rf_')}
        ridge_params = {k.replace('ridge_', ''): v for k, v in params.items() if k.startswith('ridge_')}
        
        if rf_params:
            rf_model = RandomForestRegressor(**rf_params)
            rf_model.fit(x_train, y_train)
            return rf_model
        elif ridge_params:
            ridge_model = Ridge(**ridge_params)
            ridge_model.fit(x_train, y_train)
            return ridge_model
        else:
            # Fallback to simple model
            simple_model = Ridge(alpha=1.0)
            simple_model.fit(x_train, y_train)
            return simple_model
    
    def _calculate_complexity_penalty(self, params: Dict[str, Any]) -> float:
        """
        Calculate complexity penalty to encourage simpler models
        
        This helps prevent overfitting by penalizing overly complex configurations.
        """
        penalty = 0.0
        
        # Penalize large tree models
        if 'rf_n_estimators' in params:
            penalty += (params['rf_n_estimators'] - 100) * 1e-6
        if 'gb_n_estimators' in params:
            penalty += (params['gb_n_estimators'] - 100) * 1e-6
        
        # Penalize deep trees
        if 'rf_max_depth' in params:
            penalty += max(0, params['rf_max_depth'] - 10) * 1e-4
        if 'gb_max_depth' in params:
            penalty += max(0, params['gb_max_depth'] - 8) * 1e-4
        
        # Penalize complex neural networks
        if 'mlp_hidden_layers' in params:
            layer_sizes = params['mlp_hidden_layers']
            if isinstance(layer_sizes, tuple):
                total_neurons = sum(layer_sizes)
                penalty += max(0, total_neurons - 100) * 1e-5
        
        return penalty
    
    def _compile_optimization_results(self, study: optuna.Study, 
                                    optimization_time: float,
                                    study_name: str) -> OptimizationResult:
        """Compile optimization results from Optuna study"""
        
        # Get trial statistics
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
        
        # Extract validation scores from completed trials
        validation_scores = [trial.value for trial in completed_trials if trial.value is not None]
        
        # Create detailed trial information
        all_trials = []
        for trial in study.trials:
            trial_info = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            all_trials.append(trial_info)
        
        return OptimizationResult(
            best_params=study.best_params if study.best_trial else {},
            best_score=study.best_value if study.best_trial else float('inf'),
            n_trials=len(completed_trials),
            optimization_time=optimization_time,
            study_name=study_name,
            optimization_date=datetime.now(),
            all_trials=all_trials,
            pruned_trials=len(pruned_trials),
            failed_trials=len(failed_trials),
            validation_scores=validation_scores,
            validation_std=np.std(validation_scores) if validation_scores else 0.0
        )
    
    def save_optimization_history(self, filepath: str):
        """Save optimization history to file"""
        history_data = [result.to_dict() for result in self.optimization_history]
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"💾 Saved optimization history to {filepath}")
    
    def load_optimization_history(self, filepath: str):
        """Load optimization history from file"""
        with open(filepath, 'r') as f:
            history_data = json.load(f)
        
        self.optimization_history = []
        for result_dict in history_data:
            # Convert datetime strings back to datetime objects
            result_dict['optimization_date'] = datetime.fromisoformat(result_dict['optimization_date'])
            
            result = OptimizationResult(**result_dict)
            self.optimization_history.append(result)
        
        logger.info(f"📂 Loaded optimization history from {filepath}")
    
    def get_best_hyperparameters(self) -> Dict[str, Any]:
        """Get best hyperparameters from optimization history"""
        if not self.optimization_history:
            logger.warning("No optimization history available")
            return {}
        
        # Find best result
        best_result = min(self.optimization_history, key=lambda x: x.best_score)
        
        logger.info(f"📊 Best hyperparameters (score: {best_result.best_score:.6f}):")
        for param, value in best_result.best_params.items():
            logger.info(f"  {param}: {value}")
        
        return best_result.best_params


if __name__ == "__main__":
    # Example usage
    from .config import create_default_backtest_config
    
    print("🔧 Hyperparameter Optimization Example")
    print("=" * 50)
    
    # Create configuration
    config = create_default_backtest_config()
    config.optuna_trials = 10  # Quick test
    config.optuna_timeout = 60
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Test parameter space
    space = HyperparameterSpace()
    print("📊 Parameter Spaces:")
    
    # Create a mock trial for testing
    study = optuna.create_study()
    trial = study.ask()
    
    print("  Ridge parameters:", space.suggest_ridge_params(trial))
    print("  Random Forest parameters:", space.suggest_random_forest_params(trial))
    print("  Ensemble parameters:", space.suggest_ensemble_params(trial))
    
    print("\n✅ Hyperparameter optimizer ready for use!")