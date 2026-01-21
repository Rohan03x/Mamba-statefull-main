"""
Meta-Learner and Gating Network Implementation

Implements the second part of the ensemble framework:
- Meta-learner training on out-of-fold predictions only
- Intelligent gating network for model selection
- Regime-aware ensemble weighting
- Options expert integration with adaptive gating
"""

import logging
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .ensemble_framework import EnsembleConfig, BaseModelTrainer, RegimeDetector, OptionsExpert

logger = logging.getLogger(__name__)


class MetaLearner:
    """
    Meta-learner that combines base model predictions using only out-of-fold data.
    
    Prevents leakage by training exclusively on OOF predictions from base models.
    Incorporates regime information and options expert signals.
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.meta_model = self._create_meta_model()
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.base_model_names = []
        self.feature_importance = {}
        
    def _create_meta_model(self) -> BaseEstimator:
        """Create meta-model based on configuration"""
        if self.config.meta_learner_type == "linear":
            return LinearRegression()
        elif self.config.meta_learner_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.config.meta_learner_type == "nn":
            return MLPRegressor(
                hidden_layer_sizes=(50, 25), 
                max_iter=500, 
                random_state=42,
                early_stopping=True
            )
        elif self.config.meta_learner_type == "xgb":
            import os
            params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'random_state': 42
            }
            # Add GPU support via environment variable
            device = os.environ.get('XGB_DEVICE', 'cpu')
            if device == 'gpu':
                params['device'] = 'cuda'
                params['tree_method'] = 'hist'
            else:
                params['tree_method'] = os.environ.get('XGB_TREE_METHOD', 'hist')
            return xgb.XGBRegressor(**params)
        else:
            logger.warning(f"Unknown meta-learner type: {self.config.meta_learner_type}. Using Ridge.")
            return Ridge(alpha=1.0)
    
    def prepare_meta_features(self, 
                            oof_predictions: Dict[str, np.ndarray],
                            regime_probs: Optional[pd.DataFrame] = None,
                            options_predictions: Optional[np.ndarray] = None,
                            options_weights: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Prepare meta-features from base model predictions and additional signals.
        
        Args:
            oof_predictions: Out-of-fold predictions from base models
            regime_probs: Regime probability features
            options_predictions: Options expert predictions
            options_weights: Options expert confidence weights
            
        Returns:
            DataFrame with meta-features
        """
        # Base model predictions
        meta_features = pd.DataFrame(oof_predictions)
        self.base_model_names = list(oof_predictions.keys())
        
        # Add interaction features between base models
        if len(self.base_model_names) >= 2:
            # Model agreement/disagreement features
            predictions_array = np.column_stack(list(oof_predictions.values()))
            
            # Average prediction
            meta_features['ensemble_mean'] = np.nanmean(predictions_array, axis=1)
            
            # Prediction variance (disagreement)
            meta_features['prediction_variance'] = np.nanvar(predictions_array, axis=1)
            
            # Min/max spread
            meta_features['prediction_spread'] = (
                np.nanmax(predictions_array, axis=1) - 
                np.nanmin(predictions_array, axis=1)
            )
            
            # Median prediction
            meta_features['ensemble_median'] = np.nanmedian(predictions_array, axis=1)
        
        # Add regime information
        if regime_probs is not None:
            for col in regime_probs.columns:
                meta_features[col] = regime_probs[col].values
                
            # Create regime-weighted predictions
            for model_name in self.base_model_names:
                for regime_col in regime_probs.columns:
                    feature_name = f"{model_name}_x_{regime_col}"
                    meta_features[feature_name] = (
                        meta_features[model_name] * meta_features[regime_col]
                    )
        
        # Add options expert information
        if options_predictions is not None:
            meta_features['options_expert'] = options_predictions
            
            if options_weights is not None:
                meta_features['options_confidence'] = options_weights
                
                # Options-weighted ensemble
                meta_features['options_weighted_ensemble'] = (
                    meta_features['ensemble_mean'] * (1 - options_weights) +
                    options_predictions * options_weights
                )
        
        # Handle missing values
        meta_features = meta_features.fillna(0)
        
        logger.info(f"Created meta-features: {meta_features.shape[1]} features")
        return meta_features
    
    def fit(self, oof_predictions: Dict[str, np.ndarray], 
            y_train: pd.Series,
            regime_probs: Optional[pd.DataFrame] = None,
            options_predictions: Optional[np.ndarray] = None,
            options_weights: Optional[np.ndarray] = None) -> 'MetaLearner':
        """
        Train meta-learner on out-of-fold predictions.
        
        Args:
            oof_predictions: OOF predictions from base models
            y_train: Training targets
            regime_probs: Regime probabilities
            options_predictions: Options expert predictions
            options_weights: Options expert confidence weights
            
        Returns:
            Self for method chaining
        """
        logger.info("Training meta-learner on out-of-fold predictions...")
        
        # Prepare meta-features
        meta_features = self.prepare_meta_features(
            oof_predictions, regime_probs, options_predictions, options_weights
        )
        
        # Align with targets and remove NaN samples
        valid_mask = ~np.isnan(np.column_stack(list(oof_predictions.values()))).any(axis=1)
        meta_features_clean = meta_features.loc[valid_mask]
        y_train_clean = y_train.iloc[valid_mask]
        
        logger.info(f"Training on {len(meta_features_clean)} valid samples "
                   f"({len(y_train) - len(meta_features_clean)} removed due to NaN)")
        
        if len(meta_features_clean) < 100:
            logger.warning(f"Very few samples for meta-learner training: {len(meta_features_clean)}")
        
        # Scale features
        meta_features_scaled = self.feature_scaler.fit_transform(meta_features_clean)
        
        # Train meta-model
        self.meta_model.fit(meta_features_scaled, y_train_clean)
        self.is_trained = True
        
        # Calculate training performance
        meta_pred = self.meta_model.predict(meta_features_scaled)
        meta_mse = mean_squared_error(y_train_clean, meta_pred)
        logger.info(f"Meta-learner training RMSE: {np.sqrt(meta_mse):.6f}")
        
        # Calculate feature importance if available
        if hasattr(self.meta_model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                meta_features.columns, 
                self.meta_model.feature_importances_
            ))
            
            # Log top features
            top_features = sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            logger.info("Top meta-features:")
            for feat, importance in top_features:
                logger.info(f"  {feat}: {importance:.4f}")
        
        return self
    
    def predict(self, base_predictions: Dict[str, np.ndarray],
               regime_probs: Optional[pd.DataFrame] = None,
               options_predictions: Optional[np.ndarray] = None,
               options_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate meta-learner predictions.
        
        Args:
            base_predictions: Predictions from base models
            regime_probs: Regime probabilities
            options_predictions: Options expert predictions
            options_weights: Options expert confidence weights
            
        Returns:
            Meta-learner predictions
        """
        if not self.is_trained:
            raise ValueError("Meta-learner not trained. Call fit() first.")
        
        # Prepare meta-features
        meta_features = self.prepare_meta_features(
            base_predictions, regime_probs, options_predictions, options_weights
        )
        
        # Scale features
        meta_features_scaled = self.feature_scaler.transform(meta_features)
        
        # Generate predictions
        return self.meta_model.predict(meta_features_scaled)


class GatingNetwork:
    """
    Intelligent gating network that decides when to trust different experts.
    
    Learns when to rely on base models vs options expert vs regime-specific models
    based on market conditions and model confidence.
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.gate_model = MLPRegressor(
            hidden_layer_sizes=(20, 10),
            activation='tanh',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        self.gate_scaler = StandardScaler()
        self.is_trained = False
        
    def create_gating_features(self, X: pd.DataFrame,
                             regime_probs: Optional[pd.DataFrame] = None,
                             options_weights: Optional[np.ndarray] = None,
                             prediction_variance: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create features for gating decisions.
        
        Args:
            X: Base features
            regime_probs: Regime probabilities
            options_weights: Options expert confidence
            prediction_variance: Disagreement between base models
            
        Returns:
            DataFrame with gating features
        """
        gating_features = pd.DataFrame(index=X.index)
        
        # Market condition features
        vol_cols = [c for c in X.columns if 'volatility' in c.lower()]
        if vol_cols:
            volatility = X[vol_cols[0]]
            gating_features['market_volatility'] = volatility
            gating_features['vol_regime'] = (volatility > volatility.rolling(252).quantile(0.75)).astype(float)
        
        # Momentum features
        momentum_cols = [c for c in X.columns if 'momentum' in c.lower()]
        if momentum_cols:
            momentum = X[momentum_cols[0]]
            gating_features['momentum_strength'] = np.abs(momentum)
            gating_features['trend_direction'] = np.sign(momentum)
        
        # Technical indicator confidence
        rsi_cols = [c for c in X.columns if 'rsi' in c.lower()]
        if rsi_cols:
            rsi = X[rsi_cols[0]]
            gating_features['rsi_extreme'] = ((rsi > 70) | (rsi < 30)).astype(float)
            gating_features['rsi_neutral'] = ((rsi >= 40) & (rsi <= 60)).astype(float)
        
        # Volume information
        volume_cols = [c for c in X.columns if 'volume' in c.lower()]
        if volume_cols:
            volume_ratio = X[volume_cols[0]]
            gating_features['high_volume'] = (volume_ratio > 2.0).astype(float)
        
        # Add regime information
        if regime_probs is not None:
            for col in regime_probs.columns:
                gating_features[col] = regime_probs[col].values
                
            # Regime uncertainty (entropy)
            regime_entropy = -np.sum(
                regime_probs.values * np.log(regime_probs.values + 1e-8), 
                axis=1
            )
            gating_features['regime_uncertainty'] = regime_entropy
        
        # Options market signals
        if options_weights is not None:
            gating_features['options_confidence'] = options_weights
            gating_features['high_iv_period'] = (options_weights > 1.5).astype(float)
        
        # Model disagreement
        if prediction_variance is not None:
            gating_features['model_disagreement'] = prediction_variance
            gating_features['high_disagreement'] = (
                prediction_variance > np.quantile(prediction_variance, 0.8)
            ).astype(float)
        
        # Time-based features
        gating_features['month'] = X.index.month
        gating_features['quarter_end'] = ((X.index.month % 3) == 0).astype(float)
        gating_features['year_end'] = (X.index.month == 12).astype(float)
        
        # Handle missing values
        gating_features = gating_features.fillna(0)
        
        return gating_features
    
    def fit(self, X_train: pd.DataFrame, 
            base_predictions: Dict[str, np.ndarray],
            options_predictions: np.ndarray,
            options_weights: np.ndarray,
            regime_probs: pd.DataFrame,
            y_train: pd.Series) -> 'GatingNetwork':
        """
        Train gating network to learn optimal model selection.
        
        The gating network learns when to trust different experts based on
        their historical performance in different market conditions.
        """
        logger.info("Training gating network...")
        
        # Calculate prediction variance
        predictions_array = np.column_stack(list(base_predictions.values()))
        prediction_variance = np.var(predictions_array, axis=1)
        
        # Create gating features
        gating_features = self.create_gating_features(
            X_train, regime_probs, options_weights, prediction_variance
        )
        
        # Create target: which model performed best in each sample
        # This is a simplified approach - in practice you'd want a more sophisticated target
        base_errors = {}
        for model_name, predictions in base_predictions.items():
            valid_mask = ~np.isnan(predictions)
            if valid_mask.sum() > 0:
                base_errors[model_name] = np.full(len(y_train), np.inf)
                base_errors[model_name][valid_mask] = np.abs(
                    predictions[valid_mask] - y_train.iloc[valid_mask]
                )
        
        # Add options expert
        options_error = np.abs(options_predictions - y_train.values)
        base_errors['options_expert'] = options_error
        
        # Find best model for each sample
        error_matrix = np.column_stack(list(base_errors.values()))
        best_model_idx = np.argmin(error_matrix, axis=1)
        
        # Convert to soft targets (probabilities)
        n_models = len(base_errors)
        gate_targets = np.zeros((len(y_train), n_models))
        for i, best_idx in enumerate(best_model_idx):
            gate_targets[i, best_idx] = 1.0
        
        # Add some smoothing to avoid overfitting to single best model
        smoothing = 0.1
        gate_targets = gate_targets * (1 - smoothing) + smoothing / n_models
        
        # Train gating network
        valid_samples = ~np.isnan(predictions_array).any(axis=1)
        gating_features_clean = gating_features.loc[valid_samples]
        gate_targets_clean = gate_targets[valid_samples]
        
        # Use the first model's weight as a simplified target for regression
        # In practice, you'd want a more sophisticated multi-output approach
        gate_target_simple = gate_targets_clean[:, 0]  # Weight for first model
        
        gating_features_scaled = self.gate_scaler.fit_transform(gating_features_clean)
        self.gate_model.fit(gating_features_scaled, gate_target_simple)
        self.is_trained = True
        
        # Evaluate gating performance
        gate_pred = self.gate_model.predict(gating_features_scaled)
        gate_mse = mean_squared_error(gate_target_simple, gate_pred)
        logger.info(f"Gating network training RMSE: {np.sqrt(gate_mse):.6f}")
        
        return self
    
    def get_model_weights(self, X: pd.DataFrame,
                         regime_probs: Optional[pd.DataFrame] = None,
                         options_weights: Optional[np.ndarray] = None,
                         prediction_variance: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Get adaptive weights for different models based on current conditions.
        
        Returns:
            Dictionary mapping model names to weight arrays
        """
        if not self.is_trained:
            logger.warning("Gating network not trained. Using equal weights.")
            n_samples = len(X)
            return {
                'base_ensemble': np.ones(n_samples) * 0.7,
                'options_expert': np.ones(n_samples) * 0.3
            }
        
        # Create gating features
        gating_features = self.create_gating_features(
            X, regime_probs, options_weights, prediction_variance
        )
        
        # Get gating predictions
        gating_features_scaled = self.gate_scaler.transform(gating_features)
        base_weight = self.gate_model.predict(gating_features_scaled)
        
        # Clip weights to reasonable bounds
        base_weight = np.clip(base_weight, 0.1, 0.9)
        options_weight = 1.0 - base_weight
        
        # Boost options expert during high IV periods
        if options_weights is not None:
            high_iv_mask = options_weights > 1.5
            options_weight[high_iv_mask] = np.minimum(
                options_weight[high_iv_mask] * 1.5, 0.8
            )
            base_weight = 1.0 - options_weight
        
        return {
            'base_ensemble': base_weight,
            'options_expert': options_weight
        }


class EnsembleStacker:
    """
    Main ensemble stacking class that orchestrates the entire process.
    
    Combines:
    - Base model training with proper OOF generation
    - Regime detection
    - Options expert
    - Meta-learner training
    - Gating network
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.base_trainer = BaseModelTrainer(config)
        self.regime_detector = RegimeDetector(
            config.n_regimes, config.regime_features
        ) if config.enable_regime_detection else None
        self.options_expert = OptionsExpert(
            config.options_lookback_days, config.earnings_window_days
        ) if config.enable_options_expert else None
        self.meta_learner = MetaLearner(config)
        self.gating_network = GatingNetwork(config)
        self.is_trained = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, cv_splitter) -> 'EnsembleStacker':
        """
        Train the complete ensemble stack.
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_splitter: Cross-validation splitter for base models
            
        Returns:
            Self for method chaining
        """
        logger.info("Training ensemble stack...")
        
        # Step 1: Train base models and generate OOF predictions
        logger.info("Step 1: Training base models...")
        oof_predictions = self.base_trainer.train_base_models(X_train, y_train, cv_splitter)
        
        # Step 2: Train regime detector
        regime_probs = None
        if self.regime_detector is not None:
            logger.info("Step 2: Training regime detector...")
            self.regime_detector.fit(X_train)
            regime_probs = self.regime_detector.predict_regime_probabilities(X_train)
        
        # Step 3: Train options expert
        options_predictions = None
        options_weights = None
        if self.options_expert is not None:
            logger.info("Step 3: Training options expert...")
            self.options_expert.fit(X_train, y_train)
            options_predictions = self.options_expert.predict(X_train)
            options_weights = self.options_expert.get_confidence_weights(X_train)
        
        # Step 4: Train meta-learner on OOF predictions
        logger.info("Step 4: Training meta-learner...")
        self.meta_learner.fit(
            oof_predictions, y_train, regime_probs, 
            options_predictions, options_weights
        )
        
        # Step 5: Train gating network
        logger.info("Step 5: Training gating network...")
        if self.options_expert is not None and regime_probs is not None:
            self.gating_network.fit(
                X_train, oof_predictions, options_predictions, 
                options_weights, regime_probs, y_train
            )
        
        self.is_trained = True
        logger.info("Ensemble stack training completed!")
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Final ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call fit() first.")
        
        logger.info("Generating ensemble predictions...")
        
        # Get base model predictions
        base_predictions = self.base_trainer.predict_with_base_models(X_test)
        
        # Get regime probabilities
        regime_probs = None
        if self.regime_detector is not None:
            regime_probs = self.regime_detector.predict_regime_probabilities(X_test)
        
        # Get options expert predictions
        options_predictions = None
        options_weights = None
        if self.options_expert is not None:
            options_predictions = self.options_expert.predict(X_test)
            options_weights = self.options_expert.get_confidence_weights(X_test)
        
        # Get meta-learner predictions
        meta_predictions = self.meta_learner.predict(
            base_predictions, regime_probs, options_predictions, options_weights
        )
        
        # Get gating weights
        prediction_variance = None
        if len(base_predictions) > 1:
            predictions_array = np.column_stack(list(base_predictions.values()))
            prediction_variance = np.var(predictions_array, axis=1)
        
        model_weights = self.gating_network.get_model_weights(
            X_test, regime_probs, options_weights, prediction_variance
        )
        
        # Combine predictions using gating weights
        final_predictions = (
            meta_predictions * model_weights['base_ensemble']
        )
        
        if options_predictions is not None:
            final_predictions += (
                options_predictions * model_weights['options_expert']
            )
        
        logger.info("Ensemble predictions generated")
        return final_predictions
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about model performance and feature importance"""
        insights = {}
        
        if hasattr(self.meta_learner, 'feature_importance'):
            insights['meta_feature_importance'] = self.meta_learner.feature_importance
        
        if self.regime_detector is not None and hasattr(self.regime_detector, 'regime_names'):
            insights['regime_names'] = self.regime_detector.regime_names
        
        insights['base_models'] = list(self.base_trainer.base_models.keys())
        insights['config'] = self.config
        
        return insights