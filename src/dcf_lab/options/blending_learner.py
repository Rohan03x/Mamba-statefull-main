"""
Blending Learner for Options-anchored Framework

This module implements small learners that combine options priors with
technical/macro/news features to predict realized moves/returns.

Key components:
- Feature blending neural networks
- Prior weighting mechanisms
- Blend optimization strategies
- Integration with ensemble framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import logging

from .options_prior import ImpliedDistribution

logger = logging.getLogger(__name__)

@dataclass
class BlendingResult:
    """Results from blending options priors with features"""
    blended_prediction: float
    prior_weight: float
    feature_weight: float
    options_prior_value: float
    feature_prediction: float
    confidence_score: float
    blend_components: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class BlendingConfig:
    """Configuration for blending learner"""
    model_type: str = 'neural_network'  # 'neural_network', 'random_forest', 'gradient_boosting', 'linear'
    hidden_sizes: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    regularization: float = 0.01
    dropout_rate: float = 0.2
    prior_weight_range: Tuple[float, float] = (0.0, 1.0)
    adaptive_weighting: bool = True
    feature_scaling: bool = True

class PriorWeightingNetwork(nn.Module):
    """Neural network for learning optimal prior weights"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer for prior weight (sigmoid to ensure 0-1 range)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class FeatureBlendingNetwork(nn.Module):
    """Neural network for blending features with options prior"""
    
    def __init__(self, feature_size: int, hidden_sizes: List[int] = None,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        # Feature processing branch
        feature_layers = []
        prev_size = feature_size
        
        for hidden_size in hidden_sizes:
            feature_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Feature prediction head
        feature_layers.append(nn.Linear(prev_size, 1))
        self.feature_network = nn.Sequential(*feature_layers)
        
        # Prior weighting network (takes features + prior as input)
        self.prior_weighting = PriorWeightingNetwork(
            feature_size + 1, hidden_sizes[:-1], dropout_rate
        )
        
    def forward(self, features, prior_value):
        # Get feature-based prediction
        feature_pred = self.feature_network(features)
        
        # Combine features and prior for weight calculation
        weight_input = torch.cat([features, prior_value.unsqueeze(-1)], dim=-1)
        prior_weight = self.prior_weighting(weight_input)
        
        # Blend predictions
        blended = prior_weight * prior_value.unsqueeze(-1) + (1 - prior_weight) * feature_pred
        
        return blended, feature_pred, prior_weight

class OptionsBlendingLearner:
    """Main blending learner for options-anchored framework"""
    
    def __init__(self, config: BlendingConfig = None):
        self.config = config or BlendingConfig()
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        
        # Initialize model based on config
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the blending model based on configuration"""
        
        if self.config.model_type == 'neural_network':
            # Will be initialized when we know feature size
            self.model = None
        elif self.config.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        elif self.config.model_type == 'linear':
            self.model = Ridge(alpha=self.config.regularization)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def prepare_training_data(self, features: pd.DataFrame,
                            priors: List[ImpliedDistribution],
                            targets: np.ndarray,
                            confidence_level: float = 0.68) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for blending learner
        
        Args:
            features: Feature matrix
            priors: List of options priors
            targets: Target values (realized moves/returns)
            confidence_level: Confidence level for prior extraction
            
        Returns:
            Tuple of (feature_matrix, prior_values, targets)
        """
        
        # Extract prior values
        prior_values = []
        for prior in priors:
            # Use expected move as prior value
            prior_value = prior.expected_move_percent / 100  # Convert to decimal return
            prior_values.append(prior_value)
        
        prior_values = np.array(prior_values)
        
        # Validate data alignment
        if len(features) != len(prior_values) or len(features) != len(targets):
            raise ValueError("Features, priors, and targets must have same length")
        
        # Feature scaling if enabled
        if self.config.feature_scaling:
            from sklearn.preprocessing import StandardScaler
            if self.scaler is None:
                self.scaler = StandardScaler()
                feature_matrix = self.scaler.fit_transform(features)
            else:
                feature_matrix = self.scaler.transform(features)
        else:
            feature_matrix = features.values
        
        return feature_matrix, prior_values, targets
    
    def train(self, features: pd.DataFrame,
             priors: List[ImpliedDistribution],
             targets: np.ndarray,
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the blending learner
        
        Args:
            features: Feature matrix
            priors: Options priors
            targets: Target values
            validation_split: Fraction for validation
            
        Returns:
            Training results dictionary
        """
        
        # Prepare data
        feature_matrix, prior_values, targets = self.prepare_training_data(
            features, priors, targets
        )
        
        # Split data
        n_samples = len(feature_matrix)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Time-series split (validation at end)
        train_features = feature_matrix[:n_train]
        train_priors = prior_values[:n_train]
        train_targets = targets[:n_train]
        
        val_features = feature_matrix[n_train:]
        val_priors = prior_values[n_train:]
        val_targets = targets[n_train:]
        
        if self.config.model_type == 'neural_network':
            return self._train_neural_network(
                train_features, train_priors, train_targets,
                val_features, val_priors, val_targets
            )
        else:
            return self._train_sklearn_model(
                train_features, train_priors, train_targets,
                val_features, val_priors, val_targets
            )
    
    def _train_neural_network(self, train_features, train_priors, train_targets,
                            val_features, val_priors, val_targets) -> Dict[str, Any]:
        """Train neural network blending model"""
        
        # Initialize model with correct input size
        feature_size = train_features.shape[1]
        if self.model is None:
            self.model = FeatureBlendingNetwork(
                feature_size=feature_size,
                hidden_sizes=self.config.hidden_sizes,
                dropout_rate=self.config.dropout_rate
            )
        
        # Convert to tensors
        train_features_tensor = torch.FloatTensor(train_features)
        train_priors_tensor = torch.FloatTensor(train_priors)
        train_targets_tensor = torch.FloatTensor(train_targets).unsqueeze(-1)
        
        val_features_tensor = torch.FloatTensor(val_features)
        val_priors_tensor = torch.FloatTensor(val_priors)
        val_targets_tensor = torch.FloatTensor(val_targets).unsqueeze(-1)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            blended_pred, feature_pred, prior_weights = self.model(
                train_features_tensor, train_priors_tensor
            )
            
            loss = criterion(blended_pred, train_targets_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if len(val_features) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_blended, val_feature, val_weights = self.model(
                        val_features_tensor, val_priors_tensor
                    )
                    val_loss = criterion(val_blended, val_targets_tensor)
                    val_losses.append(val_loss.item())
            
            # Early stopping check
            if epoch > 20 and len(val_losses) > 10:
                if val_losses[-1] > np.mean(val_losses[-10:]):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.is_trained = True
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            train_pred, _, train_weights = self.model(train_features_tensor, train_priors_tensor)
            train_mse = mean_squared_error(train_targets, train_pred.numpy().flatten())
            
            if len(val_features) > 0:
                val_pred, _, val_weights = self.model(val_features_tensor, val_priors_tensor)
                val_mse = mean_squared_error(val_targets, val_pred.numpy().flatten())
            else:
                val_mse = None
        
        results = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_epoch': epoch + 1,
            'avg_prior_weight': float(train_weights.mean()),
            'model_type': 'neural_network'
        }
        
        self.training_history.append(results)
        return results
    
    def _train_sklearn_model(self, train_features, train_priors, train_targets,
                           val_features, val_priors, val_targets) -> Dict[str, Any]:
        """Train sklearn-based blending model"""
        
        # For sklearn models, we create composite features [features, prior]
        train_X = np.hstack([train_features, train_priors.reshape(-1, 1)])
        
        # Train model
        self.model.fit(train_X, train_targets)
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(train_X)
        train_mse = mean_squared_error(train_targets, train_pred)
        
        if len(val_features) > 0:
            val_X = np.hstack([val_features, val_priors.reshape(-1, 1)])
            val_pred = self.model.predict(val_X)
            val_mse = mean_squared_error(val_targets, val_pred)
        else:
            val_mse = None
        
        results = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'model_type': self.config.model_type,
            'feature_importance': getattr(self.model, 'feature_importances_', None)
        }
        
        self.training_history.append(results)
        return results
    
    def predict_blend(self, features: pd.DataFrame,
                     priors: List[ImpliedDistribution]) -> List[BlendingResult]:
        """
        Generate blended predictions
        
        Args:
            features: Feature matrix
            priors: Options priors
            
        Returns:
            List of BlendingResult objects
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        if self.scaler is not None:
            feature_matrix = self.scaler.transform(features)
        else:
            feature_matrix = features.values
        
        # Extract prior values
        prior_values = np.array([prior.expected_move_percent / 100 for prior in priors])
        
        results = []
        
        if self.config.model_type == 'neural_network':
            results = self._predict_neural_network(feature_matrix, prior_values)
        else:
            results = self._predict_sklearn_model(feature_matrix, prior_values)
        
        return results
    
    def _predict_neural_network(self, features, priors) -> List[BlendingResult]:
        """Generate predictions using neural network"""
        
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            priors_tensor = torch.FloatTensor(priors)
            
            blended_pred, feature_pred, prior_weights = self.model(
                features_tensor, priors_tensor
            )
            
            # Convert to numpy
            blended = blended_pred.numpy().flatten()
            feature_only = feature_pred.numpy().flatten()
            weights = prior_weights.numpy().flatten()
        
        results = []
        for i in range(len(blended)):
            result = BlendingResult(
                blended_prediction=blended[i],
                prior_weight=weights[i],
                feature_weight=1.0 - weights[i],
                options_prior_value=priors[i],
                feature_prediction=feature_only[i],
                confidence_score=weights[i],  # Use weight as confidence proxy
                blend_components={
                    'prior_contribution': weights[i] * priors[i],
                    'feature_contribution': (1.0 - weights[i]) * feature_only[i]
                },
                metadata={
                    'model_type': 'neural_network',
                    'prior_weight': weights[i],
                    'feature_weight': 1.0 - weights[i]
                }
            )
            results.append(result)
        
        return results
    
    def _predict_sklearn_model(self, features, priors) -> List[BlendingResult]:
        """Generate predictions using sklearn model"""
        
        # Create composite input
        X = np.hstack([features, priors.reshape(-1, 1)])
        blended_pred = self.model.predict(X)
        
        # For sklearn models, we approximate feature-only predictions
        # by using features without prior (set prior to 0)
        X_no_prior = np.hstack([features, np.zeros((len(features), 1))])
        feature_pred = self.model.predict(X_no_prior)
        
        # Estimate weights by comparing predictions
        results = []
        for i in range(len(blended_pred)):
            # Approximate weight calculation
            if abs(priors[i] - feature_pred[i]) > 1e-6:
                weight = (blended_pred[i] - feature_pred[i]) / (priors[i] - feature_pred[i])
                weight = np.clip(weight, 0.0, 1.0)
            else:
                weight = 0.5  # Default if similar values
            
            result = BlendingResult(
                blended_prediction=blended_pred[i],
                prior_weight=weight,
                feature_weight=1.0 - weight,
                options_prior_value=priors[i],
                feature_prediction=feature_pred[i],
                confidence_score=0.5,  # Default confidence for sklearn models
                blend_components={
                    'prior_contribution': weight * priors[i],
                    'feature_contribution': (1.0 - weight) * feature_pred[i]
                },
                metadata={
                    'model_type': self.config.model_type,
                    'approximate_weights': True
                }
            )
            results.append(result)
        
        return results

class FeatureBlender:
    """Utility class for feature combination and preprocessing"""
    
    @staticmethod
    def combine_features(technical_features: pd.DataFrame,
                        macro_features: pd.DataFrame,
                        news_features: pd.DataFrame) -> pd.DataFrame:
        """Combine different feature types into unified matrix"""
        
        combined = pd.DataFrame()
        
        # Add technical features
        if not technical_features.empty:
            tech_cols = [f"tech_{col}" for col in technical_features.columns]
            tech_df = technical_features.copy()
            tech_df.columns = tech_cols
            combined = pd.concat([combined, tech_df], axis=1)
        
        # Add macro features
        if not macro_features.empty:
            macro_cols = [f"macro_{col}" for col in macro_features.columns]
            macro_df = macro_features.copy()
            macro_df.columns = macro_cols
            combined = pd.concat([combined, macro_df], axis=1)
        
        # Add news features
        if not news_features.empty:
            news_cols = [f"news_{col}" for col in news_features.columns]
            news_df = news_features.copy()
            news_df.columns = news_cols
            combined = pd.concat([combined, news_df], axis=1)
        
        return combined
    
    @staticmethod
    def create_feature_groups(features: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by type for analysis"""
        
        groups = {
            'technical': [],
            'macro': [],
            'news': [],
            'other': []
        }
        
        for col in features.columns:
            if col.startswith('tech_'):
                groups['technical'].append(col)
            elif col.startswith('macro_'):
                groups['macro'].append(col)
            elif col.startswith('news_'):
                groups['news'].append(col)
            else:
                groups['other'].append(col)
        
        return groups

class BlendingOptimizer:
    """Optimize blending parameters and architecture"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_blend_weights(self, learner: OptionsBlendingLearner,
                             features: pd.DataFrame,
                             priors: List[ImpliedDistribution],
                             targets: np.ndarray,
                             cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize blending weights using cross-validation
        
        Args:
            learner: Trained blending learner
            features: Feature matrix
            priors: Options priors
            targets: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Optimization results
        """
        
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        feature_matrix, prior_values, _ = learner.prepare_training_data(features, priors, targets)
        
        for train_idx, val_idx in tscv.split(feature_matrix):
            # Split data
            train_features = feature_matrix[train_idx]
            prior_values[train_idx]
            train_targets = targets[train_idx]
            
            val_features = feature_matrix[val_idx]
            prior_values[val_idx]
            val_targets = targets[val_idx]
            
            # Train temporary model
            temp_learner = OptionsBlendingLearner(learner.config)
            temp_learner.scaler = learner.scaler  # Use same scaler
            
            # Convert back to DataFrame for training interface
            train_df = pd.DataFrame(train_features)
            train_priors_list = [priors[i] for i in train_idx]
            
            temp_learner.train(train_df, train_priors_list, train_targets, validation_split=0.0)
            
            # Evaluate
            val_df = pd.DataFrame(val_features)
            val_priors_list = [priors[i] for i in val_idx]
            
            predictions = temp_learner.predict_blend(val_df, val_priors_list)
            pred_values = [pred.blended_prediction for pred in predictions]
            
            score = mean_squared_error(val_targets, pred_values)
            cv_scores.append(score)
        
        optimization_result = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'best_config': learner.config
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result