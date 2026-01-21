from __future__ import annotations

from typing import Dict
import logging
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class SimpleRegressor:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def train_history_models(x_growth: np.ndarray,
                         y_growth: np.ndarray,
                         x_margin: np.ndarray,
                         y_margin: np.ndarray,
                         x_capex: np.ndarray,
                         y_capex: np.ndarray,
                         x_nwc: np.ndarray,
                         y_nwc: np.ndarray) -> Dict[str,
                                                    SimpleRegressor]:
    """
    Train simple models for growth and ratios from historical supervised data.

    Args:
        x_growth: Features for growth model
        y_growth: Target for growth model
        x_margin: Features for margin model
        y_margin: Target for margin model
        x_capex: Features for capex model
        y_capex: Target for capex model
        x_nwc: Features for NWC model
        y_nwc: Target for NWC model

    Returns:
        Dictionary of trained models
    """
    # Growth: ElasticNet for robustness
    g = SimpleRegressor(ElasticNet(alpha=0.0001, l1_ratio=0.2, max_iter=8000))
    if len(x_growth):
        g.fit(x_growth, y_growth)

    # Margins: RandomForest (nonlinear), fallback to ElasticNet if
    # insufficient samples
    def rf_or_en(x_features, y_target):
        if len(x_features) >= 6:
            m = SimpleRegressor(
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42))
        else:
            m = SimpleRegressor(
                ElasticNet(
                    alpha=0.0001,
                    l1_ratio=0.2,
                    max_iter=8000))
        if len(x_features):
            m.fit(x_features, y_target)
        return m

    em = rf_or_en(x_margin, y_margin)
    cx = rf_or_en(x_capex, y_capex)
    nw = rf_or_en(x_nwc, y_nwc)
    return {
        "growth": g,
        "ebit_margin": em,
        "capex_to_rev": cx,
        "nwc_to_rev": nw}


def one_step_predictions(models: Dict[str,
                                      SimpleRegressor],
                         x_growth_input,
                         x_margin_input,
                         x_capex_input,
                         x_nwc_input) -> Dict[str,
                                              float]:
    """
    Make predictions for a single step using trained models

    Args:
        models: Dictionary of trained models
        x_growth_input: Input features for growth model
        x_margin_input: Input features for margin model
        x_capex_input: Input features for capex model
        x_nwc_input: Input features for NWC model

    Returns:
        Dictionary of predicted values
    """
    out = {}
    try:
        out["rev_cagr"] = float(
            models["growth"].predict(
                x_growth_input.reshape(
                    1, -1))[0])
    except Exception:
        pass

    feature_map = [
        ("ebit_margin", x_margin_input),
        ("capex_to_rev", x_capex_input),
        ("nwc_to_rev", x_nwc_input)
    ]

    for key, x_input in feature_map:
        try:
            out[key] = float(models[key].predict(x_input.reshape(1, -1))[0])
        except Exception:
            pass

    return out


class ModelTrainer:
    """
    Advanced model training pipeline with multiple architectures for continuous retraining
    """
    
    def __init__(self):
        self.scalers = {}
        self.trained_models = {}
        self.training_history = []
        
        # Model configurations
        self.model_configs = {
            'enhanced_tf_v2.1': {
                'type': 'ensemble',
                'base_models': ['random_forest', 'logistic_regression'],
                'params': {
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'random_state': 42
                    },
                    'logistic_regression': {
                        'C': 1.0,
                        'random_state': 42
                    }
                }
            },
            'lstm_ensemble_v1.5': {
                'type': 'time_series',
                'sequence_length': 20,
                'params': {
                    'units': 50,
                    'dropout': 0.2,
                    'epochs': 50
                }
            },
            'transformer_v1.2': {
                'type': 'attention',
                'attention_heads': 8,
                'params': {
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'dropout': 0.1
                }
            }
        }
    
    def train_model(self, 
                   model_type: str,
                   X_train: np.ndarray,
                   y_direction_train: np.ndarray,
                   y_return_train: np.ndarray,
                   X_val: np.ndarray = None,
                   y_direction_val: np.ndarray = None,
                   y_return_val: np.ndarray = None) -> Dict:
        """
        Train a specific model type with fresh data
        """
        
        logger.info(f"Training {model_type} model...")
        
        try:
            if model_type not in self.model_configs:
                # Use ensemble as default
                model_type = 'enhanced_tf_v2.1'
                logger.warning(f"Unknown model type, using {model_type}")
            
            config = self.model_configs[model_type]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val) if X_val is not None else None
            
            # Store scaler
            self.scalers[model_type] = scaler
            
            # Train based on model type
            if config['type'] == 'ensemble':
                model = self._train_ensemble_model(config, X_train_scaled, y_direction_train, y_return_train)
            else:
                # For LSTM and Transformer, use enhanced RF for demo
                model = self._train_enhanced_model(config, X_train_scaled, y_direction_train, y_return_train)
            
            # Validate model
            metrics = {}
            if X_val is not None and y_direction_val is not None:
                metrics = self._validate_model(model, X_val_scaled, y_direction_val, y_return_val)
            
            # Store trained model
            self.trained_models[model_type] = {
                'model': model,
                'scaler': scaler,
                'config': config,
                'timestamp': datetime.now()
            }
            
            logger.info(f"{model_type} training completed successfully")
            
            return {
                'model': model,
                'scaler': scaler,
                'config': config,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            return {'error': str(e)}
    
    def _train_ensemble_model(self, config: Dict, X_train: np.ndarray, 
                            y_direction: np.ndarray, y_return: np.ndarray) -> Dict:
        """Train ensemble model (Random Forest + Logistic Regression)"""
        
        models = {}
        
        # Direction classifier
        rf_classifier = RandomForestClassifier(**config['params']['random_forest'])
        rf_classifier.fit(X_train, y_direction)
        models['direction_rf'] = rf_classifier
        
        lr_classifier = LogisticRegression(**config['params']['logistic_regression'])
        lr_classifier.fit(X_train, y_direction)
        models['direction_lr'] = lr_classifier
        
        # Return regressor
        rf_regressor = RandomForestRegressor(**config['params']['random_forest'])
        rf_regressor.fit(X_train, y_return)
        models['return_rf'] = rf_regressor
        
        lr_regressor = LinearRegression()
        lr_regressor.fit(X_train, y_return)
        models['return_lr'] = lr_regressor
        
        return models
    
    def _train_enhanced_model(self, config: Dict, X_train: np.ndarray,
                            y_direction: np.ndarray, y_return: np.ndarray) -> Dict:
        """Train enhanced model (for LSTM/Transformer approximation)"""
        
        models = {}
        
        # Enhanced parameters for LSTM/Transformer
        n_estimators = 200 if config['type'] == 'attention' else 150
        max_depth = 15 if config['type'] == 'attention' else 12
        
        # Direction prediction
        direction_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        direction_model.fit(X_train, y_direction)
        models['direction'] = direction_model
        
        # Return prediction
        return_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        return_model.fit(X_train, y_return)
        models['return'] = return_model
        
        return models
    
    def _validate_model(self, model: Dict, X_val: np.ndarray,
                       y_direction_val: np.ndarray, y_return_val: np.ndarray) -> Dict:
        """Validate trained model on validation set"""
        
        metrics = {}
        
        try:
            # Get predictions based on model structure
            if 'direction_rf' in model:  # Ensemble model
                dir_pred_rf = model['direction_rf'].predict(X_val)
                dir_pred_lr = model['direction_lr'].predict(X_val)
                dir_pred = (dir_pred_rf + dir_pred_lr) / 2  # Average ensemble
                dir_pred = (dir_pred > 0.5).astype(int)
                
                ret_pred_rf = model['return_rf'].predict(X_val)
                ret_pred_lr = model['return_lr'].predict(X_val)
                ret_pred = (ret_pred_rf + ret_pred_lr) / 2
                
            else:  # Enhanced model
                dir_pred = model['direction'].predict(X_val)
                ret_pred = model['return'].predict(X_val)
            
            # Calculate metrics
            if y_direction_val is not None:
                metrics['directional_accuracy'] = accuracy_score(y_direction_val, dir_pred)
            
            if y_return_val is not None:
                metrics['return_rmse'] = np.sqrt(mean_squared_error(y_return_val, ret_pred))
                metrics['return_mae'] = mean_absolute_error(y_return_val, ret_pred)
                
                # Information Coefficient
                correlation = np.corrcoef(y_return_val, ret_pred)[0, 1]
                metrics['information_coefficient'] = correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            metrics['error'] = str(e)
        
        return metrics


class ModelEvaluator:
    """
    Model evaluation and performance assessment for continuous training
    """
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_model(self, model: Dict, X_test: np.ndarray, 
                      y_direction_test: np.ndarray, y_return_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation for continuous training loop
        """
        
        try:
            # Generate predictions
            if 'direction_rf' in model:  # Ensemble model
                dir_pred_rf = model['direction_rf'].predict(X_test)
                dir_pred_lr = model['direction_lr'].predict(X_test)
                dir_pred = (dir_pred_rf + dir_pred_lr) / 2
                dir_pred = (dir_pred > 0.5).astype(int)
                
                ret_pred_rf = model['return_rf'].predict(X_test)
                ret_pred_lr = model['return_lr'].predict(X_test)
                ret_pred = (ret_pred_rf + ret_pred_lr) / 2
                
            else:  # Enhanced model
                dir_pred = model['direction'].predict(X_test)
                ret_pred = model['return'].predict(X_test)
            
            # Calculate metrics
            metrics = {
                'directional_accuracy': accuracy_score(y_direction_test, dir_pred),
                'return_rmse': np.sqrt(mean_squared_error(y_return_test, ret_pred)),
                'return_mae': mean_absolute_error(y_return_test, ret_pred),
            }
            
            # Information Coefficient
            ic = np.corrcoef(y_return_test, ret_pred)[0, 1]
            metrics['information_coefficient'] = ic if not np.isnan(ic) else 0.0
            
            # Win rate
            metrics['win_rate'] = np.mean(ret_pred > 0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            return {'error': str(e)}
