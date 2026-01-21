"""
Ensemble Learning Framework with Regime-Aware Stacking and Gating

This module implements sophisticated ensemble methods for financial ML with:
1. Base learner training on training folds only
2. Meta-learner training on out-of-fold predictions (no leakage)
3. Regime detection with HMM/Markov switching
4. Options-anchored expert with intelligent gating
5. Guardrails against overfitting and regime changes

Key Features:
- Proper cross-validation for base learners
- Out-of-fold prediction generation
- Regime-aware meta-learning
- Volatility and macro regime detection
- Options market signals integration
- Earnings/IV event-aware gating

References:
- Wolpert (1992) - Stacked Generalization
- Breiman (1996) - Stacked Regressions
- Advances in Financial Machine Learning (López de Prado)
- Machine Learning for Asset Management (Roncalli)
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble learning with regime awareness"""
    # Base learner configuration
    use_linear_models: bool = True
    use_tree_models: bool = True
    use_neural_networks: bool = True
    use_svm: bool = False  # Can be slow on large datasets
    
    # Meta-learner configuration
    meta_learner_type: str = "ridge"  # ridge, linear, nn, xgb
    meta_cv_folds: int = 3  # Cross-validation folds for meta-learner
    
    # Regime detection
    enable_regime_detection: bool = True
    n_regimes: int = 3  # Bear, neutral, bull markets
    regime_features: List[str] = None  # Auto-detected if None
    
    # Options expert
    enable_options_expert: bool = True
    options_lookback_days: int = 21  # Lookback for IV calculations
    earnings_window_days: int = 5  # Days around earnings to boost options expert
    
    # Guardrails
    max_base_learners: int = 8  # Prevent overfitting
    min_samples_per_regime: int = 100  # Minimum samples to estimate regime
    ensemble_weight_bounds: Tuple[float, float] = (0.0, 1.0)  # Weight constraints
    
    # Validation
    validate_oof_predictions: bool = True
    require_positive_weights: bool = False  # Allow negative weights for hedging
    
    def __post_init__(self):
        if self.regime_features is None:
            self.regime_features = [
                'volatility_20', 'volume_ratio', 'momentum_21', 
                'rsi_14', 'vix_proxy'
            ]


class BaseModelTrainer:
    """
    Trains base learners with proper cross-validation and OOF prediction generation.
    
    Ensures no leakage by training base models only on training folds and
    generating out-of-fold predictions for meta-learner training.
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.base_models = self._create_base_models()
        self.trained_models = {}
        self.oof_predictions = {}
        
    def _create_base_models(self) -> Dict[str, BaseEstimator]:
        """Create base model pool based on configuration"""
        models = {}
        
        if self.config.use_linear_models:
            models.update({
                'linear_reg': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.01),
                'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5)
            })
            
        if self.config.use_tree_models:
            models.update({
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42
                ),
                'gradient_boost': GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, random_state=42
                )
            })
            
        if self.config.use_neural_networks:
            models.update({
                'mlp': MLPRegressor(
                    hidden_layer_sizes=(50, 25), max_iter=500, 
                    random_state=42, early_stopping=True
                )
            })
            
        if self.config.use_svm:
            models.update({
                'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
            })
            
        # Limit number of models to prevent overfitting
        if len(models) > self.config.max_base_learners:
            # Keep most diverse models
            priority_order = [
                'ridge', 'random_forest', 'gradient_boost', 'mlp',
                'linear_reg', 'lasso', 'elastic_net', 'svr'
            ]
            selected_models = {}
            for model_name in priority_order:
                if model_name in models:
                    selected_models[model_name] = models[model_name]
                    if len(selected_models) >= self.config.max_base_learners:
                        break
            models = selected_models
            
        logger.info(f"Created {len(models)} base models: {list(models.keys())}")
        return models
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         cv_splitter) -> Dict[str, np.ndarray]:
        """
        Train base models and generate out-of-fold predictions.
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_splitter: Cross-validation splitter (must be fitted)
            
        Returns:
            Dictionary of out-of-fold predictions for each model
        """
        logger.info(f"Training {len(self.base_models)} base models with OOF predictions")
        
        oof_predictions = {}
        trained_models = {}
        
        for model_name, model in self.base_models.items():
            logger.info(f"Training {model_name}...")
            
            # Initialize OOF predictions array
            oof_preds = np.full(len(y_train), np.nan)
            fold_models = []
            
            try:
                # Generate OOF predictions using cross-validation
                for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train)):
                    # Get fold data
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    
                    # Handle missing values
                    if X_fold_train.isna().any().any():
                        X_fold_train = X_fold_train.fillna(X_fold_train.median())
                        X_fold_val = X_fold_val.fillna(X_fold_train.median())
                    
                    # Clone and train model
                    fold_model = clone(model)
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Generate out-of-fold predictions
                    fold_preds = fold_model.predict(X_fold_val)
                    oof_preds[val_idx] = fold_preds
                    
                    fold_models.append(fold_model)
                    
                    logger.debug(f"  Fold {fold_idx}: {len(train_idx)} train, "
                               f"{len(val_idx)} val samples")
                
                # Validate OOF predictions
                valid_preds = ~np.isnan(oof_preds)
                if valid_preds.sum() < len(y_train) * 0.8:
                    logger.warning(f"{model_name}: Only {valid_preds.sum()}/{len(y_train)} "
                                 f"valid OOF predictions")
                
                # Calculate OOF performance
                if valid_preds.sum() > 0:
                    oof_mse = mean_squared_error(
                        y_train[valid_preds], oof_preds[valid_preds]
                    )
                    oof_mae = mean_absolute_error(
                        y_train[valid_preds], oof_preds[valid_preds]
                    )
                    logger.info(f"  {model_name} OOF RMSE: {np.sqrt(oof_mse):.6f}, "
                              f"MAE: {oof_mae:.6f}")
                
                oof_predictions[model_name] = oof_preds
                trained_models[model_name] = fold_models
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Store results
        self.oof_predictions = oof_predictions
        self.trained_models = trained_models
        
        logger.info(f"Successfully trained {len(oof_predictions)} base models")
        return oof_predictions
    
    def predict_with_base_models(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions from all trained base models"""
        if not self.trained_models:
            raise ValueError("No trained models available. Call train_base_models first.")
        
        predictions = {}
        
        for model_name, fold_models in self.trained_models.items():
            # Average predictions across folds
            fold_predictions = []
            
            for fold_model in fold_models:
                X_test_clean = X_test.fillna(X_test.median()) if X_test.isna().any().any() else X_test
                fold_pred = fold_model.predict(X_test_clean)
                fold_predictions.append(fold_pred)
            
            # Average across folds
            avg_prediction = np.mean(fold_predictions, axis=0)
            predictions[model_name] = avg_prediction
            
        return predictions


class RegimeDetector:
    """
    Detects market regimes using Hidden Markov Models.
    
    Estimates regimes (bear/neutral/bull) on training data only,
    then generates regime probabilities for meta-learner.
    """
    
    def __init__(self, n_regimes: int = 3, regime_features: List[str] = None):
        self.n_regimes = n_regimes
        self.regime_features = regime_features or [
            'volatility_20', 'momentum_21', 'rsi_14'
        ]
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_names = self._get_regime_names()
        
    def _get_regime_names(self) -> List[str]:
        """Get regime names based on number of regimes"""
        if self.n_regimes == 2:
            return ['low_vol', 'high_vol']
        elif self.n_regimes == 3:
            return ['bear', 'neutral', 'bull']
        elif self.n_regimes == 4:
            return ['bear', 'low_vol', 'high_vol', 'bull']
        else:
            return [f'regime_{i}' for i in range(self.n_regimes)]
    
    def fit(self, X_train: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit HMM model on training data to detect market regimes.
        
        Args:
            X_train: Training features containing regime indicators
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting HMM regime detector with {self.n_regimes} regimes")
        
        # Extract regime features
        available_features = [f for f in self.regime_features if f in X_train.columns]
        if not available_features:
            logger.warning(f"No regime features found in data. Available: {X_train.columns.tolist()}")
            # Fall back to basic features
            available_features = [c for c in X_train.columns if any(
                keyword in c.lower() for keyword in ['volatility', 'momentum', 'rsi', 'returns']
            )][:3]
        
        self.regime_features = available_features
        logger.info(f"Using regime features: {self.regime_features}")
        
        if len(self.regime_features) == 0:
            logger.error("No suitable regime features found")
            raise ValueError("Cannot detect regimes without suitable features")
        
        # Prepare regime feature matrix
        regime_data = X_train[self.regime_features].copy()
        
        # Handle missing values
        regime_data = regime_data.fillna(regime_data.median())
        
        # Scale features
        regime_data_scaled = self.scaler.fit_transform(regime_data)
        
        # Fit HMM model
        try:
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=42,
                n_iter=100
            )
            
            # Fit model
            self.hmm_model.fit(regime_data_scaled)
            
            # Validate model
            log_likelihood = self.hmm_model.score(regime_data_scaled)
            logger.info(f"HMM model fitted. Log-likelihood: {log_likelihood:.2f}")
            
            # Get regime assignments for interpretation
            regime_states = self.hmm_model.predict(regime_data_scaled)
            unique_regimes = np.unique(regime_states)
            logger.info(f"Detected {len(unique_regimes)} active regimes in training data")
            
            for regime_idx in unique_regimes:
                regime_mask = regime_states == regime_idx
                regime_count = regime_mask.sum()
                regime_pct = regime_count / len(regime_states) * 100
                logger.info(f"  {self.regime_names[regime_idx]}: {regime_count} samples ({regime_pct:.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to fit HMM model: {e}")
            # Fall back to simple volatility-based regimes
            self._fit_simple_regime_detector(regime_data)
        
        return self
    
    def _fit_simple_regime_detector(self, regime_data: pd.DataFrame):
        """Fall back to simple volatility-based regime detection"""
        logger.warning("Falling back to simple volatility-based regime detection")
        
        # Use volatility as main regime indicator
        vol_col = None
        for col in regime_data.columns:
            if 'volatility' in col.lower() or 'vol' in col.lower():
                vol_col = col
                break
        
        if vol_col is None:
            # Use returns volatility as proxy
            vol_col = regime_data.columns[0]
            logger.warning(f"Using {vol_col} as volatility proxy for regime detection")
        
        volatility = regime_data[vol_col]
        
        # Define regime thresholds based on volatility quantiles
        if self.n_regimes == 2:
            self.vol_threshold = volatility.median()
        elif self.n_regimes == 3:
            self.vol_thresholds = [
                volatility.quantile(0.33),
                volatility.quantile(0.67)
            ]
        
        self.hmm_model = None  # Mark as simple detector
        logger.info("Simple regime detector fitted based on volatility quantiles")
    
    def predict_regime_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities for given data.
        
        Args:
            X: Features for regime prediction
            
        Returns:
            DataFrame with regime probabilities
        """
        if self.hmm_model is None and not hasattr(self, 'vol_thresholds'):
            raise ValueError("Regime detector not fitted. Call fit() first.")
        
        # Extract regime features
        regime_data = X[self.regime_features].copy()
        regime_data = regime_data.fillna(regime_data.median())
        
        if self.hmm_model is not None:
            # Use HMM model
            regime_data_scaled = self.scaler.transform(regime_data)
            regime_probs = self.hmm_model.predict_proba(regime_data_scaled)
        else:
            # Use simple volatility-based detection
            regime_probs = self._predict_simple_regimes(regime_data)
        
        # Create DataFrame with regime probabilities
        regime_columns = [f'regime_prob_{name}' for name in self.regime_names]
        regime_df = pd.DataFrame(
            regime_probs, 
            index=X.index, 
            columns=regime_columns
        )
        
        return regime_df
    
    def _predict_simple_regimes(self, regime_data: pd.DataFrame) -> np.ndarray:
        """Simple volatility-based regime prediction"""
        vol_col = self.regime_features[0]
        volatility = regime_data[vol_col]
        
        n_samples = len(volatility)
        regime_probs = np.zeros((n_samples, self.n_regimes))
        
        if self.n_regimes == 2:
            # Low vol / High vol
            low_vol_mask = volatility <= self.vol_threshold
            regime_probs[low_vol_mask, 0] = 1.0
            regime_probs[~low_vol_mask, 1] = 1.0
        elif self.n_regimes == 3:
            # Bear / Neutral / Bull based on volatility
            bear_mask = volatility >= self.vol_thresholds[1]
            bull_mask = volatility <= self.vol_thresholds[0]
            neutral_mask = ~(bear_mask | bull_mask)
            
            regime_probs[bear_mask, 0] = 1.0  # Bear (high vol)
            regime_probs[neutral_mask, 1] = 1.0  # Neutral
            regime_probs[bull_mask, 2] = 1.0  # Bull (low vol)
        
        return regime_probs


class OptionsExpert:
    """
    Options-anchored expert that specializes in high IV and earnings periods.
    
    Uses implied volatility signals and earnings calendars to provide
    specialized predictions during options-relevant periods.
    """
    
    def __init__(self, lookback_days: int = 21, earnings_window_days: int = 5):
        self.lookback_days = lookback_days
        self.earnings_window_days = earnings_window_days
        self.model = Ridge(alpha=1.0)  # Conservative linear model
        self.iv_scaler = StandardScaler()
        self.is_trained = False
        
    def create_options_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create options-specific features from base features.
        
        Args:
            X: Base features
            
        Returns:
            DataFrame with options-relevant features
        """
        options_features = pd.DataFrame(index=X.index)
        
        # Implied volatility proxy (realized volatility)
        vol_cols = [c for c in X.columns if 'volatility' in c.lower()]
        if vol_cols:
            main_vol = X[vol_cols[0]]
            options_features['iv_proxy'] = main_vol
            options_features['iv_rank'] = main_vol.rolling(252).rank(pct=True)
            options_features['iv_percentile'] = main_vol.rolling(60).rank(pct=True)
        
        # Volatility spike detection
        if vol_cols:
            vol_ma = main_vol.rolling(self.lookback_days).mean()
            options_features['vol_spike'] = (main_vol / vol_ma - 1).clip(-1, 2)
        
        # Return momentum near high IV
        ret_cols = [c for c in X.columns if 'returns' in c.lower()]
        if ret_cols:
            returns = X[ret_cols[0]]
            options_features['returns_1d'] = returns
            if vol_cols:
                # Sharpe-like ratio
                vol_window = main_vol.rolling(self.lookback_days).mean()
                options_features['risk_adj_returns'] = returns / (vol_window + 1e-8)
        
        # Volume/momentum features
        vol_ratio_cols = [c for c in X.columns if 'volume' in c.lower()]
        if vol_ratio_cols:
            options_features['volume_signal'] = X[vol_ratio_cols[0]]
        
        momentum_cols = [c for c in X.columns if 'momentum' in c.lower()]
        if momentum_cols:
            options_features['momentum'] = X[momentum_cols[0]]
        
        # Technical indicators relevant for options
        rsi_cols = [c for c in X.columns if 'rsi' in c.lower()]
        if rsi_cols:
            rsi = X[rsi_cols[0]]
            options_features['rsi'] = rsi
            options_features['rsi_extreme'] = ((rsi > 70) | (rsi < 30)).astype(float)
        
        # Price vs moving average (support/resistance)
        sma_cols = [c for c in X.columns if 'sma' in c.lower()]
        if sma_cols:
            price_vs_sma = [c for c in X.columns if 'price_vs' in c]
            if price_vs_sma:
                options_features['price_vs_ma'] = X[price_vs_sma[0]]
        
        # Fill missing values
        options_features = options_features.fillna(method='ffill').fillna(0)
        
        logger.info(f"Created {options_features.shape[1]} options-specific features")
        return options_features
    
    def detect_earnings_periods(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Detect likely earnings periods based on patterns.
        
        In practice, this would use an earnings calendar API.
        For now, we'll use quarterly patterns as a proxy.
        """
        # Simple earnings proxy: end of quarters + some randomness
        earnings_mask = pd.Series(False, index=dates)
        
        for date in dates:
            # Quarterly earnings typically in Jan, Apr, Jul, Oct
            if date.month in [1, 4, 7, 10]:
                # Higher probability in first 3 weeks of these months
                if 1 <= date.day <= 21:
                    # Add some randomness to simulate different companies
                    if np.random.random() < 0.15:  # 15% chance on any day
                        earnings_mask[date] = True
        
        return earnings_mask
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'OptionsExpert':
        """
        Train options expert on training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Self for method chaining
        """
        logger.info("Training options expert...")
        
        # Create options-specific features
        options_features = self.create_options_features(X_train)
        
        # Scale features
        options_scaled = self.iv_scaler.fit_transform(options_features)
        
        # Train model
        self.model.fit(options_scaled, y_train)
        self.is_trained = True
        
        # Calculate training performance
        train_pred = self.model.predict(options_scaled)
        train_mse = mean_squared_error(y_train, train_pred)
        logger.info(f"Options expert training RMSE: {np.sqrt(train_mse):.6f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from options expert"""
        if not self.is_trained:
            raise ValueError("Options expert not trained. Call fit() first.")
        
        options_features = self.create_options_features(X)
        options_scaled = self.iv_scaler.transform(options_features)
        
        return self.model.predict(options_scaled)
    
    def get_confidence_weights(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get confidence weights for options expert based on market conditions.
        
        Higher weights during:
        - High IV periods
        - Earnings windows
        - Extreme RSI levels
        """
        options_features = self.create_options_features(X)
        weights = np.ones(len(X))
        
        # Boost weight during high IV
        if 'iv_percentile' in options_features.columns:
            iv_percentile = options_features['iv_percentile']
            high_iv_mask = iv_percentile > 0.8
            weights[high_iv_mask] *= 2.0
        
        # Boost weight during volatility spikes
        if 'vol_spike' in options_features.columns:
            vol_spike = options_features['vol_spike']
            spike_mask = vol_spike > 0.5
            weights[spike_mask] *= 1.5
        
        # Boost weight during extreme RSI
        if 'rsi_extreme' in options_features.columns:
            extreme_rsi = options_features['rsi_extreme']
            weights[extreme_rsi == 1] *= 1.3
        
        # Simulate earnings period boost
        earnings_mask = self.detect_earnings_periods(X.index)
        weights[earnings_mask] *= 2.5
        
        # Normalize weights
        weights = np.clip(weights, 0.1, 5.0)  # Reasonable bounds
        
        return weights