"""
HF Extension for DCF Family
============================

Adds ML-based fair value and regime classification to DCF features.

This module is part of the DCF_FAMILY, NOT a separate family.

Structure:
- compute_hf_fair_value_model(): Predict fair value using light ML
- compute_hf_regime_classifier(): Classify trend regimes
- compute_hf_extension(): Main entry point combining both

Training Rules:
- Train fresh per fold (no leakage)
- Use only training window data
- Input: raw DCF + ML_FRAMEWORK + microstructure features
- NO future data, NO cross-futures, NO manual tuning
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class DCFHFExtension:
    """
    HF Extension block for DCF Family
    
    Provides:
    1. Fair Value Model: ML-predicted intrinsic value
    2. Regime Classifier: Trend/mean-reversion regime detection
    
    Uses ONLY:
    - Raw DCF features
    - ML_FRAMEWORK technical features  
    - Microstructure features (optional)
    - Quantile features (optional)
    """
    
    def __init__(
        self,
        model_type: str = 'lgbm',
        retrain_frequency: int = 63,
        min_training_samples: int = 252
    ):
        """
        Initialize HF extension
        
        Args:
            model_type: 'lgbm', 'mlp', 'elasticnet', or 'gbm'
            retrain_frequency: Retrain model every N samples
            min_training_samples: Minimum samples needed for training
        """
        self.model_type = model_type if LGBM_AVAILABLE else 'gbm'
        self.retrain_frequency = retrain_frequency
        self.min_training_samples = min_training_samples
        
        self.fair_value_model = None
        self.regime_model = None
        self.scaler = None
        self.last_train_idx = -1
        
        logger.info(f"DCF HF Extension initialized with model_type={self.model_type}")
    
    def _create_fair_value_model(self):
        """Create fair value regression model (light ML only)"""
        if self.model_type == 'lgbm' and LGBM_AVAILABLE:
            return LGBMRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                max_iter=100,
                early_stopping=True,
                random_state=42,
                verbose=False
            )
        elif self.model_type == 'elasticnet':
            return ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42
            )
        else:  # gbm fallback
            return GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
    
    def _create_regime_classifier(self):
        """Create regime classification model"""
        if self.model_type == 'lgbm' and LGBM_AVAILABLE:
            return LGBMClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                num_leaves=7,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation='relu',
                max_iter=100,
                early_stopping=True,
                random_state=42,
                verbose=False
            )
        else:  # Use simple logistic as fallback
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=500,
                random_state=42
            )
    
    def _derive_regime_labels(self, raw_features: pd.DataFrame) -> pd.Series:
        """
        Derive regime labels from raw indicators (self-supervised)
        
        Regimes:
        0 = Mean Reversion (low momentum, high z-score)
        1 = Trending Up (positive slope, positive momentum)
        2 = Trending Down (negative slope, negative momentum)
        3 = Choppy/Sideways (low slope, low momentum)
        
        Args:
            raw_features: DataFrame with dcf_trend_slope_1m, dcf_mom_1m, dcf_zscore_1m
            
        Returns:
            Series with regime labels (0-3)
        """
        # Extract relevant features
        slope = raw_features.get('dcf_trend_slope_1m', pd.Series(0, index=raw_features.index))
        momentum = raw_features.get('dcf_mom_1m', pd.Series(1, index=raw_features.index)) - 1.0
        zscore = raw_features.get('dcf_zscore_1m', pd.Series(0, index=raw_features.index))
        
        # Initialize regime as choppy (3)
        regime = pd.Series(3, index=raw_features.index)
        
        # Mean reversion: high absolute z-score, low momentum
        mean_reversion_mask = (np.abs(zscore) > 1.5) & (np.abs(momentum) < 0.02)
        regime[mean_reversion_mask] = 0
        
        # Trending up: positive slope and momentum
        trending_up_mask = (slope > 0.001) & (momentum > 0.02)
        regime[trending_up_mask] = 1
        
        # Trending down: negative slope and momentum
        trending_down_mask = (slope < -0.001) & (momentum < -0.02)
        regime[trending_down_mask] = 2
        
        return regime
    
    def _prepare_features(
        self,
        raw_dcf: pd.DataFrame,
        ml_framework: Optional[pd.DataFrame] = None,
        microstructure: Optional[pd.DataFrame] = None,
        quantile: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Combine input features for HF models
        
        Args:
            raw_dcf: Raw DCF features (required)
            ml_framework: ML_FRAMEWORK technical features (optional)
            microstructure: Microstructure features (optional)
            quantile: Quantile forecast features (optional)
            
        Returns:
            Combined feature DataFrame
        """
        features = raw_dcf.copy()
        
        if ml_framework is not None and not ml_framework.empty:
            features = features.join(ml_framework, how='left')
        
        if microstructure is not None and not microstructure.empty:
            # Only use a subset of microstructure features to avoid overfitting
            micro_subset = [c for c in microstructure.columns if any(
                x in c for x in ['volume_surge', 'ofi_proxy', 'overnight_vol', 'impact_ratio']
            )]
            if micro_subset:
                features = features.join(microstructure[micro_subset], how='left')
        
        if quantile is not None and not quantile.empty:
            # Use median quantile prediction as feature
            q50_cols = [c for c in quantile.columns if 'q50' in c or 'q_50' in c]
            if q50_cols:
                features = features.join(quantile[q50_cols[:1]], how='left')
        
        # Fill NaNs
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def compute_hf_fair_value_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        is_training: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Compute fair value predictions using light ML
        
        Args:
            features: Combined features (DCF + ML + micro)
            target: Close price (for training) or None (for prediction)
            is_training: Whether we're in training mode
            
        Returns:
            Dictionary with fair value features
        """
        result = {}
        
        # Ensure we have enough data
        if len(features) < self.min_training_samples:
            logger.warning(f"Insufficient data for HF fair value: {len(features)} < {self.min_training_samples}")
            result['hf_dcf_fairvalue'] = pd.Series(np.nan, index=features.index)
            result['hf_dcf_mispricing_score'] = pd.Series(0.0, index=features.index)
            result['hf_dcf_confidence'] = pd.Series(0.0, index=features.index)
            return result
        
        # Check if we need to retrain
        should_train = (
            is_training and
            (self.fair_value_model is None or
             len(features) - self.last_train_idx >= self.retrain_frequency)
        )
        
        if should_train and target is not None:
            # Train model on available data
            X_train = features.values
            y_train = target.values
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Create and train model
            self.fair_value_model = self._create_fair_value_model()
            self.fair_value_model.fit(X_train_scaled, y_train)
            self.last_train_idx = len(features)
            
            logger.info(f"Trained fair value model on {len(X_train)} samples")
        
        # Predict fair value
        if self.fair_value_model is not None and self.scaler is not None:
            X_pred = self.scaler.transform(features.values)
            fair_value_pred = self.fair_value_model.predict(X_pred)
            
            result['hf_dcf_fairvalue'] = pd.Series(fair_value_pred, index=features.index)
            
            # Mispricing score (actual - predicted)
            if target is not None:
                result['hf_dcf_mispricing_score'] = target - result['hf_dcf_fairvalue']
            else:
                result['hf_dcf_mispricing_score'] = pd.Series(0.0, index=features.index)
            
            # Confidence (simplified: use R² on training data as proxy)
            if is_training and target is not None:
                train_r2 = r2_score(target.values, fair_value_pred)
                confidence = max(0.0, min(1.0, train_r2))
            else:
                confidence = 0.5  # Default confidence
            
            result['hf_dcf_confidence'] = pd.Series(confidence, index=features.index)
        else:
            # No model available
            result['hf_dcf_fairvalue'] = pd.Series(np.nan, index=features.index)
            result['hf_dcf_mispricing_score'] = pd.Series(0.0, index=features.index)
            result['hf_dcf_confidence'] = pd.Series(0.0, index=features.index)
        
        return result
    
    def compute_hf_regime_classifier(
        self,
        features: pd.DataFrame,
        raw_dcf: pd.DataFrame,
        is_training: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Classify trend regime using light ML
        
        Args:
            features: Combined features for prediction
            raw_dcf: Raw DCF features for label derivation
            is_training: Whether we're in training mode
            
        Returns:
            Dictionary with regime features
        """
        result = {}
        
        # Derive regime labels from raw features
        regime_labels = self._derive_regime_labels(raw_dcf)
        
        # Ensure we have enough data
        if len(features) < self.min_training_samples:
            result['hf_trend_regime'] = pd.Series(3, index=features.index)  # Default to choppy
            result['hf_trend_confidence'] = pd.Series(0.0, index=features.index)
            return result
        
        # Check if we need to retrain
        should_train = (
            is_training and
            (self.regime_model is None or
             len(features) - self.last_train_idx >= self.retrain_frequency)
        )
        
        if should_train:
            # Train classifier
            X_train = features.values
            y_train = regime_labels.values
            
            # Create and train model
            self.regime_model = self._create_regime_classifier()
            self.regime_model.fit(X_train, y_train)
            
            logger.info(f"Trained regime classifier on {len(X_train)} samples")
        
        # Predict regime
        if self.regime_model is not None:
            X_pred = features.values
            regime_pred = self.regime_model.predict(X_pred)
            
            result['hf_trend_regime'] = pd.Series(regime_pred, index=features.index)
            
            # Confidence (max probability)
            if hasattr(self.regime_model, 'predict_proba'):
                regime_proba = self.regime_model.predict_proba(X_pred)
                confidence = np.max(regime_proba, axis=1)
            else:
                confidence = np.ones(len(features)) * 0.5
            
            result['hf_trend_confidence'] = pd.Series(confidence, index=features.index)
        else:
            result['hf_trend_regime'] = pd.Series(3, index=features.index)
            result['hf_trend_confidence'] = pd.Series(0.0, index=features.index)
        
        return result
    
    def compute_hf_extension(
        self,
        raw_dcf: pd.DataFrame,
        target: Optional[pd.Series] = None,
        ml_framework: Optional[pd.DataFrame] = None,
        microstructure: Optional[pd.DataFrame] = None,
        quantile: Optional[pd.DataFrame] = None,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Main entry point: Compute all HF extension features
        
        Args:
            raw_dcf: Raw DCF features (required)
            target: Close price for training (optional)
            ml_framework: ML_FRAMEWORK features (optional)
            microstructure: Microstructure features (optional)
            quantile: Quantile forecast features (optional)
            is_training: Whether we're in training mode
            
        Returns:
            DataFrame with HF extension features:
            - hf_dcf_fairvalue
            - hf_dcf_mispricing_score
            - hf_dcf_confidence
            - hf_trend_regime
            - hf_trend_confidence
        """
        # Prepare combined features
        features = self._prepare_features(
            raw_dcf=raw_dcf,
            ml_framework=ml_framework,
            microstructure=microstructure,
            quantile=quantile
        )
        
        # Compute fair value features
        fair_value_features = self.compute_hf_fair_value_model(
            features=features,
            target=target,
            is_training=is_training
        )
        
        # Compute regime features
        regime_features = self.compute_hf_regime_classifier(
            features=features,
            raw_dcf=raw_dcf,
            is_training=is_training
        )
        
        # Combine all HF features
        hf_df = pd.DataFrame(index=raw_dcf.index)
        
        # Add fair value features
        for key, series in fair_value_features.items():
            hf_df[key] = series
        
        # Add regime features
        for key, series in regime_features.items():
            hf_df[key] = series
        
        return hf_df


def create_dcf_hf_extension(
    model_type: str = 'lgbm',
    retrain_frequency: int = 63
) -> DCFHFExtension:
    """
    Factory function to create DCF HF Extension
    
    Args:
        model_type: 'lgbm', 'mlp', 'elasticnet', or 'gbm'
        retrain_frequency: Retrain every N samples
        
    Returns:
        DCFHFExtension instance
    """
    return DCFHFExtension(
        model_type=model_type,
        retrain_frequency=retrain_frequency
    )
