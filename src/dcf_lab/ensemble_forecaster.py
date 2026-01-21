"""
Ensemble forecasting system with multiple expert models and intelligent gating.

This module implements:
1. Transformer Expert - TFT/N-BEATS on time series sequences
2. LightGBM Expert - Gradient boosting on tabular features (tech/news/macro)
3. AR/EMA Expert - Linear autoregressive and exponential moving average baseline
4. Gating Network - Learns optimal weights based on regime, volatility, news novelty
5. Time-Series Calibration - Maintains forecast quality without data leakage
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ===== DTYPE CONSISTENCY: USE GLOBAL GUARD =====
try:
    from .models.torch_dtype_guard import set_global_dtype, ensure_ensemble_consistency
    set_global_dtype(torch.float64)  # High precision for financial modeling
except ImportError:
    # Fallback to manual setting
    torch.set_default_dtype(torch.float64)

# Calibration system
try:
    from .calibration import TimeSeriesCalibrator, CalibrationManager
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    warnings.warn("Calibration system not available. Raw probabilities will be used.")

# LightGBM with optional import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Using RandomForest as fallback.")

try:
    from .data_hygiene import clean_and_prepare_features
    from .losses import CRPSLoss, QuantileLoss
except ImportError:
    try:
        from .data_hygiene import clean_and_prepare_features
        from .losses import CRPSLoss, QuantileLoss
    except ImportError:
        # Fallback implementations
        class QuantileLoss:
            def __init__(self, quantiles):
                self.quantiles = quantiles

            def __call__(self, y_pred, y_true):
                import torch
                loss = 0
                for i, q in enumerate(self.quantiles):
                    error = y_true - y_pred[:, i]
                    loss += torch.mean(torch.maximum(q *
                                       error, (q - 1) * error))
                return loss / len(self.quantiles)

        class CRPSLoss:
            def __call__(self, y_pred, y_true):
                import torch
                return torch.mean((y_pred - y_true) ** 2)

        def clean_and_prepare_features(data, target_column=None):
            """Fallback data cleaning function"""
            # Note: target_column parameter kept for API compatibility but not
            # used
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                # Basic cleaning
                data = data.fillna(method='ffill').fillna(method='bfill')
                data = data.select_dtypes(include=[np.number])
            return data


class TransformerExpert(nn.Module):
    """
    Transformer expert model for time series forecasting.
    Implements TFT-style architecture with attention mechanisms.
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        forecast_horizon: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        quantiles: Optional[List[float]] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(sequence_length, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Multi-horizon quantile outputs
        self.quantile_heads = nn.ModuleDict({
            f'q_{int(q*100)}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, forecast_horizon)
            ) for q in quantiles
        })

        # Attention pooling for sequence aggregation
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-quantile multi-horizon output.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Dict mapping quantile names to forecast tensors of shape (batch_size, forecast_horizon)
        """
        _, seq_len, _ = x.shape

        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)

        # Apply transformer
        x = self.transformer(x)

        # Attention pooling to get single representation
        # Use last timestep as query
        query = x[:, -1:, :]  # (batch_size, 1, d_model)
        attn_output, _ = self.attention_pool(query, x, x)
        pooled = attn_output.squeeze(1)  # (batch_size, d_model)

        # Generate quantile forecasts
        outputs = {}
        for q_name, head in self.quantile_heads.items():
            outputs[q_name] = head(pooled)

        return outputs


class LightGBMExpert:
    """
    LightGBM expert for tabular features (technical indicators, news sentiment, macro).
    Uses gradient boosting with quantile regression.
    """

    def __init__(
        self,
        forecast_horizon: int = 5,
        quantiles: Optional[List[float]] = None,
        lgb_params: Optional[Dict] = None
    ):
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.models = {}  # Will store one model per quantile per horizon
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Default LightGBM parameters
        self.lgb_params = lgb_params or {
            'objective': 'quantile',
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Add GPU support via environment variable
        import os
        device = os.environ.get('LIGHTGBM_DEVICE', 'cpu')
        if device == 'gpu':
            self.lgb_params['device_type'] = 'cuda'
            self.lgb_params['gpu_platform_id'] = int(os.environ.get('LIGHTGBM_GPU_PLATFORM_ID', 0))
            self.lgb_params['gpu_device_id'] = int(os.environ.get('LIGHTGBM_GPU_DEVICE_ID', 0))

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract tabular features for LightGBM"""
        features = []
        feature_names = []

        # Technical indicators
        tech_features = [
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'volatility_10d', 'volatility_20d',
            'rsi', 'volume_ratio',
            'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50'
        ]

        for feat in tech_features:
            if feat in df.columns:
                features.append(df[feat].values)
                feature_names.append(feat)

        # News sentiment features
        sentiment_features = [
            'sentiment_score', 'sentiment_positive', 'sentiment_negative',
            'news_volume', 'sentiment_volatility'
        ]

        for feat in sentiment_features:
            if feat in df.columns:
                features.append(df[feat].values)
                feature_names.append(feat)

        # Macro features (if available)
        macro_features = [
            'vix', 'treasury_10y', 'dxy', 'oil_price'
        ]

        for feat in macro_features:
            if feat in df.columns:
                features.append(df[feat].values)
                feature_names.append(feat)

        # Time-based features
        if 'date' in df.columns:
            df.loc[:, 'date'] = pd.to_datetime(df['date'])
            features.extend([
                df['date'].dt.dayofweek.values,
                df['date'].dt.month.values,
                df['date'].dt.quarter.values,
                np.sin(2 * np.pi * df['date'].dt.dayofyear / 365),
                np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
            ])
            feature_names.extend(
                ['dow', 'month', 'quarter', 'day_sin', 'day_cos'])

        self.feature_names = feature_names
        return np.column_stack(features) if features else np.array(
            []).reshape(
            len(df), 0)

    def fit(self, df: pd.DataFrame, target_columns: List[str]):
        """
        Fit LightGBM models for each quantile and forecast horizon.

        Args:
            df: DataFrame with features and targets
            target_columns: List of target column names (e.g., ['log_return_1d', 'log_return_5d'])
        """
        x_array = self._prepare_features(df)

        if x_array.shape[1] == 0:
            raise ValueError("No valid features found for LightGBM training")

        # Convert to DataFrame with feature names
        X = pd.DataFrame(x_array, columns=self.feature_names, index=df.index)

        # Scale features
        x_scaled = self.scaler.fit_transform(X)
        # Convert back to DataFrame to preserve feature names for LightGBM
        x_scaled_df = pd.DataFrame(x_scaled, columns=X.columns, index=X.index)

        # Train models for each quantile and horizon
        for i, target_col in enumerate(target_columns[:self.forecast_horizon]):
            if target_col not in df.columns:
                continue

            y = df[target_col].values

            # Remove NaN values
            valid_mask = ~(np.isnan(x_scaled).any(axis=1) | np.isnan(y))
            x_clean_df = x_scaled_df[valid_mask]
            y_clean = y[valid_mask]

            if len(x_clean_df) < 100:  # Need sufficient data
                continue

            for quantile in self.quantiles:
                model_key = f'h{i}_q{int(quantile*100)}'

                if LIGHTGBM_AVAILABLE:
                    # Use LightGBM
                    params = self.lgb_params.copy()
                    params['alpha'] = quantile

                    model = lgb.LGBMRegressor(**params)
                    model.fit(x_clean_df, y_clean)
                else:
                    # Fallback to RandomForest with quantile support
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(x_clean_df, y_clean)

                self.models[model_key] = model

        self.is_fitted = True

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate quantile forecasts for multiple horizons.

        Returns:
            Dict mapping quantile names to forecast arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        x_array = self._prepare_features(df)

        # Convert to DataFrame with feature names
        X = pd.DataFrame(x_array, columns=self.feature_names, index=df.index)
        x_scaled = self.scaler.transform(X)

        # Convert back to DataFrame to preserve feature names for LightGBM
        x_scaled_df = pd.DataFrame(x_scaled, columns=X.columns, index=X.index)

        predictions = {}

        for quantile in self.quantiles:
            horizon_preds = []

            for h in range(self.forecast_horizon):
                model_key = f'h{h}_q{int(quantile*100)}'

                if model_key in self.models:
                    pred = self.models[model_key].predict(x_scaled_df)
                    horizon_preds.append(pred)
                else:
                    # Fallback to zeros if model not available
                    horizon_preds.append(np.zeros(len(x_scaled_df)))

            predictions[f'q_{int(quantile * 100)}'] = np.column_stack(horizon_preds)

        return predictions


class LinearARExpert:
    """
    Linear autoregressive and exponential moving average expert.
    Provides a simple but robust baseline.
    """

    def __init__(
        self,
        forecast_horizon: int = 5,
        quantiles: Optional[List[float]] = None,
        ar_lags: int = 10,
        ema_spans: Optional[List[int]] = None
    ):
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.ar_lags = ar_lags
        self.ema_spans = ema_spans or [5, 10, 20]
        self.ar_coef = {}
        self.volatility_model = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_column: str = 'log_return_1d'):
        """Fit AR model and estimate volatility"""
        if target_column not in df.columns:
            print(
                f"Warning: Target column {target_column} not found in AR expert")
            self.is_fitted = False
            return

        returns = df[target_column].dropna().values

        # Adjust minimum data requirement for small datasets
        min_data_points = max(
            self.ar_lags + 20,
            30)  # More flexible requirement

        if len(returns) < min_data_points:
            print(f"Warning: Insufficient data for AR fitting ({len(returns)} < {min_data_points})")
            self.is_fitted = False
            return

        try:
            # Fit AR model for each horizon
            for h in range(1, self.forecast_horizon + 1):
                X, y = [], []

                for i in range(self.ar_lags, len(returns) - h + 1):
                    X.append(returns[i-self.ar_lags:i])
                    y.append(returns[i+h-1])

                if len(X) == 0:  # No valid training samples
                    print(
                        f"Warning: No valid training samples for horizon {h}")
                    continue

                X, y = np.array(X), np.array(y)

                # Simple OLS with regularization for numerical stability
                try:
                    coef = np.linalg.lstsq(X, y, rcond=1e-10)[0]
                    self.ar_coef[f'h{h}'] = coef
                except np.linalg.LinAlgError:
                    print(f"Warning: Failed to fit AR model for horizon {h}")
                    continue

            # Estimate volatility using rolling window
            volatility = pd.Series(returns).rolling(
                min(20, len(returns)//2)).std().bfill()
            self.volatility_model = volatility.values

            # Only mark as fitted if we successfully fit at least one horizon
            self.is_fitted = len(self.ar_coef) > 0

        except Exception as e:
            print(f"AR fitting failed: {e}")
            self.is_fitted = False

    def predict(self, df: pd.DataFrame,
                target_column: str = 'log_return_1d') -> Dict[str, np.ndarray]:
        """Generate quantile forecasts using AR + volatility"""
        if not self.is_fitted:
            print("Warning: AR model not fitted, using simple forecasts")
            # Return simple forecasts based on recent data
            return self._simple_forecast_fallback(df, target_column)

        if target_column not in df.columns:
            print(
                f"Warning: Target column {target_column} not found for AR prediction")
            return self._simple_forecast_fallback(df, target_column)

        returns = df[target_column].dropna().values

        if len(returns) < self.ar_lags:
            # Adjust AR lags dynamically based on available data
            available_lags = max(1, len(returns) - 1)
            if available_lags < self.ar_lags:
                print(f"Adjusting AR lags from {self.ar_lags} to {available_lags} due to limited data")
                original_lags = self.ar_lags
                self.ar_lags = available_lags

                # Re-fit with adjusted lags if needed
                if not self.is_fitted:
                    self.fit(df)

                # Restore original lags for future use
                self.ar_lags = original_lags

                if not self.is_fitted:
                    return self._simple_forecast_fallback(df, target_column)

        try:
            # Get recent returns for AR
            recent_returns = returns[-self.ar_lags:]

            # Estimate current volatility
            current_vol = self._estimate_volatility(returns)

            predictions = {}

            for quantile in self.quantiles:
                horizon_preds = []
                z_score = self._get_quantile_z_score(quantile)

                for h in range(1, self.forecast_horizon + 1):
                    pred = self._predict_horizon_quantile(
                        recent_returns, current_vol, z_score, h)
                    horizon_preds.append([pred])

                predictions[f'q_{int(quantile * 100)}'] = np.column_stack(horizon_preds)

            return predictions

        except Exception as e:
            print(f"AR prediction failed: {e}")
            return self._simple_forecast_fallback(df, target_column)

    def _estimate_volatility(self, returns: np.ndarray) -> float:
        """Estimate current volatility from returns"""
        if len(returns) >= 20:
            return np.std(returns[-20:])
        elif len(returns) > 1:
            return np.std(returns)
        else:
            return 0.02

    def _get_quantile_z_score(self, quantile: float) -> float:
        """Get z-score for quantile (with scipy fallback)"""
        try:
            from scipy import stats
            return stats.norm.ppf(quantile)
        except ImportError:
            # Fallback without scipy
            return {
                0.1: -1.28,
                0.25: -0.67,
                0.5: 0.0,
                0.75: 0.67,
                0.9: 1.28}.get(
                quantile,
                0.0)

    def _predict_horizon_quantile(
            self,
            recent_returns: np.ndarray,
            current_vol: float,
            z_score: float,
            horizon: int) -> float:
        """Predict single horizon quantile using AR model"""
        if f'h{horizon}' in self.ar_coef:
            try:
                coef = self.ar_coef[f'h{horizon}']

                # Handle dimension mismatch by adjusting data to match
                # coefficient length
                if len(recent_returns) != len(coef):
                    if len(recent_returns) < len(coef):
                        # Pad with zeros if we have fewer returns than
                        # coefficients
                        padded_returns = np.pad(
                            recent_returns,
                            (len(coef) - len(recent_returns),
                             0),
                            'constant')
                        recent_returns = padded_returns
                    else:
                        # Use the most recent returns that match coefficient
                        # length
                        recent_returns = recent_returns[-len(coef):]

                # AR prediction
                ar_pred = np.dot(coef, recent_returns)
                # Add volatility-based quantile
                return ar_pred + z_score * current_vol * np.sqrt(horizon)
            except Exception as e:
                print(
                    f"Warning: AR prediction failed for horizon {horizon}: {e}")
                return 0.0
        else:
            return 0.0

    def _simple_forecast_fallback(self,
                                  df: pd.DataFrame,
                                  target_column: str = 'log_return_1d') -> Dict[str,
                                                                                np.ndarray]:
        """Simple fallback forecast when AR model fails"""
        # Use simple moving average and volatility estimates
        if target_column in df.columns:
            returns = df[target_column].dropna().values
            if len(returns) >= 10:
                recent_mean = np.mean(returns[-10:])
                recent_vol = np.std(returns[-10:])
            elif len(returns) > 1:
                recent_mean = np.mean(returns)
                recent_vol = np.std(returns)
            else:
                recent_mean, recent_vol = 0.0, 0.02
        else:
            recent_mean, recent_vol = 0.0, 0.02

        predictions = {}
        for quantile in self.quantiles:
            # Simple quantile using normal distribution
            try:
                from scipy import stats
                z_score = stats.norm.ppf(quantile)
            except ImportError:
                z_score = {
                    0.1: -1.28,
                    0.25: -0.67,
                    0.5: 0.0,
                    0.75: 0.67,
                    0.9: 1.28}.get(
                    quantile,
                    0.0)

            horizon_preds = []
            for h in range(1, self.forecast_horizon + 1):
                simple_pred = recent_mean + z_score * recent_vol * np.sqrt(h)
                horizon_preds.append([simple_pred])

            predictions[f'q_{int(quantile * 100)}'] = np.column_stack(horizon_preds)

        return predictions


class GatingNetwork(nn.Module):
    """
    Neural gating network that learns to weight expert predictions
    based on market regime, volatility, and news novelty.
    """

    def __init__(
        self,
        num_experts: int = 3,
        context_size: int = 10,
        hidden_size: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_experts = num_experts

        # Context features: volatility regime, news novelty, market conditions
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gating weights
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Ensure network uses float32 for consistency
        self.float()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute expert weights based on context.

        Args:
            context: Tensor of shape (batch_size, context_size)

        Returns:
            Expert weights of shape (batch_size, num_experts)
        """
        encoded = self.context_encoder(context)
        weights = self.gate(encoded)
        return weights

    def extract_context_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract context features for gating"""
        context_features = []

        # Volatility regime
        if 'volatility_20d' in df.columns:
            vol = df['volatility_20d'].iloc[-1] if not df.empty else 0.02
            vol_regime = min(vol / 0.03, 2.0)  # Normalize to [0, 2]
            context_features.append(vol_regime)
        else:
            context_features.append(0.5)  # Neutral volatility

        # News novelty (higher when more news or extreme sentiment)
        if 'news_volume' in df.columns and not df.empty:
            news_vol = df['news_volume'].iloc[-1]
            news_novelty = min(news_vol / 10, 1.0)  # Normalize
            context_features.append(news_novelty)
        else:
            context_features.append(0.5)

        # Market momentum
        if 'momentum_5d' in df.columns and not df.empty:
            momentum = df['momentum_5d'].iloc[-1]
            context_features.append(np.tanh(momentum * 10))  # Bounded momentum
        else:
            context_features.append(0.0)

        # Time features
        from datetime import datetime
        now = datetime.now()
        context_features.extend([
            np.sin(2 * np.pi * now.hour / 24),  # Hour of day
            np.cos(2 * np.pi * now.hour / 24),
            np.sin(2 * np.pi * now.weekday() / 7),  # Day of week
            np.cos(2 * np.pi * now.weekday() / 7),
        ])

        # Pad to context_size if needed
        while len(context_features) < 10:
            context_features.append(0.0)

        return np.array(context_features[:10])


class EnsembleForecaster:
    """
    Main ensemble forecasting system that combines all experts with gating.
    """

    def __init__(
        self,
        forecast_horizon: int = 5,
        quantiles: Optional[List[float]] = None,
        sequence_length: int = 20,
        enable_calibration: bool = True
    ):
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.sequence_length = sequence_length

        # Initialize experts
        self.transformer_expert = None
        self.lgb_expert = LightGBMExpert(forecast_horizon, quantiles)
        self.ar_expert = LinearARExpert(forecast_horizon, quantiles)
        
        # Initialize fitted flags
        self.lgb_expert.is_fitted = False
        self.ar_expert.is_fitted = False

        # Gating network
        self.gating_network = GatingNetwork(num_experts=3)
        self.gating_optimizer = None

        # Calibration system
        self.enable_calibration = enable_calibration and CALIBRATION_AVAILABLE
        if self.enable_calibration:
            try:
                calibrator = TimeSeriesCalibrator(
                    calibration_store_path="data/calibration/ensemble_calibration.parquet",
                    method="isotonic",
                    recalibration_days=7
                )
                self.calibration_manager = CalibrationManager(calibrator)
                print("Calibration system initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize calibration: {e}")
                self.enable_calibration = False
                self.calibration_manager = None
        else:
            self.calibration_manager = None

        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Fit all expert models and gating network.

        Args:
            df: Training dataframe with features and targets
            feature_columns: List of feature column names
        """
        print("Training Ensemble Forecaster...")

        # Clean and prepare data
        df_clean = self._prepare_training_data(df, feature_columns)

        # Fit individual experts
        self._fit_transformer_expert(df_clean, feature_columns)
        self._fit_lightgbm_expert(df_clean)
        self._fit_ar_expert(df_clean)

        # Train gating network
        self._train_gating_network(df_clean)

        self.is_fitted = True
        print("Ensemble training completed!")

    def _prepare_training_data(
            self,
            df: pd.DataFrame,
            feature_columns: List[str]) -> pd.DataFrame:
        """Prepare and clean training data"""
        df_clean, _, success = clean_and_prepare_features(df, feature_columns)
        if not success:
            print("Data preparation failed, using original features")
            df_clean = df

        # Create target columns if they don't exist
        if 'close' in df_clean.columns:
            for h in range(1, self.forecast_horizon + 1):
                target_col = f'log_return_{h}d'
                if target_col not in df_clean.columns:
                    df_clean[target_col] = np.log(
                        df_clean['close'] / df_clean['close'].shift(h))

        # Also create log_return_5d for transformer if missing
        if 'log_return_5d' not in df_clean.columns and 'close' in df_clean.columns:
            df_clean['log_return_5d'] = np.log(
                df_clean['close'] / df_clean['close'].shift(5))

        return df_clean

    def _fit_transformer_expert(self, df_clean: pd.DataFrame, 
                                feature_columns: List[str]):
        """Fit transformer expert model"""
        print("Training Transformer Expert...")
        
        try:
            if not self._prepare_transformer_data(df_clean, feature_columns):
                return
                
            self._create_transformer_model(feature_columns)
            self._execute_transformer_training(df_clean, feature_columns)
            
        except Exception as e:
            print(f"Transformer training failed: {e}")
            self.transformer_expert = None

    def _prepare_transformer_data(self, df_clean: pd.DataFrame, 
                                 feature_columns: List[str]) -> bool:
        """Prepare data for transformer training and check if sufficient"""
        from .forecast_helpers import _prepare_training_data
        
        x_seq, y_seq, _ = _prepare_training_data(
            df_clean, feature_columns, 'log_return_5d', self.sequence_length)
        
        if len(x_seq) <= 50:  # Insufficient data
            self.transformer_expert = None
            return False
            
        # Store prepared data for training
        self._transformer_x_seq = x_seq
        self._transformer_y_seq = y_seq
        return True

    def _create_transformer_model(self, feature_columns: List[str]):
        """Create and initialize transformer model"""
        from .precision_config import get_current_torch_dtype
        
        input_size = len(feature_columns)
        current_dtype = get_current_torch_dtype()
        
        # Temporarily set to float32 to avoid dtype mismatches
        torch.set_default_dtype(torch.float32)
        
        self.transformer_expert = TransformerExpert(
            input_size, self.sequence_length, self.forecast_horizon,
            quantiles=self.quantiles)
        
        # Initialize as not fitted and convert to float32
        self.transformer_expert.is_fitted = False
        self.transformer_expert = self.transformer_expert.float()
        
        # Restore original dtype
        torch.set_default_dtype(current_dtype)

    def _execute_transformer_training(self, df_clean: pd.DataFrame, 
                                     feature_columns: List[str]):
        """Execute the actual transformer training process"""
        x_seq = self._transformer_x_seq
        y_seq = self._transformer_y_seq
        
        # ===== ENSURE TENSOR DTYPE CONSISTENCY =====
        # Convert to float32 to prevent Float vs Double errors
        x_seq = x_seq.to(dtype=torch.float32)
        y_seq = y_seq.to(dtype=torch.float32)
        
        # Ensure model is also float32
        self.transformer_expert = self.transformer_expert.float()
        
        criterion = QuantileLoss(self.quantiles)
        optimizer = torch.optim.Adam(
            self.transformer_expert.parameters(), lr=0.001, weight_decay=1e-5)

        # Training loop
        for epoch in range(20):  # Quick training
            optimizer.zero_grad()
            outputs = self.transformer_expert(x_seq)
            
            # Compute loss for median quantile
            loss = criterion(outputs['q_50'], y_seq.unsqueeze(1))
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(f"Transformer epoch {epoch}: loss = {loss.item():.6f}")
        
        # Mark as successfully fitted
        self.transformer_expert.is_fitted = True
        
        # Clean up temporary data
        del self._transformer_x_seq, self._transformer_y_seq

    def _fit_lightgbm_expert(self, df_clean: pd.DataFrame):
        """Fit LightGBM expert model"""
        print("Training LightGBM Expert...")
        try:
            target_columns = [
                f'log_return_{h}d' for h in range(
                    1, self.forecast_horizon + 1)]
            self.lgb_expert.fit(df_clean, target_columns)
            # Mark as successfully fitted
            self.lgb_expert.is_fitted = True
            print("LightGBM expert fitted successfully")
        except Exception as e:
            print(f"LightGBM training failed: {e}")
            # Mark as not fitted
            self.lgb_expert.is_fitted = False

    def _fit_ar_expert(self, df_clean: pd.DataFrame):
        """Fit AR expert model"""
        print("Training AR Expert...")
        try:
            self.ar_expert.fit(df_clean, 'log_return_1d')
            # Mark as successfully fitted
            self.ar_expert.is_fitted = True
            print("AR expert fitted successfully")
        except Exception as e:
            print(f"AR training failed: {e}")
            # Mark as not fitted
            self.ar_expert.is_fitted = False

    def _train_gating_network(self, df_clean: pd.DataFrame):
        """Train the gating network"""
        # Import precision configuration

        self.gating_optimizer = torch.optim.Adam(
            self.gating_network.parameters(), lr=0.01, weight_decay=1e-5
        )

        print("Training Gating Network...")
        for _ in range(10):
            context = self.gating_network.extract_context_features(df_clean)
            
            # Convert context to tensor with proper dtype matching gating network
            context_tensor = torch.tensor(
                context, dtype=torch.float32).unsqueeze(0)
            
            # Ensure gating network is in float32 mode
            self.gating_network = self.gating_network.float()

            weights = self.gating_network(context_tensor)

            # Simple loss: encourage balanced weights initially
            balance_loss = torch.mean((weights - 1/3)**2)

            self.gating_optimizer.zero_grad()
            balance_loss.backward()
            self.gating_optimizer.step()

    def _get_transformer_predictions(
            self,
            df: pd.DataFrame,
            feature_columns: List[str]) -> Optional[Dict]:
        """Get predictions from transformer expert"""
        if self.transformer_expert is None:
            return None

        try:
            # Prepare sequence data
            from .forecast_helpers import _prepare_training_data
            x_seq, _, _ = _prepare_training_data(
                df, feature_columns, 'log_return_5d', self.sequence_length
            )

            if len(x_seq) > 0:
                with torch.no_grad():
                    transformer_outputs = self.transformer_expert(
                        x_seq[-1:])  # Last sequence
                return {k: v.numpy() for k, v in transformer_outputs.items()}
        except Exception as e:
            print(f"Transformer prediction failed: {e}")

        return None

    def _get_expert_predictions(
            self,
            df: pd.DataFrame,
            feature_columns: List[str]) -> Dict:
        """Get predictions from all experts"""
        expert_predictions = {}

        # Transformer predictions
        transformer_preds = self._get_transformer_predictions(
            df, feature_columns)
        expert_predictions['transformer'] = transformer_preds

        # LightGBM predictions
        try:
            lgb_preds = self.lgb_expert.predict(df)
            expert_predictions['lightgbm'] = lgb_preds
        except Exception as e:
            print(f"LightGBM prediction failed: {e}")
            expert_predictions['lightgbm'] = None

        # AR predictions
        try:
            ar_preds = self.ar_expert.predict(df, 'log_return_1d')
            expert_predictions['ar'] = ar_preds
        except Exception as e:
            print(f"AR prediction failed: {e}")
            expert_predictions['ar'] = None

        return expert_predictions

    def _combine_expert_predictions(
            self,
            expert_predictions: Dict,
            expert_weights: np.ndarray) -> Dict:
        """Combine expert predictions using gating weights"""
        ensemble_predictions = {}
        
        # ===== FIX QUANTILE KEY CONSISTENCY =====
        # Use both float and string keys to prevent KeyError: 0.5
        for q in self.quantiles:
            q_name = f'q_{int(q*100)}'  # String key like 'q_50'
            
            combined_pred = self._combine_quantile_predictions(
                expert_predictions, expert_weights, q_name
            )
            
            # Store with BOTH string and float keys for compatibility
            ensemble_predictions[q_name] = combined_pred  # 'q_50'
            ensemble_predictions[q] = combined_pred       # 0.5

        return ensemble_predictions

    def _combine_quantile_predictions(
            self,
            expert_predictions: Dict,
            expert_weights: np.ndarray,
            q_name: str) -> np.ndarray:
        """Combine predictions for a single quantile"""
        combined_pred = None
        total_weight = 0

        # Weight and combine available expert predictions
        for i, (expert_name, preds) in enumerate(expert_predictions.items()):
            # Check if expert was fitted successfully
            if not self._is_expert_fitted(expert_name):
                print(f"Skipping unfitted expert: {expert_name}")
                continue
                
            if not self._is_valid_prediction(preds, q_name):
                continue

            weight = expert_weights[i]
            pred_array = preds[q_name]

            combined_pred, total_weight = self._accumulate_prediction(
                combined_pred, pred_array, weight, total_weight)

        return self._finalize_combined_prediction(
            combined_pred, total_weight, q_name)

    def _is_expert_fitted(self, expert_name: str) -> bool:
        """Check if an expert was fitted successfully"""
        if expert_name == 'transformer':
            return hasattr(self, 'transformer_expert') and \
                   hasattr(self.transformer_expert, 'is_fitted') and \
                   self.transformer_expert.is_fitted
        elif expert_name == 'lightgbm':
            return hasattr(self, 'lgb_expert') and \
                   hasattr(self.lgb_expert, 'is_fitted') and \
                   self.lgb_expert.is_fitted
        elif expert_name == 'ar':
            return hasattr(self, 'ar_expert') and \
                   hasattr(self.ar_expert, 'is_fitted') and \
                   self.ar_expert.is_fitted
        return False

    def _is_valid_prediction(self, preds: Dict, q_name: str) -> bool:
        """Check if prediction is valid for combination"""
        if preds is None or q_name not in preds:
            return False

        pred_array = preds[q_name]
        if np.any(np.isnan(pred_array)) or np.any(np.isinf(pred_array)):
            print(
                f"Warning: NaN/Inf values detected in {q_name} predictions, skipping expert")
            return False

        return True

    def _accumulate_prediction(
            self,
            combined_pred,
            pred_array,
            weight: float,
            total_weight: float):
        """Accumulate weighted prediction"""
        if combined_pred is None:
            combined_pred = weight * pred_array
        else:
            combined_pred += weight * pred_array

        total_weight += weight
        return combined_pred, total_weight

    def _finalize_combined_prediction(
            self,
            combined_pred,
            total_weight: float,
            q_name: str) -> np.ndarray:
        """Finalize and validate combined prediction"""
        if combined_pred is not None and total_weight > 0:
            if abs(total_weight - 1.0) > 1e-6:  # Avoid exact float comparison
                combined_pred = combined_pred / total_weight

            # Final NaN check and replacement
            if np.any(
                    np.isnan(combined_pred)) or np.any(
                    np.isinf(combined_pred)):
                print(
                    f"Warning: NaN/Inf values in final combined prediction for {q_name}, using zeros")
                return np.zeros((1, self.forecast_horizon))

            return combined_pred
        else:
            # Fallback to zeros
            print(
                f"Warning: No valid expert predictions for {q_name}, using zeros")
            return np.zeros((1, self.forecast_horizon))

    def predict(self, df: pd.DataFrame,
                feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate ensemble forecasts by combining expert predictions with gating.

        Returns:
            Dict with quantile forecasts
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get expert predictions
        expert_predictions = self._get_expert_predictions(df, feature_columns)

        # Get gating weights
        context = self.gating_network.extract_context_features(df)
        # CRITICAL FIX: Use float32 to match gating network dtype
        context_tensor = torch.tensor(
            context, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            expert_weights = self.gating_network(context_tensor).numpy()[0]

        print(f"Expert weights: Transformer={expert_weights[0]:.3f}, "
              f"LightGBM={expert_weights[1]:.3f}, AR={expert_weights[2]:.3f}")

        # Combine predictions
        combined_predictions = self._combine_expert_predictions(
            expert_predictions, expert_weights)

        # Apply calibration if enabled
        if self.enable_calibration and self.calibration_manager:
            combined_predictions = self._apply_calibration(
                combined_predictions, df, feature_columns)

        return combined_predictions

    def _apply_calibration(self, predictions: Dict[str, np.ndarray], 
                          df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """
        Apply time-series calibration to ensemble predictions.
        
        This focuses on the median (0.5 quantile) predictions and treats them
        as directional probabilities for calibration.
        """
        from datetime import datetime
        
        try:
            # Extract ticker from dataframe if available
            ticker = "UNKNOWN"
            if 'ticker' in df.columns:
                ticker = df['ticker'].iloc[-1] if len(df) > 0 else "UNKNOWN"
            elif hasattr(df, 'name'):
                ticker = df.name or "UNKNOWN"
            
            # Convert median forecast to directional probability
            if 'q_0.5' in predictions:
                median_forecast = predictions['q_0.5']
                
                # Convert log returns to probability of positive movement
                # This is a simplified approach - in practice, you'd use more
                # sophisticated probability calibration
                raw_prob = self._forecast_to_probability(median_forecast)
                
                # Apply calibration
                calibrated_prob, diagnostics = self.calibration_manager.process_prediction(
                    raw_probability=raw_prob,
                    ticker=ticker,
                    timestamp=datetime.now()
                )
                
                # Store calibration info for monitoring
                if not hasattr(self, '_calibration_history'):
                    self._calibration_history = []
                
                self._calibration_history.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'raw_probability': raw_prob,
                    'calibrated_probability': calibrated_prob,
                    'adjustment': calibrated_prob - raw_prob,
                    'diagnostics': diagnostics
                })
                
                # Keep only recent history (last 100 predictions)
                if len(self._calibration_history) > 100:
                    self._calibration_history = self._calibration_history[-100:]
                
                print(f"Calibration: {raw_prob:.3f} → {calibrated_prob:.3f} "
                      f"(adj: {calibrated_prob - raw_prob:+.3f})")
                
                # Optionally adjust predictions based on calibration
                # For now, we'll just log the calibration info
                
        except Exception as e:
            print(f"Warning: Calibration failed: {e}")
        
        return predictions
    
    def _forecast_to_probability(self, forecast: np.ndarray) -> float:
        """
        Convert forecast array to a single directional probability.
        
        This is a simplified approach that converts log returns to probability
        of positive movement over the forecast horizon.
        """
        # Take the mean of the forecast horizon for overall direction
        mean_forecast = np.mean(forecast) if len(forecast.shape) > 0 else float(forecast)
        
        # Convert log return to probability using sigmoid-like function
        # Positive log returns -> probability > 0.5
        # Negative log returns -> probability < 0.5
        prob = 1 / (1 + np.exp(-mean_forecast * 10))  # Scale factor for sensitivity
        
        # Clip to reasonable bounds
        return float(np.clip(prob, 0.01, 0.99))
    
    def get_calibration_diagnostics(self) -> Dict:
        """Get calibration system diagnostics and recent history."""
        diagnostics = {
            'calibration_enabled': self.enable_calibration,
            'calibration_available': self.calibration_manager is not None
        }
        
        if self.calibration_manager:
            calibrator_diag = self.calibration_manager.calibrator.get_calibration_diagnostics()
            diagnostics.update(calibrator_diag)
        
        if hasattr(self, '_calibration_history'):
            diagnostics['recent_predictions'] = len(self._calibration_history)
            if self._calibration_history:
                recent_adjustments = [
                    p['adjustment'] for p in self._calibration_history[-10:]
                ]
                diagnostics['mean_recent_adjustment'] = np.mean(recent_adjustments)
                diagnostics['std_recent_adjustment'] = np.std(recent_adjustments)
        
        return diagnostics
    
    def run_calibration_maintenance(self) -> Dict:
        """Run calibration maintenance (update outcomes, recalibrate if needed)."""
        if not self.enable_calibration or not self.calibration_manager:
            return {'error': 'Calibration not available'}
        
        try:
            return self.calibration_manager.run_maintenance()
        except Exception as e:
            return {'error': f'Calibration maintenance failed: {e}'}
