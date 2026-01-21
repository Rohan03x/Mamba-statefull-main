"""
Temporal Fusion Transformer (TFT) Implementation

This module implements the TFT architecture following the original paper:
"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"

Key features:
- Variable selection networks for feature importance
- LSTM encoders for sequence processing  
- Multi-head attention for temporal patterns
- Quantile regression for uncertainty estimation
- Interpretable attention weights and variable importance

Implementation follows TFT best practices with proper:
- Static, known-future, and observed-past feature handling
- Gating mechanisms and skip connections
- Layer normalization and dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from types import SimpleNamespace
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import os
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    try:
        if os.getenv("DISABLE_TF32", "0").lower() not in {"1", "true"}:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        if os.getenv("DISABLE_CUDNN_BENCHMARK", "0").lower() not in {"1", "true"}:
            torch.backends.cudnn.benchmark = True
    except Exception as _perf_err:
        logger.debug(f"Torch performance tuning skipped: {_perf_err}")

@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer"""
    
    # Model architecture
    hidden_size: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    
    # Feature dimensions
    num_static_features: int = 0
    num_known_future_features: int = 0  
    num_observed_past_features: int = 0
    num_targets: int = 1
    
    # Sequence lengths
    input_length: int = 60
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 20, 60])
    
    # Training parameters
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-5
    use_gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    early_stopping_patience: int = 10
    
    # Variable selection
    use_variable_selection: bool = True
    variable_selection_threshold: float = 0.01
    
    # Interpretability
    compute_attention_weights: bool = True
    compute_variable_importance: bool = True

class VariableSelectionNetwork(nn.Module):
    """Variable selection network for feature importance"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Variable selection layers
        self.flattened_grn = GatedResidualNetwork(
            input_size, hidden_size, output_size=input_size, dropout=dropout
        )
        
        # Variable weights
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size=1, hidden_size=hidden_size, dropout=dropout)
            for _ in range(input_size)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, flattened_inputs: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for variable selection
        
        Args:
            flattened_inputs: [batch, input_size] or [batch, time, input_size]
            context: Optional context vector
            
        Returns:
            Tuple of (selected_features, selection_weights)
        """
        
        # Flatten if needed
        original_shape = flattened_inputs.shape
        if len(original_shape) > 2:
            batch_size, time_steps = original_shape[0], original_shape[1]
            flattened_inputs = flattened_inputs.view(-1, original_shape[-1])
        else:
            batch_size, time_steps = original_shape[0], 1
        
        # Variable selection weights
        sparse_weights = self.flattened_grn(flattened_inputs, context)
        sparse_weights = self.softmax(sparse_weights)
        
        # Process each variable separately
        processed_inputs = []
        for i, grn in enumerate(self.single_variable_grns):
            processed_inputs.append(
                grn(flattened_inputs[..., i:i+1])
            )
        
        processed_inputs = torch.cat(processed_inputs, dim=-1)
        
        # Apply selection weights
        selected_features = processed_inputs * sparse_weights
        
        # Reshape back if needed
        if len(original_shape) > 2:
            selected_features = selected_features.view(batch_size, time_steps, -1)
            sparse_weights = sparse_weights.view(batch_size, time_steps, -1)
        
        return selected_features, sparse_weights

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with skip connections"""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 output_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        if output_size is None:
            output_size = input_size
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # Gating layer
        self.gating_layer = nn.Linear(hidden_size, output_size)
        
        # Skip connection
        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = None
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with gating and skip connection"""
        
        # Store input for skip connection
        residual = x
        
        # Main path
        x = F.elu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        
        # Apply context if provided
        if context is not None:
            x = x + context
        
        x = F.elu(x)
        x = self.dropout(x)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gating_layer(x))
        x = self.linear3(x) * gate
        
        # Skip connection
        if self.skip_layer is not None:
            residual = self.skip_layer(residual)
        
        # Add residual and normalize
        x = x + residual
        x = self.layer_norm(x)
        
        return x

class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability features"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)  
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Store attention weights for interpretability
        self.attention_weights = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with attention weight storage
        
        Args:
            query, key, value: [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # Store for interpretability
        
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get stored attention weights for interpretability"""
        return self.attention_weights

class TemporalFusionTransformer(nn.Module):
    """Complete TFT implementation"""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Input processing
        self._build_input_layers()
        
        # Variable selection networks
        if config.use_variable_selection:
            self._build_variable_selection()
        
        # Sequence encoders
        self._build_sequence_encoders()
        
        # Attention mechanism
        self._build_attention()
        
        # Output layers
        self._build_output_layers()
        
        # Initialize weights
        self._init_weights()
    
    def _build_input_layers(self):
        """Build input embedding and processing layers"""
        
        # Static feature processing
        if self.config.num_static_features > 0:
            self.static_embedding = nn.Linear(
                self.config.num_static_features, self.hidden_size
            )
            self.static_context_grn = GatedResidualNetwork(
                self.hidden_size, self.hidden_size, dropout=self.config.dropout
            )
        
        # Observed past feature processing
        if self.config.num_observed_past_features > 0:
            self.observed_embedding = nn.Linear(
                self.config.num_observed_past_features, self.hidden_size
            )
        
        # Known future feature processing
        if self.config.num_known_future_features > 0:
            self.known_future_embedding = nn.Linear(
                self.config.num_known_future_features, self.hidden_size
            )
    
    def _build_variable_selection(self):
        """Build variable selection networks"""
        
        # Historical variable selection
        hist_input_size = 0
        if hasattr(self, 'observed_embedding'):
            hist_input_size += self.hidden_size
        
        if hist_input_size > 0:
            self.historical_vs = VariableSelectionNetwork(
                hist_input_size, self.hidden_size, self.config.dropout
            )
        
        # Future variable selection
        future_input_size = 0
        if hasattr(self, 'known_future_embedding'):
            future_input_size += self.hidden_size
        
        if future_input_size > 0:
            self.future_vs = VariableSelectionNetwork(
                future_input_size, self.hidden_size, self.config.dropout
            )
    
    def _build_sequence_encoders(self):
        """Build LSTM encoders for sequence processing"""
        
        # Historical sequence encoder
        self.historical_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.config.num_encoder_layers,
            dropout=self.config.dropout if self.config.num_encoder_layers > 1 else 0,
            batch_first=True
        )
        
        # Future sequence encoder  
        self.future_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.config.num_decoder_layers,
            dropout=self.config.dropout if self.config.num_decoder_layers > 1 else 0,
            batch_first=True
        )
    
    def _build_attention(self):
        """Build attention mechanism"""
        
        self.multihead_attention = InterpretableMultiHeadAttention(
            d_model=self.hidden_size,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout
        )
        
        # Post-attention processing
        self.post_attention_grn = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, dropout=self.config.dropout
        )
    
    def _build_output_layers(self):
        """Build output layers for multi-horizon prediction"""
        
        # Output for each horizon
        self.output_layers = nn.ModuleDict()
        
        for horizon in self.config.prediction_horizons:
            horizon_layers = nn.ModuleDict()
            
            # Quantile outputs for each target
            for target_idx in range(self.config.num_targets):
                target_layers = nn.ModuleDict()
                
                for quantile in self.config.quantiles:
                    quantile_key = f"q{int(quantile * 100)}"
                    target_layers[quantile_key] = nn.Sequential(
                        GatedResidualNetwork(
                            self.hidden_size, self.hidden_size, 
                            output_size=self.hidden_size, dropout=self.config.dropout
                        ),
                        nn.Linear(self.hidden_size, 1)
                    )
                
                horizon_layers[f"target_{target_idx}"] = target_layers
            
            self.output_layers[f"horizon_{horizon}"] = horizon_layers
    
    def _init_weights(self):
        """Initialize model weights"""
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _encode_history(self, historical_features: torch.Tensor) -> torch.Tensor:
        hist_encoded, _ = self.historical_lstm(historical_features)
        return hist_encoded

    def _decode_future(self, future_features: torch.Tensor) -> torch.Tensor:
        future_encoded, _ = self.future_lstm(future_features)
        return future_encoded

    def _self_attention(self, combined_features: torch.Tensor) -> torch.Tensor:
        return self.multihead_attention(combined_features, combined_features, combined_features)

    def _post_attention(self, attended_features: torch.Tensor) -> torch.Tensor:
        return self.post_attention_grn(attended_features)

    def _compute_static_context(self, inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if 'static' not in inputs or not hasattr(self, 'static_embedding'):
            return None
        static_features = self.static_embedding(inputs['static'])
        return self.static_context_grn(static_features)

    def _encode_historical_state(
        self,
        inputs: Dict[str, torch.Tensor],
        static_context: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        if 'observed_past' not in inputs or not hasattr(self, 'observed_embedding'):
            return torch.zeros(batch_size, 1, self.hidden_size, device=device)

        observed_embedded = self.observed_embedding(inputs['observed_past'])
        if hasattr(self, 'historical_vs'):
            historical_features, _ = self.historical_vs(observed_embedded, static_context)
        else:
            historical_features = observed_embedded

        if historical_features is None:
            return torch.zeros(batch_size, 1, self.hidden_size, device=device)

        encoder_fn = self._encode_history
        hist_encoded = (
            checkpoint(encoder_fn, historical_features)
            if use_checkpoint
            else encoder_fn(historical_features)
        )
        return hist_encoded[:, -1:, :]

    def _prepare_future_representation(
        self,
        horizon: int,
        inputs: Dict[str, torch.Tensor],
        static_context: Optional[torch.Tensor],
        encoded_state: torch.Tensor,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        if 'known_future' not in inputs or not hasattr(self, 'known_future_embedding'):
            return encoded_state

        future_slice = inputs['known_future'][:, -horizon:, :]
        future_embedded = self.known_future_embedding(future_slice)
        if hasattr(self, 'future_vs'):
            future_features, _ = self.future_vs(future_embedded, static_context)
        else:
            future_features = future_embedded

        decoder_fn = self._decode_future
        future_encoded = (
            checkpoint(decoder_fn, future_features)
            if use_checkpoint
            else decoder_fn(future_features)
        )
        return torch.cat([encoded_state, future_encoded], dim=1)

    def _apply_attention_stack(
        self,
        combined_features: torch.Tensor,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        attention_fn = self._self_attention
        post_fn = self._post_attention
        attended = (
            checkpoint(attention_fn, combined_features)
            if use_checkpoint
            else attention_fn(combined_features)
        )
        return checkpoint(post_fn, attended) if use_checkpoint else post_fn(attended)

    def _predict_horizon_outputs(
        self,
        horizon_key: str,
        final_features: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        prediction_input = final_features[:, -1, :]
        horizon_results: Dict[str, Dict[str, torch.Tensor]] = {}

        for target_idx in range(self.config.num_targets):
            target_key = f"target_{target_idx}"
            target_layers = self.output_layers[horizon_key][target_key]
            horizon_results[target_key] = self._predict_target_quantiles(
                target_layers, prediction_input
            )

        return horizon_results

    def _predict_target_quantiles(
        self,
        target_layers: Dict[str, nn.Module],
        prediction_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_results: Dict[str, torch.Tensor] = {}
        for quantile in self.config.quantiles:
            quantile_key = f"q{int(quantile * 100)}"
            prediction = target_layers[quantile_key](prediction_input)
            target_results[quantile_key] = prediction.squeeze(-1)
        return target_results
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa: C901, CCR001
        """
        Forward pass through TFT
        
        Args:
            inputs: Dictionary with tensors:
                - 'static': [batch, static_features] 
                - 'observed_past': [batch, input_length, obs_features]
                - 'known_future': [batch, total_length, known_features]
                
        Returns:
            Dictionary with predictions for each horizon and quantile
        """
        
        batch_size = next(iter(inputs.values())).size(0)
        device = next(iter(inputs.values())).device
        use_checkpoint = bool(self.config.use_gradient_checkpointing)

        static_context = self._compute_static_context(inputs)
        encoded_state = self._encode_historical_state(
            inputs, static_context, batch_size, device, use_checkpoint
        )

        results: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for horizon in self.config.prediction_horizons:
            horizon_key = f"horizon_{horizon}"
            combined_features = self._prepare_future_representation(
                horizon, inputs, static_context, encoded_state, use_checkpoint
            )
            final_features = self._apply_attention_stack(combined_features, use_checkpoint)
            results[horizon_key] = self._predict_horizon_outputs(horizon_key, final_features)

        return results
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights for interpretability"""
        return self.multihead_attention.get_attention_weights()

class TFTPredictor:
    """Wrapper for TFT model with training and prediction utilities"""
    
    def __init__(self, config: TFTConfig):
        self.config = config
        self.model = TemporalFusionTransformer(config)
        compile_enabled = os.getenv("DISABLE_TORCH_COMPILE", "0").lower() not in {"1", "true"}
        if compile_enabled and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
                logger.debug("TFT model compiled with torch.compile")
            except Exception as _compile_err:
                logger.debug(f"torch.compile skipped for TFT: {_compile_err}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self._amp_enabled = (
            self.device.type == 'cuda'
            and os.getenv('DISABLE_TORCH_AMP', '0').lower() not in {'1', 'true'}
        )
        self._amp_dtype = torch.float16 if self._amp_enabled else torch.float32
        self.grad_scaler = GradScaler(enabled=self._amp_enabled)
        self.max_grad_norm = float(getattr(config, 'max_grad_norm', 1.0) or 1.0)
        env_checkpoint = os.getenv('TFT_USE_CHECKPOINT', '0').lower() in {'1', 'true', 'yes'}
        if env_checkpoint:
            config.use_gradient_checkpointing = True
        if bool(getattr(config, 'use_gradient_checkpointing', False)):
            self.model.config.use_gradient_checkpointing = True  # type: ignore[attr-defined]
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.training_history = []

    def _create_dataloader(
        self,
        windows: Dict[str, np.ndarray],
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create PyTorch DataLoader from windows"""

        batch = int(batch_size or self.config.batch_size or 32)
        num_workers = max(0, int(getattr(self.config, 'dataloader_num_workers', 0) or 0))
        pin_memory = bool(getattr(self.config, 'dataloader_pin_memory', True)) and self.device.type == 'cuda'
        persistent_workers = bool(getattr(self.config, 'dataloader_persistent_workers', True)) and num_workers > 0

        tensors = {
            key: torch.as_tensor(array, dtype=torch.float32)
            for key, array in windows.items()
        }
        
        # Create dataset
        tensor_list = []
        keys = []
        
        for key in ['static', 'observed_past', 'known_future', 'targets']:
            if key in tensors:
                tensor_list.append(tensors[key])
                keys.append(key)
        
        dataset = TensorDataset(*tensor_list)
        dataloader = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        
        # Store key mapping for reconstruction
        dataloader.key_mapping = keys
        
        return dataloader
    
    def _reconstruct_batch(self, batch: Tuple[torch.Tensor, ...], 
                          key_mapping: List[str]) -> Dict[str, torch.Tensor]:
        """Reconstruct input dictionary from batch"""
        
        inputs = {}
        targets = None
        
        for i, key in enumerate(key_mapping):
            tensor = batch[i].to(self.device)
            if key == 'targets':
                targets = tensor
            else:
                inputs[key] = tensor
        
        return inputs, targets

    def _iter_prediction_blocks(
        self, outputs: Dict[str, Any]
    ) -> Iterator[Tuple[int, int, Dict[str, Any]]]:
        """Yield valid target blocks for each horizon and target index."""

        for h_idx, horizon in enumerate(self.config.prediction_horizons):
            horizon_block = outputs.get(f"horizon_{horizon}")
            if not isinstance(horizon_block, dict):
                continue

            for t_idx in range(self.config.num_targets):
                target_block = horizon_block.get(f"target_{t_idx}")
                if isinstance(target_block, dict):
                    yield h_idx, t_idx, target_block

    def _iter_quantile_predictions(
        self, target_block: Dict[str, Any]
    ) -> Iterator[Tuple[float, torch.Tensor]]:
        """Yield available quantile predictions from a target block."""

        for quantile in self.config.quantiles:
            quantile_key = f"q{int(quantile * 100)}"
            predictions = target_block.get(quantile_key)
            if predictions is not None:
                yield quantile, predictions

    def _aggregate_quantile_loss(  # noqa: C901, CCR001  # pylint: disable=too-many-branches,too-many-statements
        self,
        outputs: Dict[str, Any],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        total = torch.zeros((), device=targets.device, dtype=torch.float32)
        loss_count = 0

        for h_idx, t_idx, target_block in self._iter_prediction_blocks(outputs):
            target_values = targets[:, h_idx, t_idx]
            valid_mask = ~torch.isnan(target_values)
            if not torch.any(valid_mask):
                continue

            for quantile, predictions in self._iter_quantile_predictions(target_block):
                loss = self.quantile_loss(
                    predictions[valid_mask],
                    target_values[valid_mask],
                    quantile,
                )
                total = total + loss
                loss_count += 1

        return total, loss_count
    
    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                     quantile: float) -> torch.Tensor:
        """Compute quantile loss"""
        
        errors = targets - predictions
        loss = torch.max(
            quantile * errors,
            (quantile - 1) * errors
        )
        return loss.mean()
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            inputs, targets = self._reconstruct_batch(batch, train_loader.key_mapping)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(inputs)
                batch_loss, loss_count = self._aggregate_quantile_loss(outputs, targets)

            if loss_count == 0:
                continue

            batch_loss = batch_loss / loss_count
            loss_value = float(batch_loss.detach().item())

            if self._amp_enabled:
                self.grad_scaler.scale(batch_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss_value
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._reconstruct_batch(batch, val_loader.key_mapping)
                with autocast(
                    device_type=self.device.type,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ):
                    outputs = self.model(inputs)
                    batch_loss, loss_count = self._aggregate_quantile_loss(outputs, targets)

                if loss_count == 0:
                    continue

                avg_loss = float((batch_loss / loss_count).detach().item())
                total_loss += avg_loss
                num_batches += 1
        
        return total_loss / max(num_batches, 1)

class TFTTrainer:
    """Complete TFT training pipeline"""
    
    def __init__(self, config: TFTConfig):
        self.config = config
        self.predictor = TFTPredictor(config)
        self.training_results = SimpleNamespace(
            training_history=[],
            curriculum_history=[],
            best_epoch=0,
            best_validation_loss=float("inf"),
            training_duration=0.0,
            model_size=0,
            total_parameters=0,
            final_horizons=list(config.prediction_horizons),
            interpretability_results={},
            evaluation_results={},
        )
        self.curriculum_scheduler = None
        
    def train(
        self,
        train_windows: Dict[str, np.ndarray],
        val_windows: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Train TFT model with scheduler, early stopping, and OOM fallback."""

        logger.info("Starting TFT training...")

        base_batch = max(1, int(self.config.batch_size or 32))
        batch_size = base_batch
        min_batch = max(8, base_batch // 4) if self.predictor.device.type == "cuda" else base_batch

        while True:
            try:
                return self._train_internal(train_windows, val_windows, batch_size)
            except RuntimeError as err:  # pragma: no cover - defensive guard
                message = str(err).lower()
                can_retry = (
                    "out of memory" in message
                    and self.predictor.device.type == "cuda"
                    and batch_size > min_batch
                )
                if not can_retry:
                    raise

                batch_size = max(batch_size // 2, min_batch)
                logger.warning("CUDA OOM detected; retrying with batch_size=%d", batch_size)
                torch.cuda.empty_cache()

    def _train_internal(
        self,
        train_windows: Dict[str, np.ndarray],
        val_windows: Dict[str, np.ndarray],
        batch_size: int,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        self._prepare_training_state(batch_size)

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        early_stop_patience = max(1, int(getattr(self.config, "early_stopping_patience", 10)))

        for epoch in range(self.config.max_epochs):
            self._apply_curriculum()

            train_loader, val_loader = self._build_epoch_loaders(
                train_windows, val_windows, batch_size
            )
            train_loss = self.predictor.train_epoch(train_loader)
            val_loss = self.predictor.validate_epoch(val_loader)

            self._step_scheduler(val_loss)
            self._record_curriculum(epoch, val_loss)

            best_val_loss, best_state, patience_counter = self._update_epoch_metrics(
                epoch,
                train_loss,
                val_loss,
                batch_size,
                best_val_loss,
                best_state,
                patience_counter,
            )

            if self._should_stop(patience_counter, early_stop_patience, epoch):
                break

        if best_state is not None:
            self.predictor.model.load_state_dict(best_state)

        self._finalize_training(best_val_loss, start_time)

        return {
            "best_val_loss": best_val_loss,
            "training_history": self.training_results.training_history,
            "total_epochs": len(self.training_results.training_history),
            "batch_size": batch_size,
        }

    def _prepare_training_state(self, batch_size: int) -> None:
        self.config.batch_size = batch_size
        self.predictor.training_history = []
        self.training_results.training_history.clear()
        self.training_results.curriculum_history.clear()
        self.training_results.best_epoch = 0
        self.training_results.best_validation_loss = float("inf")
        self.training_results.final_horizons = list(self.config.prediction_horizons)
        self._setup_optimizer_and_scheduler()

    def _setup_optimizer_and_scheduler(self) -> None:
        self.predictor.optimizer = torch.optim.AdamW(
            self.predictor.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler_patience = max(1, int(getattr(self.config, "lr_scheduler_patience", 5)))
        scheduler_factor = float(getattr(self.config, "lr_scheduler_factor", 0.5))
        self.predictor.scheduler = ReduceLROnPlateau(
            self.predictor.optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
        )

    def _build_epoch_loaders(
        self,
        train_windows: Dict[str, np.ndarray],
        val_windows: Dict[str, np.ndarray],
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = self.predictor._create_dataloader(
            train_windows, batch_size, shuffle=True
        )
        val_loader = self.predictor._create_dataloader(
            val_windows, batch_size, shuffle=False
        )
        return train_loader, val_loader

    def _apply_curriculum(self) -> None:
        if not self.curriculum_scheduler:
            return
        horizons = self.curriculum_scheduler.get_current_horizons()
        if horizons:
            self.config.prediction_horizons = horizons

    def _step_scheduler(self, val_loss: float) -> None:
        if self.predictor.scheduler is not None:
            self.predictor.scheduler.step(val_loss)

    def _record_curriculum(self, epoch: int, val_loss: float) -> None:
        if not self.curriculum_scheduler:
            return
        curriculum_results = self.curriculum_scheduler.step(epoch, val_loss)
        if curriculum_results:
            self.training_results.curriculum_history.append(curriculum_results)
            horizons = curriculum_results.get("current_horizons")
            if horizons:
                self.config.prediction_horizons = horizons

    def _update_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        batch_size: int,
        best_val_loss: float,
        best_state: Optional[Dict[str, torch.Tensor]],
        patience_counter: int,
    ) -> Tuple[float, Optional[Dict[str, torch.Tensor]], int]:
        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = self._snapshot_model()
            patience_counter = 0
            self.training_results.best_epoch = epoch
        else:
            patience_counter += 1

        self._append_history(epoch, train_loss, val_loss, batch_size)
        self._log_epoch(epoch, train_loss, val_loss, batch_size)

        return best_val_loss, best_state, patience_counter

    def _append_history(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        batch_size: int,
    ) -> None:
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": self._current_learning_rate(),
            "batch_size": batch_size,
        }
        self.predictor.training_history.append(history_entry)
        self.training_results.training_history.append(history_entry)

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        batch_size: int,
    ) -> None:
        if epoch % 5 != 0 and epoch >= 5:
            return
        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | lr=%.2e | batch=%d",
            epoch,
            train_loss,
            val_loss,
            self._current_learning_rate(),
            batch_size,
        )

    def _should_stop(self, patience_counter: int, patience_limit: int, epoch: int) -> bool:
        if patience_counter < patience_limit:
            return False
        logger.info("⏹️ Early stopping triggered at epoch %d", epoch + 1)
        return True

    def _snapshot_model(self) -> Dict[str, torch.Tensor]:
        return {
            key: tensor.detach().cpu()
            for key, tensor in self.predictor.model.state_dict().items()
        }

    def _current_learning_rate(self) -> float:
        if not self.predictor.optimizer:
            return 0.0
        return float(self.predictor.optimizer.param_groups[0]["lr"])

    def _finalize_training(self, best_val_loss: float, start_time: datetime) -> None:
        duration = (datetime.now() - start_time).total_seconds()
        self.training_results.training_duration = duration
        self.training_results.best_validation_loss = best_val_loss
        self.training_results.model_size = int(
            sum(parameter.numel() for parameter in self.predictor.model.parameters())
        )
        self.training_results.total_parameters = self.training_results.model_size
        self.training_results.final_horizons = list(self.config.prediction_horizons)