"""
Temporal Fusion Transformer (TFT) for financial forecasting

This module implements the Temporal Fusion Transformer architecture for multi-horizon financial
forecasting with uncertainty quantification. TFT handles mixed static/temporal variables and
provides variable-level feature importance via self-attention mechanisms.

Based on the paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
by Lim et al. (2020).
"""

import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class TimeDistributed(nn.Module):
    """Apply a layer to each time step of a sequence."""

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x shape: [batch, time, features]
        batch_size, time_steps, features = x.size()

        # Reshape to [batch * time, features]
        x_reshaped = x.contiguous().view(-1, features)

        # Apply module
        y = self.module(x_reshaped)

        # Reshape back to [batch, time, module_output_features]
        output_features = y.size(-1)
        y = y.contiguous().view(batch_size, time_steps, output_features)

        return y


class GLU(nn.Module):
    """Gated Linear Unit for TFT."""

    def __init__(self, input_size, hidden_size=None):
        super(GLU, self).__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for variable selection and processing."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        dropout=0.1,
        context_size=None,
        batch_norm=False
    ):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.context_size = context_size
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Main network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

        # Skip connection if input and output sizes differ
        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = None

        # Context network (for conditioning)
        if context_size is not None:
            self.context_layer = nn.Linear(
                context_size, hidden_size, bias=False)

        # Gating layer
        self.gating_layer = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

        # Batch norm layer
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(output_size)

    def forward(self, x, context=None):
        # Main network branch
        residual = x

        # First fully connected layer
        x = self.fc1(x)

        # Add context if provided
        if context is not None and self.context_size is not None:
            x = x + self.context_layer(context)

        # Non-linearity and second fully connected layer
        x = self.elu(x)
        x = self.fc2(self.dropout_layer(x))

        # Skip connection
        if self.skip_layer is not None:
            residual = self.skip_layer(residual)

        # Gating mechanism
        gating = self.sigmoid(self.gating_layer(residual))
        output = residual + gating * x

        # Apply batch norm if specified
        if self.batch_norm and output.shape[0] > 1:  # Batch size > 1
            # Handle 2D and 3D tensors
            if len(output.shape) == 2:
                output = self.batch_norm_layer(output)
            else:  # 3D tensor [batch, time, features]
                original_shape = output.shape
                output = self.batch_norm_layer(
                    output.reshape(-1, original_shape[-1])
                ).reshape(original_shape)

        return output


class TemporalSelfAttention(nn.Module):
    """Temporal self-attention mechanism for the Transformer."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        use_kv_norm=False
    ):
        super(TemporalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_kv_norm = use_kv_norm

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}")

        # Linear layers for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Key-value normalization (optional)
        if use_kv_norm:
            self.k_norm = nn.LayerNorm(self.head_dim)
            self.v_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, attn_mask=None):
        # x shape: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.size()

        # Project inputs to queries, keys, values
        q = self.q_proj(x).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim)
        k = self.k_proj(x).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim)
        v = self.v_proj(x).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim)

        # Apply key-value normalization if specified
        if self.use_kv_norm:
            k = self.k_norm(k)
            v = self.v_norm(v)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back
        # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention with separate attention layers per head.
    This allows extracting head-specific variable importance.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1
    ):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Create separate attention layers for each head
        self.attention_layers = nn.ModuleList([
            TemporalSelfAttention(
                embed_dim=embed_dim,
                num_heads=1,  # Each head gets its own attention mechanism
                dropout=dropout,
                use_kv_norm=True
            )
            for _ in range(num_heads)
        ])

        # Final output projection
        self.out_proj = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x, attn_mask=None):
        # Apply each attention head separately
        outputs = []
        attention_weights = []

        for layer in self.attention_layers:
            output, weights = layer(x, attn_mask)
            outputs.append(output)
            attention_weights.append(weights)

        # Concatenate outputs from all heads
        combined = torch.cat(outputs, dim=-1)

        # Project back to embedding dimension
        output = self.out_proj(combined)

        # Stack attention weights for all heads
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.cat(attention_weights, dim=1)

        return output, attn_weights


class TFTEncoder(nn.Module):
    """
    Encoder for Temporal Fusion Transformer.

    Processes input variables, applies variable selection, and enriches temporal features.
    """

    def __init__(
        self,
        input_size,
        static_cov_size,
        time_varying_cov_size,
        hidden_size,
        dropout=0.1,
        batch_norm=False
    ):
        super(TFTEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.static_cov_size = static_cov_size
        self.time_varying_cov_size = time_varying_cov_size

        # Static covariate encoders
        if static_cov_size > 0:
            self.static_context_grn = GatedResidualNetwork(
                input_size=static_cov_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                batch_norm=batch_norm
            )

            self.static_variable_selection = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=time_varying_cov_size,
                dropout=dropout,
                batch_norm=batch_norm
            )

        # Time-varying encoders for covariates
        self.variable_encoders = nn.ModuleList([
            TimeDistributed(
                GatedResidualNetwork(
                    input_size=1,  # Each variable is encoded separately
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
            )
            for _ in range(time_varying_cov_size)
        ])

        # Variable selection GRN
        self.variable_selection = TimeDistributed(
            GatedResidualNetwork(
                input_size=hidden_size * time_varying_cov_size,
                hidden_size=hidden_size,
                output_size=time_varying_cov_size,
                dropout=dropout,
                context_size=hidden_size,  # Static context
                batch_norm=batch_norm
            )
        )

        # Historical GRN for transforming selected variables
        self.encoder_grn = TimeDistributed(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                batch_norm=batch_norm
            )
        )

    def forward(self, x, static_covs=None):
        """
        Forward pass for the encoder.

        Args:
            x: Time-varying covariates [batch, time, features]
            static_covs: Static covariates [batch, features]

        Returns:
            Tuple of (encoded_inputs, variable_weights)
        """
        batch_size, seq_len, _ = x.size()

        # Process static covariates if provided
        if static_covs is not None and self.static_cov_size > 0:
            static_context = self.static_context_grn(static_covs)
            static_var_weights = self.static_variable_selection(static_context)
            static_var_weights = torch.softmax(static_var_weights, dim=-1)

            # Broadcast static context to temporal dimension
            static_context = static_context.unsqueeze(
                1).expand(-1, seq_len, -1)
        else:
            static_context = None
            static_var_weights = None

        # Split the input into separate variables
        var_splits = torch.split(x, 1, dim=-1)

        # Encode each variable separately
        var_encodings = []
        for i, (var, encoder) in enumerate(
                zip(var_splits, self.variable_encoders)):
            encoding = encoder(var)  # [batch, time, hidden]
            var_encodings.append(encoding)

        # Concatenate all variable encodings
        # [batch, time, hidden*vars]
        var_encodings = torch.cat(var_encodings, dim=-1)

        # Variable selection
        if static_context is not None:
            var_weights = self.variable_selection(
                var_encodings, static_context)
        else:
            # Create dummy context if none provided
            dummy_context = torch.zeros(
                (batch_size, seq_len, self.hidden_size),
                device=x.device
            )
            var_weights = self.variable_selection(var_encodings, dummy_context)

        var_weights = torch.softmax(var_weights, dim=-1)

        # Apply variable selection weights to create combined representation
        var_encodings = torch.stack(
            var_encodings.chunk(
                self.time_varying_cov_size, dim=-1))
        var_encodings = torch.sum(
            var_encodings * var_weights.unsqueeze(-1),  # Apply weights
            dim=0  # Sum across variables
        )

        # Apply encoder GRN
        encoded = self.encoder_grn(var_encodings)

        return encoded, var_weights, static_var_weights


class TFTDecoder(nn.Module):
    """
    Decoder for Temporal Fusion Transformer.

    Includes temporal self-attention and processes future inputs for multi-horizon forecasting.
    """

    def __init__(
        self,
        hidden_size,
        output_size,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        batch_norm=False
    ):
        super(TFTDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Initial layer to combine encoded inputs
        self.initial_layer = TimeDistributed(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                batch_norm=batch_norm
            )
        )

        # Attention blocks
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Position-wise feed forward networks
        self.ffn_layers = nn.ModuleList([
            TimeDistributed(
                GatedResidualNetwork(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
            )
            for _ in range(num_layers)
        ])

        # Layer normalization layers
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])

        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])

        # Output layer for mean prediction
        self.mean_projection = TimeDistributed(
            nn.Linear(hidden_size, output_size))

        # Output layer for scale parameter (for quantile prediction)
        self.scale_projection = TimeDistributed(
            nn.Linear(hidden_size, output_size))

    def forward(self, x, attention_mask=None, return_attentions=False):
        """
        Forward pass for the decoder.

        Args:
            x: Encoded inputs [batch, time, hidden]
            attention_mask: Optional mask for attention [batch, time, time]
            return_attentions: Whether to return attention weights

        Returns:
            Tuple of (mean, scale) or (mean, scale, attention_weights) if return_attentions
        """
        # Initial layer
        x = self.initial_layer(x)

        # Store attention weights if requested
        all_attentions = []

        # Apply attention blocks
        for i in range(self.num_layers):
            residual = x

            # Layer norm before attention
            x = self.layer_norms1[i](x)

            # Self-attention
            x_attn, attn_weights = self.attention_layers[i](x, attention_mask)
            if return_attentions:
                all_attentions.append(attn_weights)

            # Residual connection
            x = residual + x_attn

            # FFN with residual connection
            residual = x
            x = self.layer_norms2[i](x)
            x = residual + self.ffn_layers[i](x)

        # Project to output
        mean = self.mean_projection(x)
        scale = F.softplus(self.scale_projection(x)) + 1e-3  # Ensure positive

        if return_attentions:
            return mean, scale, all_attentions
        else:
            return mean, scale


class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty estimation."""

    def __init__(self, quantiles=None):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(self, preds, target):
        """
        Calculate quantile loss.

        Args:
            preds: Quantile predictions [batch, time, quantiles]
            target: Actual values [batch, time, 1]

        Returns:
            Quantile loss
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i:i+1]
            losses.append(torch.max((q - 1) * errors,
                          q * errors).unsqueeze(-1))

        # Stack losses for all quantiles
        loss = torch.cat(losses, dim=-1)
        return loss.mean()


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) model for interpretable time series forecasting.

    Based on the paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    by Lim et al. (2020).
    """

    def __init__(
        self,
        input_size,
        output_size=1,
        static_cov_size=0,
        time_varying_cov_size=None,
        hidden_size=64,
        lstm_hidden_size=128,
        lstm_layers=1,
        num_attention_heads=4,
        num_attention_layers=1,
        dropout=0.1,
        batch_norm=False,
        quantiles=None
    ):
        """
        Initialize the TFT model.

        Args:
            input_size: Total input size
            output_size: Output size (typically 1 for univariate forecasting)
            static_cov_size: Number of static covariates
            time_varying_cov_size: Number of time-varying covariates (defaults to input_size)
            hidden_size: Hidden size for all components
            lstm_hidden_size: LSTM hidden size
            lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            num_attention_layers: Number of attention layers
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            quantiles: Quantiles to predict for uncertainty estimation
        """
        super(TemporalFusionTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.static_cov_size = static_cov_size
        self.time_varying_cov_size = time_varying_cov_size or input_size
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        # Encoder
        self.encoder = TFTEncoder(
            input_size=input_size,
            static_cov_size=static_cov_size,
            time_varying_cov_size=self.time_varying_cov_size,
            hidden_size=hidden_size,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # LSTM for temporal processing
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # Reduce bidirectional LSTM output to hidden_size
        self.post_lstm_gate = TimeDistributed(
            GLU(lstm_hidden_size * 2, hidden_size)
        )

        # Enrichment layer (for combining LSTM and direct inputs)
        self.enrichment = TimeDistributed(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                batch_norm=batch_norm
            )
        )

        # Decoder
        self.decoder = TFTDecoder(
            hidden_size=hidden_size,
            # One output per quantile
            output_size=output_size * len(quantiles),
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # Loss function
        self.loss_fn = QuantileLoss(quantiles=quantiles)

    def forward(
        self,
        x,
        static_covs=None,
        attention_mask=None,
        return_attentions=False,
        return_decomposition=False
    ):
        """
        Forward pass for the TFT model.

        Args:
            x: Input tensor [batch, time, features]
            static_covs: Static covariates [batch, features]
            attention_mask: Optional mask for attention
            return_attentions: Whether to return attention weights
            return_decomposition: Whether to return variable decomposition

        Returns:
            Dict containing model outputs and optionally attention weights and decomposition
        """
        batch_size, seq_len, _ = x.size()

        # Encoder
        encoded, var_weights, static_weights = self.encoder(x, static_covs)

        # LSTM
        lstm_output, _ = self.lstm_encoder(encoded)

        # Post-LSTM gate
        temporal = self.post_lstm_gate(lstm_output)

        # Enrichment layer
        enriched = self.enrichment(temporal)

        # Decoder
        if return_attentions:
            mean, scale, attentions = self.decoder(
                enriched,
                attention_mask=attention_mask,
                return_attentions=True
            )
        else:
            mean, scale = self.decoder(
                enriched,
                attention_mask=attention_mask,
                return_attentions=False
            )
            attentions = None

        # Reshape for quantile outputs
        n_quantiles = len(self.quantiles)
        if n_quantiles > 1:
            # Reshape to [batch, time, output_size, n_quantiles]
            mean = mean.view(
                batch_size,
                seq_len,
                self.output_size,
                n_quantiles)

            # Different scale per quantile
            scale = scale.view(
                batch_size,
                seq_len,
                self.output_size,
                n_quantiles)

            # Create quantile predictions
            quantile_preds = mean + scale * torch.randn_like(mean)

            # Transpose for loss calculation [batch, time, n_quantiles,
            # output_size]
            quantile_preds = quantile_preds.permute(0, 1, 3, 2)

            # Flatten quantile dimension [batch, time, n_quantiles *
            # output_size]
            quantile_preds = quantile_preds.reshape(batch_size, seq_len, -1)
        else:
            quantile_preds = mean

        results = {
            'mean': mean,
            'scale': scale,
            'predictions': quantile_preds,
            'var_weights': var_weights,
            'static_weights': static_weights
        }

        if return_attentions:
            results['attentions'] = attentions

        if return_decomposition:
            # Simple decomposition using variable weights
            decomposition = []
            for i in range(self.time_varying_cov_size):
                # Extract weight for this variable
                weight = var_weights[:, :, i:i+1]

                # Apply weight to the mean prediction
                contribution = weight * mean  # Simplified approximation
                decomposition.append(contribution)

            results['decomposition'] = torch.stack(decomposition, dim=-1)

        return results

    def calculate_loss(self, preds, targets):
        """
        Calculate quantile loss.

        Args:
            preds: Dict from forward pass
            targets: Target values [batch, time, output_size]

        Returns:
            Loss value
        """
        return self.loss_fn(preds['predictions'], targets)


class TFTDataModule:
    """
    Data module for preparing data for the TFT model.

    Handles scaling, windowing, and batching of time series data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        timestamp_col: str,
        static_cols: List[str] = None,
        known_future_cols: List[str] = None,
        unknown_future_cols: List[str] = None,
        context_length: int = 30,
        prediction_length: int = 5,
        batch_size: int = 32,
        scaling: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.1
    ):
        """
        Initialize the data module.

        Args:
            data: Pandas DataFrame with time series data
            target_col: Column name for the target variable
            timestamp_col: Column name for timestamps
            static_cols: List of static column names (same for all time steps)
            known_future_cols: List of columns known in the future (e.g., calendar features)
            unknown_future_cols: List of columns unknown in the future
            context_length: Number of time steps for context
            prediction_length: Number of time steps to predict
            batch_size: Batch size for training
            scaling: Whether to scale the data
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        self.data = data.copy()
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.static_cols = static_cols or []
        self.known_future_cols = known_future_cols or []
        self.unknown_future_cols = unknown_future_cols or []
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.scaling = scaling
        self.val_split = val_split
        self.test_split = test_split

        # Initialize random number generator for modern numpy
        self._rng = np.random.default_rng(42)

        # Determine all feature columns
        self.feature_cols = self.static_cols + \
            self.known_future_cols + self.unknown_future_cols

        # Ensure data is sorted by timestamp
        if self.timestamp_col in self.data.columns:
            self.data = self.data.sort_values(
                self.timestamp_col).reset_index(
                drop=True)

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for training."""
        # Extract feature and target data
        self.features = self.data[self.feature_cols].values.astype(np.float32)
        self.targets = self.data[self.target_col].values.astype(np.float32)

        # Initialize scalers
        if self.scaling:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

            # Fit and transform feature data
            self.features = self.feature_scaler.fit_transform(self.features)

            # Fit and transform target data
            self.targets = self.target_scaler.fit_transform(
                self.targets.reshape(-1, 1)).flatten()

        # Split data into train, validation, and test sets
        total_samples = len(self.data)
        test_size = int(total_samples * self.test_split)
        val_size = int(total_samples * self.val_split)
        train_size = total_samples - val_size - test_size

        self.train_features = self.features[:train_size]
        self.train_targets = self.targets[:train_size]

        self.val_features = self.features[train_size:train_size + val_size]
        self.val_targets = self.targets[train_size:train_size + val_size]

        self.test_features = self.features[train_size + val_size:]
        self.test_targets = self.targets[train_size + val_size:]

        # Create windowed batches
        self.train_data = self._create_windows(
            self.train_features, self.train_targets)
        self.val_data = self._create_windows(
            self.val_features, self.val_targets)
        self.test_data = self._create_windows(
            self.test_features, self.test_targets)

    def _create_windows(self, features, targets):
        """Create windowed batches from the data."""
        windows = []
        total_length = len(features)

        for i in range(
                total_length -
                self.context_length -
                self.prediction_length +
                1):
            # Extract context window
            feature_window = features[i:i +
                                      self.context_length +
                                      self.prediction_length]
            target_window = targets[i:i +
                                    self.context_length +
                                    self.prediction_length]

            # Split into context and prediction horizon
            context_features = feature_window[:self.context_length]
            future_features = feature_window[self.context_length:]

            context_targets = target_window[:self.context_length]
            future_targets = target_window[self.context_length:]

            windows.append({
                'context_features': context_features,
                'context_targets': context_targets,
                'future_features': future_features,
                'future_targets': future_targets
            })

        return windows

    def _extract_static_features(self, features):
        """Extract static features from the feature array."""
        if not self.static_cols:
            return None

        # Static features are the same for all time steps, so take the first
        # one
        static_indices = [self.feature_cols.index(
            col) for col in self.static_cols]
        return features[:, 0, static_indices]

    def get_train_dataloader(self):
        """Get training data loader."""
        self._rng.shuffle(self.train_data)

        for i in range(0, len(self.train_data), self.batch_size):
            batch = self.train_data[i:i + self.batch_size]

            # Prepare batch data
            context_features = np.stack([x['context_features'] for x in batch])
            context_targets = np.stack([x['context_targets'] for x in batch])
            future_features = np.stack([x['future_features'] for x in batch])
            future_targets = np.stack([x['future_targets'] for x in batch])

            # Extract static features
            static_features = self._extract_static_features(context_features)

            # Combine context and future for encoder-decoder architecture
            combined_features = np.concatenate(
                [context_features, future_features], axis=1)
            combined_targets = np.concatenate(
                [context_targets, future_targets], axis=1)

            # Convert to tensors
            combined_features = torch.tensor(
                combined_features, dtype=torch.float32)
            combined_targets = torch.tensor(
                combined_targets, dtype=torch.float32)

            if static_features is not None:
                static_features = torch.tensor(
                    static_features, dtype=torch.float32)

            # Create attention mask (causal mask for decoder)
            seq_len = combined_features.shape[1]
            attention_mask = torch.ones(seq_len, seq_len)
            attention_mask = torch.tril(
                attention_mask, diagonal=self.context_length)

            # Broadcast to batch size
            attention_mask = attention_mask.unsqueeze(0).expand(
                combined_features.shape[0], -1, -1
            )

            yield {
                'features': combined_features,
                # Add output dimension
                'targets': combined_targets.unsqueeze(-1),
                'static_features': static_features,
                'attention_mask': attention_mask,
                'context_length': self.context_length,
                'prediction_length': self.prediction_length
            }

    def get_val_dataloader(self):
        """Get validation data loader."""
        # Similar to train dataloader but without shuffling
        for i in range(0, len(self.val_data), self.batch_size):
            batch = self.val_data[i:i + self.batch_size]

            # Prepare batch data
            context_features = np.stack([x['context_features'] for x in batch])
            context_targets = np.stack([x['context_targets'] for x in batch])
            future_features = np.stack([x['future_features'] for x in batch])
            future_targets = np.stack([x['future_targets'] for x in batch])

            # Extract static features
            static_features = self._extract_static_features(context_features)

            # Combine context and future for encoder-decoder architecture
            combined_features = np.concatenate(
                [context_features, future_features], axis=1)
            combined_targets = np.concatenate(
                [context_targets, future_targets], axis=1)

            # Convert to tensors
            combined_features = torch.tensor(
                combined_features, dtype=torch.float32)
            combined_targets = torch.tensor(
                combined_targets, dtype=torch.float32)

            if static_features is not None:
                static_features = torch.tensor(
                    static_features, dtype=torch.float32)

            # Create attention mask (causal mask for decoder)
            seq_len = combined_features.shape[1]
            attention_mask = torch.ones(seq_len, seq_len)
            attention_mask = torch.tril(
                attention_mask, diagonal=self.context_length)

            # Broadcast to batch size
            attention_mask = attention_mask.unsqueeze(0).expand(
                combined_features.shape[0], -1, -1
            )

            yield {
                'features': combined_features,
                # Add output dimension
                'targets': combined_targets.unsqueeze(-1),
                'static_features': static_features,
                'attention_mask': attention_mask,
                'context_length': self.context_length,
                'prediction_length': self.prediction_length
            }

    def get_test_dataloader(self):
        """Get test data loader."""
        # Similar to validation dataloader
        for i in range(0, len(self.test_data), self.batch_size):
            batch = self.test_data[i:i + self.batch_size]

            # Prepare batch data
            context_features = np.stack([x['context_features'] for x in batch])
            context_targets = np.stack([x['context_targets'] for x in batch])
            future_features = np.stack([x['future_features'] for x in batch])
            future_targets = np.stack([x['future_targets'] for x in batch])

            # Extract static features
            static_features = self._extract_static_features(context_features)

            # Combine context and future for encoder-decoder architecture
            combined_features = np.concatenate(
                [context_features, future_features], axis=1)
            combined_targets = np.concatenate(
                [context_targets, future_targets], axis=1)

            # Convert to tensors
            combined_features = torch.tensor(
                combined_features, dtype=torch.float32)
            combined_targets = torch.tensor(
                combined_targets, dtype=torch.float32)

            if static_features is not None:
                static_features = torch.tensor(
                    static_features, dtype=torch.float32)

            # Create attention mask (causal mask for decoder)
            seq_len = combined_features.shape[1]
            attention_mask = torch.ones(seq_len, seq_len)
            attention_mask = torch.tril(
                attention_mask, diagonal=self.context_length)

            # Broadcast to batch size
            attention_mask = attention_mask.unsqueeze(0).expand(
                combined_features.shape[0], -1, -1
            )

            yield {
                'features': combined_features,
                # Add output dimension
                'targets': combined_targets.unsqueeze(-1),
                'static_features': static_features,
                'attention_mask': attention_mask,
                'context_length': self.context_length,
                'prediction_length': self.prediction_length
            }

    def inverse_transform_targets(self, targets):
        """Transform scaled targets back to original scale."""
        if self.scaling:
            return self.target_scaler.inverse_transform(targets).flatten()
        return targets

    def inverse_transform_features(self, features):
        """Transform scaled features back to original scale."""
        if self.scaling:
            return self.feature_scaler.inverse_transform(features)
        return features


def _create_tft_model(
        data_module,
        hidden_size,
        lstm_hidden_size,
        num_attention_heads,
        num_attention_layers,
        dropout):
    """Create TFT model with specified architecture"""
    input_size = len(data_module.feature_cols)
    static_cov_size = len(data_module.static_cols)
    time_varying_cov_size = input_size - static_cov_size

    model = TemporalFusionTransformer(
        input_size=input_size,
        output_size=1,  # Single target variable
        static_cov_size=static_cov_size,
        time_varying_cov_size=time_varying_cov_size,
        hidden_size=hidden_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=1,
        num_attention_heads=num_attention_heads,
        num_attention_layers=num_attention_layers,
        dropout=dropout,
        batch_norm=True,
        quantiles=[0.1, 0.5, 0.9]
    )
    return model


def _train_epoch(model, data_module, optimizer, device):
    """Train one epoch and return average loss"""
    model.train()
    train_loss = 0.0
    train_batches = 0

    for batch in data_module.get_train_dataloader():
        # Move data to device
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        static_features = batch['static_features'].to(
            device) if batch['static_features'] is not None else None
        attention_mask = batch['attention_mask'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            features,
            static_covs=static_features,
            attention_mask=attention_mask
        )

        # Calculate loss
        loss = model.calculate_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        train_loss += loss.item()
        train_batches += 1

    return train_loss / train_batches


def _validate_epoch(model, data_module, device):
    """Validate one epoch and return average loss"""
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in data_module.get_val_dataloader():
            # Move data to device
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            static_features = batch['static_features'].to(
                device) if batch['static_features'] is not None else None
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(
                features,
                static_covs=static_features,
                attention_mask=attention_mask
            )

            # Calculate loss
            loss = model.calculate_loss(outputs, targets)

            # Track loss
            val_loss += loss.item()
            val_batches += 1

    return val_loss / val_batches


def train_tft_model(
    data_module: TFTDataModule,
    hidden_size: int = 64,
    lstm_hidden_size: int = 128,
    num_attention_heads: int = 4,
    num_attention_layers: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    num_epochs: int = 50,
    early_stopping_patience: int = 5,
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Train the TFT model.

    Args:
        data_module: Data module for the TFT model
        hidden_size: Hidden size for all components
        lstm_hidden_size: LSTM hidden size
        num_attention_heads: Number of attention heads
        num_attention_layers: Number of attention layers
        dropout: Dropout rate
        learning_rate: Learning rate for optimization
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        device: Device to use for training ('cuda' or 'cpu')
        verbose: Whether to print training progress

    Returns:
        Trained TFT model and training history
    """
    # Configure device
    device = torch.device(device if torch.cuda.is_available()
                          and device == 'cuda' else 'cpu')

    if verbose:
        print(f"Using device: {device}")

    # Create model
    model = _create_tft_model(
        data_module,
        hidden_size,
        lstm_hidden_size,
        num_attention_heads,
        num_attention_layers,
        dropout)
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5)

    # Initialize early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    # Training history
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training
        train_loss = _train_epoch(model, data_module, optimizer, device)
        history['train_loss'].append(train_loss)

        # Validation
        val_loss = _validate_epoch(model, data_module, device)
        history['val_loss'].append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Set model to evaluation mode
    model.eval()

    return model, history


def evaluate_tft_model(model, data_module, device='cuda'):
    """
    Evaluate the TFT model on test data.

    Args:
        model: Trained TFT model
        data_module: Data module for the TFT model
        device: Device to use for evaluation

    Returns:
        Dict with evaluation metrics
    """
    device = torch.device(device if torch.cuda.is_available()
                          and device == 'cuda' else 'cpu')
    model.to(device)
    model.eval()

    # Collect predictions and targets
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_module.get_test_dataloader():
            # Move data to device
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            static_features = batch['static_features'].to(
                device) if batch['static_features'] is not None else None
            attention_mask = batch['attention_mask'].to(device)
            context_length = batch['context_length']

            # Forward pass
            outputs = model(
                features,
                static_covs=static_features,
                attention_mask=attention_mask,
                return_attentions=False,
                return_decomposition=False
            )

            # Get median predictions (quantile 0.5)
            predictions = outputs['predictions'][
                :, context_length:, 1]  # Middle quantile
            targets = targets[:, context_length:, 0]

            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Inverse transform if scaling was applied
    if data_module.scaling:
        all_predictions = data_module.inverse_transform_targets(
            all_predictions.reshape(-1, 1))
        all_targets = data_module.inverse_transform_targets(
            all_targets.reshape(-1, 1))

    # Calculate metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    mape = np.mean(np.abs((all_predictions - all_targets) / all_targets)) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def _prepare_context_data(context_data, data_module):
    """Prepare context data for forecasting"""
    if isinstance(context_data, pd.DataFrame):
        # Extract features from DataFrame
        context_features = context_data[data_module.feature_cols].values
        if data_module.scaling:
            context_features = data_module.feature_scaler.transform(
                context_features)
    else:
        context_features = context_data

    # Ensure context_features has the right shape
    if len(context_features) < data_module.context_length:
        raise ValueError(
            f"Context data must contain at least {
                data_module.context_length} steps")

    # Use the last context_length steps
    context_features = context_features[-data_module.context_length:]

    return context_features


def _extract_static_features(context_features, data_module, device):
    """Extract static features if any"""
    static_features = None
    if data_module.static_cols:
        static_indices = [data_module.feature_cols.index(
            col) for col in data_module.static_cols]
        static_features = context_features[0, static_indices]
        static_features = torch.tensor(
            static_features,
            dtype=torch.float32).unsqueeze(0).to(device)
    return static_features


def _generate_single_forecast(
        model,
        context_tensor,
        forecast_tensor,
        static_features,
        device,
        step):
    """Generate a single forecast step"""
    if step == 0:
        # Create attention mask (causal mask for decoder)
        seq_len = context_tensor.shape[1]
        attention_mask = torch.ones(seq_len, seq_len)
        attention_mask = torch.tril(attention_mask).unsqueeze(0).to(device)

        # Forward pass
        outputs = model(
            context_tensor,
            static_covs=static_features,
            attention_mask=attention_mask,
            return_attentions=False
        )
    else:
        # Add the previous prediction to the context
        combined_tensor = torch.cat([context_tensor, forecast_tensor], dim=1)

        # Create attention mask (causal mask for decoder)
        seq_len = combined_tensor.shape[1]
        attention_mask = torch.ones(seq_len, seq_len)
        attention_mask = torch.tril(attention_mask).unsqueeze(0).to(device)

        # Forward pass
        outputs = model(
            combined_tensor,
            static_covs=static_features,
            attention_mask=attention_mask,
            return_attentions=False
        )

    # Get predictions for all quantiles
    preds = outputs['predictions'][:, -1].cpu().numpy()
    return preds


def _prepare_next_features(features, device):
    """Prepare features for next forecasting step"""
    # Create a feature vector for the next step
    # For simplicity, we repeat the last feature vector
    next_features = features[-1].copy()

    # Update any target-dependent features
    # This is a simplification - in practice, you'd need to update all
    # features properly

    # Create tensor for next step
    forecast_tensor = torch.tensor(
        next_features,
        dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return forecast_tensor


def _format_forecast_results(
        forecasts,
        model,
        data_module,
        forecast_steps,
        return_confidence):
    """Format forecast results into DataFrame"""
    # Convert forecasts to array
    forecasts = np.array(forecasts)

    # Get quantile predictions
    n_quantiles = len(model.quantiles)

    # Reshape forecasts to [steps, quantiles]
    if n_quantiles > 1:
        forecasts = forecasts.reshape(forecast_steps, n_quantiles)

    # Inverse transform if scaling was applied
    if data_module.scaling:
        forecasts = data_module.inverse_transform_targets(
            forecasts.reshape(-1, 1)).reshape(forecast_steps, n_quantiles)

    # Create forecast DataFrame
    if n_quantiles > 1 and return_confidence:
        result = pd.DataFrame({
            'forecast': forecasts[:, 1],  # Median
            'lower': forecasts[:, 0],     # Lower quantile
            'upper': forecasts[:, 2]      # Upper quantile
        })
    else:
        result = pd.DataFrame({
            # Median or only quantile
            'forecast': forecasts[:, 1] if n_quantiles > 1 else forecasts
        })

    return result


def forecast_with_tft(
    model,
    data_module,
    context_data,
    forecast_steps=30,
    device='cuda',
    return_confidence=True
):
    """
    Generate forecasts using the trained TFT model.

    Args:
        model: Trained TFT model
        data_module: Data module used for training
        context_data: Data for context window
        forecast_steps: Number of steps to forecast
        device: Device to use for forecasting
        return_confidence: Whether to return confidence intervals

    Returns:
        DataFrame with forecasts
    """
    device = torch.device(device if torch.cuda.is_available()
                          and device == 'cuda' else 'cpu')
    model.to(device)
    model.eval()

    # Prepare context data
    context_features = _prepare_context_data(context_data, data_module)

    # Extract static features if any
    static_features = _extract_static_features(
        context_features, data_module, device)

    # Initialize arrays for forecasting
    features = context_features.copy()
    forecasts = []

    # Convert to tensor
    context_tensor = torch.tensor(
        features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        forecast_tensor = None

        # Generate forecasts one step at a time
        for step in range(forecast_steps):
            # Generate forecast for current step
            preds = _generate_single_forecast(
                model,
                context_tensor,
                forecast_tensor,
                static_features,
                device,
                step)

            # Store predictions
            forecasts.append(preds[0])

            # Create features for next step
            if step < forecast_steps - 1:
                forecast_tensor = _prepare_next_features(features, device)

    # Format and return results
    return _format_forecast_results(
        forecasts,
        model,
        data_module,
        forecast_steps,
        return_confidence)


def plot_tft_forecast(
    historical_data,
    forecast_data,
    target_col,
    figsize=(
        12,
        6)):
    """
    Plot historical data and forecasts with confidence intervals.

    Args:
        historical_data: DataFrame with historical data
        forecast_data: DataFrame with forecast data
        target_col: Column name for target variable
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical data
    ax.plot(
        historical_data.index,
        historical_data[target_col],
        'b-',
        label='Historical')

    # Create forecast index
    forecast_index = pd.date_range(
        start=historical_data.index[-1],
        periods=len(forecast_data) + 1,
        freq=historical_data.index.to_series().diff().mode()[0]
    )[1:]

    # Plot forecast
    ax.plot(forecast_index, forecast_data['forecast'], 'r-', label='Forecast')

    # Plot confidence interval if available
    if 'lower' in forecast_data.columns and 'upper' in forecast_data.columns:
        ax.fill_between(
            forecast_index,
            forecast_data['lower'],
            forecast_data['upper'],
            color='r',
            alpha=0.2,
            label='Confidence Interval'
        )

    ax.set_title('TFT Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig


def plot_tft_attention(model, data_module, sample_batch, figsize=(15, 10)):
    """
    Plot attention weights from the TFT model.

    Args:
        model: Trained TFT model
        data_module: Data module used for training
        sample_batch: Sample batch of data
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Move data to device
    features = sample_batch['features'].to(device)
    static_features = sample_batch['static_features'].to(
        device) if sample_batch['static_features'] is not None else None
    attention_mask = sample_batch['attention_mask'].to(device)
    context_length = sample_batch['context_length']

    # Generate predictions with attention weights
    with torch.no_grad():
        outputs = model(
            features,
            static_covs=static_features,
            attention_mask=attention_mask,
            return_attentions=True,
            return_decomposition=True
        )

    # Get attention weights
    attentions = outputs['attentions']

    # Extract weights from the last attention layer
    # Shape: [batch_size, num_heads, seq_len, seq_len]
    attention_weights = attentions[-1][0]  # First sample in batch

    # Get variable importance
    # First sample in batch
    var_weights = outputs['var_weights'][0].cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot attention weights
    im = axes[0].imshow(
        attention_weights.mean(dim=0).cpu().numpy(),  # Average over heads
        cmap='viridis',
        aspect='auto'
    )

    # Add colorbar
    fig.colorbar(im, ax=axes[0])

    # Add labels
    axes[0].set_title('Attention Weights (Average Over Heads)')
    axes[0].set_xlabel('Sequence Position')
    axes[0].set_ylabel('Sequence Position')

    # Add vertical and horizontal lines to separate context and forecast
    axes[0].axvline(
        x=context_length - 0.5,
        color='r',
        linestyle='-',
        alpha=0.7)
    axes[0].axhline(
        y=context_length - 0.5,
        color='r',
        linestyle='-',
        alpha=0.7)

    # Plot variable importance
    var_names = data_module.feature_cols

    # Average over time steps for context and forecast periods
    context_weights = var_weights[:context_length].mean(axis=0)
    forecast_weights = var_weights[context_length:].mean(axis=0)

    # Plot as horizontal bars
    y_pos = np.arange(len(var_names))

    axes[1].barh(
        y_pos - 0.2,
        context_weights,
        height=0.4,
        color='blue',
        alpha=0.7,
        label='Context')
    axes[1].barh(
        y_pos + 0.2,
        forecast_weights,
        height=0.4,
        color='red',
        alpha=0.7,
        label='Forecast')

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(var_names)
    axes[1].set_title('Variable Importance')
    axes[1].set_xlabel('Importance')
    axes[1].legend()

    plt.tight_layout()

    return fig


# Usage example
if __name__ == "__main__":
    # Sample data preparation
    import yfinance as yf

    # Download Apple stock data
    data = yf.download('AAPL', start='2018-01-01', end='2023-01-01')

    # Create features
    data['Returns'] = data['Close'].pct_change()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_30'] = data['Close'].rolling(window=30).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['Volume_Change'] = data['Volume'].pct_change()

    # Drop missing values
    data = data.dropna()

    # Create date features
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['DayOfWeek'] = data.index.dayofweek

    # Prepare data module
    data_module = TFTDataModule(
        data=data.reset_index(),
        target_col='Close',
        timestamp_col='Date',
        known_future_cols=[
            'Year',
            'Month',
            'Day',
            'DayOfWeek'],
        unknown_future_cols=[
            'Open',
            'High',
            'Low',
            'Volume',
            'Returns',
            'MA_10',
            'MA_30',
            'Volatility',
            'Volume_Change'],
        context_length=60,
        prediction_length=30,
        batch_size=32,
        scaling=True)

    # Train model
    model, history = train_tft_model(
        data_module=data_module,
        hidden_size=64,
        lstm_hidden_size=128,
        num_attention_heads=4,
        num_attention_layers=1,
        dropout=0.1,
        learning_rate=1e-4,
        num_epochs=50,
        device='cpu'  # Change to 'cuda' if available
    )

    # Evaluate model
    metrics = evaluate_tft_model(model, data_module, device='cpu')
    print(f"Evaluation metrics: {metrics}")

    # Generate forecast
    forecast = forecast_with_tft(
        model=model,
        data_module=data_module,
        context_data=data.iloc[-60:],
        forecast_steps=30,
        device='cpu'
    )

    # Plot forecast
    fig = plot_tft_forecast(
        historical_data=data.iloc[-120:],
        forecast_data=forecast,
        target_col='Close'
    )
    plt.show()
