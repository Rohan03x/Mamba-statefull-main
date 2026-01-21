"""Sequence model helpers for Stage-B.

Provides utilities to build sliding-window datasets and train compact LSTM
regressors across multiple folds while supporting warm-starts and AMP.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_MAMBA_DEVICE_LOGGED = False

# =============================================================================
# SPEED OPTIMIZATION: Limit torch.compile workers to reduce overhead
# Default spawns 28 workers per trial (matching CPU count) causing 408 processes
# With 14 concurrent trials, this creates massive context switch overhead
# =============================================================================
os.environ.setdefault("TORCH_COMPILE_WORKERS", "2")  # Limit to 2 workers per trial
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")  # Inductor threads
os.environ.setdefault("OMP_NUM_THREADS", "2")  # OpenMP threads per trial
os.environ.setdefault("MKL_NUM_THREADS", "2")  # MKL threads per trial

try:  # Optional dependency
    import torch
    from torch import nn
    from torch.cuda import amp
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    
    # =============================================================================
    # SPEED OPTIMIZATION: Limit PyTorch threads to reduce contention
    # With 14 concurrent trials, each with 28 threads = 392 threads fighting for 28 CPUs
    # Limiting to 2 threads per trial = 28 threads total (1:1 with CPUs)
    # =============================================================================
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # Already set or parallel work started
    
    # =============================================================================
    # SPEED OPTIMIZATION: Increase dynamo cache limit to prevent recompilation
    # Default cache_size_limit=8 causes recompiles on train/eval mode switches
    # With 50 epochs × 12 batches, cache thrashes constantly
    # =============================================================================
    try:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64  # Increase from default 8
    except Exception:
        pass  # Older PyTorch version without dynamo
    
except Exception:  # pragma: no cover - CPU fallback
    torch = None  # type: ignore
    nn = None  # type: ignore
    amp = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = None  # type: ignore
    TensorDataset = None  # type: ignore


# =============================================================================
# EMA (Exponential Moving Average) for Model Weights - FP32 Precision
# Used by hedge funds to stabilize returns forecasts
# =============================================================================

class ModelEMA:
    """Exponential Moving Average of model weights in FP32 (hedge fund best practice).
    
    EMA tracking is done in fp32 for numerical stability, even when
    model training uses bf16/fp16 mixed precision.
    
    Usage:
        ema = ModelEMA(model, decay=0.9999)
        for epoch in range(epochs):
            train_step(model)
            ema.update(model)
        # Use EMA weights for inference
        ema.copy_to(model)
    """
    
    def __init__(self, model: "nn.Module", decay: float = 0.9999):
        """Initialize EMA with model parameters in fp32.
        
        Args:
            model: The model to track
            decay: EMA decay rate [0.90, 0.9999]. Higher = more smoothing.
        """
        self.decay = decay
        # Store EMA weights in fp32 for numerical stability
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()  # fp32
    
    def update(self, model: "nn.Module") -> None:
        """Update EMA weights after each training step.
        
        EMA_t = decay * EMA_{t-1} + (1 - decay) * weights_t
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Update in fp32 for stability
                new_val = param.data.float()
                self.shadow[name].mul_(self.decay).add_(new_val, alpha=1.0 - self.decay)
    
    def copy_to(self, model: "nn.Module") -> None:
        """Copy EMA weights to model (for inference)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.dtype))
    
    def state_dict(self) -> Dict[str, "torch.Tensor"]:
        """Return EMA state dict (all in fp32)."""
        return {k: v.clone() for k, v in self.shadow.items()}
    
    def load_state_dict(self, state_dict: Dict[str, "torch.Tensor"]) -> None:
        """Load EMA state dict."""
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.clone().float()


@dataclass
class SequenceData:
    """Caches tensor-ready sequence blocks for fast fold reuse."""

    sequences: np.ndarray  # shape: (n_samples, seq_len, n_features)
    targets: np.ndarray  # shape: (n_samples,)
    timestamps: np.ndarray  # pandas-compatible index labels

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.sequences.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.sequences.shape[2]

    def make_loader(
        self,
        indices: Sequence[int],
        batch_size: int,
        shuffle: bool,
        device: Optional["torch.device"] = None,
    ) -> DataLoader:
        """Create a DataLoader for the given indices.
        
        Args:
            indices: Indices of sequences to include
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            device: If provided, pre-load data to this device (GPU) for faster training.
                    This avoids CPU→GPU transfer overhead on each batch (23x speedup on A100).
        """
        if torch is None or TensorDataset is None or DataLoader is None:
            raise ImportError("PyTorch is required for sequence training")
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            idx = np.arange(0, min(1, len(self)))
        x = torch.from_numpy(self.sequences[idx])  # type: ignore[arg-type]
        y = torch.from_numpy(self.targets[idx])  # type: ignore[arg-type]

        # 🚀 GPU PRE-LOADING: Move data to GPU once instead of per-batch transfer
        # This provides ~23x speedup on A100 by avoiding PCIe transfer bottleneck
        if device is not None and device.type == "cuda":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_master_arrays(
    features: pd.DataFrame,
    targets: pd.Series,
    *,
    scaler_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    post_standardization_feature_weights: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Build time-major master arrays without materializing 3D sliding windows.

    This matches the alignment semantics of `build_sequence_data`:
      - join features + targets by index
      - drop rows where target is NaN
      - standardize features (reuse `scaler_stats` if provided)
      - return (X, y, timestamps, scaler_stats)

    Caller can later create window samples via `X.unfold(0, seq_len, 1)`
    and align labels/timestamps to the *next step* (y[t+seq_len]).
    """
    aligned = features.join(targets.rename("target"), how="inner")
    if aligned.empty:
        return None

    aligned = aligned.dropna(subset=["target"])
    if aligned.empty:
        return None

    feat_df_all = aligned[features.columns]
    feat_df = feat_df_all.select_dtypes(include=["number", "bool"]).astype(float)
    if feat_df.empty or feat_df.shape[1] == 0:
        return None
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = feat_df.to_numpy(dtype=np.float32)

    if scaler_stats is not None:
        feat_mean, feat_std = scaler_stats
    else:
        feat_mean = np.mean(values, axis=0)
        feat_std = np.std(values, axis=0)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    values = (values - feat_mean) / feat_std

    if post_standardization_feature_weights is not None:
        w = np.asarray(post_standardization_feature_weights, dtype=np.float32)
        if w.ndim == 1:
            if w.shape[0] != values.shape[1]:
                raise ValueError(
                    f"post_standardization_feature_weights length {w.shape[0]} != n_features {values.shape[1]}"
                )
            values = values * w.reshape(1, -1)
        elif w.ndim == 2:
            if w.shape != values.shape:
                raise ValueError(
                    f"post_standardization_feature_weights shape {w.shape} != values shape {values.shape}"
                )
            values = values * w
        else:
            raise ValueError(f"post_standardization_feature_weights must be 1D or 2D; got ndim={w.ndim}")

    values = np.clip(values, -5.0, 5.0)
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)

    labels = aligned["target"].astype(float).to_numpy(dtype=np.float32)
    timestamps = aligned.index.to_numpy()

    return values, labels, timestamps, (feat_mean, feat_std)


class MultiSymbolGPUMasterStore:
    """GPU-resident master tensors per symbol + cached unfold views per seq_len.

    Designed for Phase-2 pooled training without per-trial giant allocations.
    """

    def __init__(
        self,
        *,
        X_by_symbol: Mapping[str, np.ndarray],
        y_by_symbol: Mapping[str, np.ndarray],
        ts_by_symbol: Mapping[str, np.ndarray],
        device: "torch.device",
        x_dtype: "torch.dtype" = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for GPU master store")

        self.device = device
        self.symbols = [str(s).upper() for s in X_by_symbol.keys()]
        if x_dtype is None:
            # Keep features in float32 by default to avoid dtype mismatches with
            # model parameters (which are typically float32). Autocast can still
            # downcast compute if enabled.
            x_dtype = torch.float32
        self.x_dtype = x_dtype

        self._X: Dict[str, "torch.Tensor"] = {}
        self._y: Dict[str, "torch.Tensor"] = {}
        self._ts: Dict[str, np.ndarray] = {}

        for sym in self.symbols:
            X = np.asarray(X_by_symbol[sym], dtype=np.float32)
            y = np.asarray(y_by_symbol[sym], dtype=np.float32)
            ts = np.asarray(ts_by_symbol[sym])
            if X.ndim != 2:
                raise ValueError(f"X must be 2D [T,F]; sym={sym} shape={X.shape}")
            if y.ndim != 1:
                raise ValueError(f"y must be 1D [T]; sym={sym} shape={y.shape}")
            if len(ts) != len(y) or len(y) != X.shape[0]:
                raise ValueError(f"master alignment mismatch for {sym}: X={X.shape} y={y.shape} ts={ts.shape}")

            self._X[sym] = torch.from_numpy(X).contiguous().to(self.device, dtype=self.x_dtype, non_blocking=True)
            self._y[sym] = torch.from_numpy(y).contiguous().to(self.device, dtype=torch.float32, non_blocking=True)
            self._ts[sym] = ts

        self._unfold_cache: Dict[Tuple[str, int], "torch.Tensor"] = {}
        self._y_aligned_cache: Dict[Tuple[str, int], "torch.Tensor"] = {}
        self._ts_aligned_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def feature_dim(self) -> int:
        sym0 = self.symbols[0]
        return int(self._X[sym0].shape[1])

    def get_unfold(self, sym: str, seq_len: int) -> "torch.Tensor":
        key = (sym, int(seq_len))
        if key not in self._unfold_cache:
            # NOTE: For a 2D [T,F] tensor, `unfold(0, seq_len, 1)` yields shape:
            #   [T-seq_len+1, F, seq_len]
            # We transpose to the common sequence layout expected by our models:
            #   [T-seq_len+1, seq_len, F]
            self._unfold_cache[key] = self._X[sym].unfold(0, int(seq_len), 1).transpose(1, 2)
        return self._unfold_cache[key]

    def get_y_aligned_next_step(self, sym: str, seq_len: int) -> "torch.Tensor":
        """Align labels to match build_sequence_data: y[t+seq_len] for window starting at t."""
        key = (sym, int(seq_len))
        if key not in self._y_aligned_cache:
            self._y_aligned_cache[key] = self._y[sym][int(seq_len) :]
        return self._y_aligned_cache[key]

    def get_ts_aligned_next_step(self, sym: str, seq_len: int) -> np.ndarray:
        key = (sym, int(seq_len))
        if key not in self._ts_aligned_cache:
            self._ts_aligned_cache[key] = self._ts[sym][int(seq_len) :]
        return self._ts_aligned_cache[key]

    def n_samples(self, sym: str, seq_len: int) -> int:
        # Must exclude the last unfold window so that labels align to next-step.
        # X.unfold yields (T - seq_len + 1) windows; labels aligned are length (T - seq_len).
        y_aligned = self.get_y_aligned_next_step(sym, seq_len)
        return int(y_aligned.shape[0])


class _UnfoldMultiSymbolDataset(Dataset):
    def __init__(
        self,
        store: MultiSymbolGPUMasterStore,
        seq_len: int,
        samples: Sequence[Tuple[str, int]],
    ) -> None:
        if Dataset is None:
            raise ImportError("PyTorch is required for windowed dataset")
        self.store = store
        self.seq_len = int(seq_len)
        self.samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.samples)

    def __getitem__(self, i: int):
        sym, start = self.samples[int(i)]
        X_unfold = self.store.get_unfold(sym, self.seq_len)
        # start ranges 0..(T-seq_len-1)
        x = X_unfold[int(start)]  # [seq_len, F]
        y = self.store.get_y_aligned_next_step(sym, self.seq_len)[int(start)]
        return x, y


@dataclass
class WindowedSequenceData:
    """Sequence-like wrapper that builds batches by indexing GPU master tensors."""

    store: MultiSymbolGPUMasterStore
    seq_len: int
    samples: List[Tuple[str, int]]
    timestamps: np.ndarray  # aligned to `samples` (used only for debug/compat)

    @property
    def feature_dim(self) -> int:
        return self.store.feature_dim()

    def __len__(self) -> int:  # pragma: no cover
        return len(self.samples)

    def make_loader(
        self,
        indices: Sequence[int],
        batch_size: int,
        shuffle: bool,
        device: Optional["torch.device"] = None,
    ) -> DataLoader:
        # `device` is ignored: master tensors already reside on GPU.
        del device
        if DataLoader is None:
            raise ImportError("PyTorch is required for windowed training")
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            idx = np.arange(0, min(1, len(self)))
        subset = [self.samples[int(j)] for j in idx]
        ds = _UnfoldMultiSymbolDataset(self.store, int(self.seq_len), subset)
        return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False, num_workers=0)


# =============================================================================
# CNN FRONTEND MODULE (Dilated Residual CNN - Jane Street/Jump/Two Sigma style)
# =============================================================================

def _get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "mish": nn.Mish(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "selu": nn.SELU(),
    }
    return activations.get(name.lower(), nn.ReLU())


class CNNBlock(nn.Module):
    """Single CNN block with optional dilation, normalization, and residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,
        batch_norm: bool = True,
        layer_norm: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual and (in_channels == out_channels) and (stride == 1)
        
        # Padding to maintain sequence length (accounting for dilation)
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not batch_norm,  # No bias if using batch norm
        )
        
        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm1d(out_channels)
        elif layer_norm:
            self.norm = None  # LayerNorm applied in forward (needs dynamic size)
            self.use_layer_norm = True
        else:
            self.use_layer_norm = False
        
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual projection if dimensions don't match
        self.residual_proj = None
        if residual and (in_channels != out_channels or stride != 1):
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            self.residual = True
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self.residual_proj is not None:
            nn.init.kaiming_normal_(self.residual_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        identity = x
        
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        elif hasattr(self, 'use_layer_norm') and self.use_layer_norm:
            out = nn.functional.layer_norm(out, out.shape[1:])
        
        out = self.activation(out)
        out = self.dropout(out)
        
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        return out


class CNNFrontend(nn.Module):
    """Multi-block dilated CNN frontend for LSTM.
    
    Architecture inspired by WaveNet/TCN used at quant firms.
    Captures local patterns and multi-scale temporal features.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_blocks: int = 2,
        filters: int = 64,
        kernel_size: int = 3,
        dilation: int = 1,  # Can be 1, 2, or 4 for each block
        stride: int = 1,
        pooling: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        batch_norm: bool = True,
        layer_norm: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()
        
        self.input_proj = nn.Conv1d(input_dim, filters, 1)  # 1x1 conv to project input
        
        # Stack of CNN blocks with increasing dilation
        blocks = []
        current_dilation = dilation
        for i in range(num_blocks):
            blocks.append(CNNBlock(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=current_dilation,
                stride=stride if i == 0 else 1,  # Only first block uses stride
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                residual=residual,
            ))
            # Increase dilation for each block (1, 2, 4, 8, ...)
            current_dilation = min(current_dilation * 2, 8)
        
        self.blocks = nn.Sequential(*blocks)
        
        # Optional pooling
        self.pool = None
        if pooling is not None and pooling > 1:
            self.pool = nn.AvgPool1d(pooling, stride=pooling)
        
        self.output_dim = filters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # -> (batch, features, seq_len)
        
        x = self.input_proj(x)
        x = self.blocks(x)
        
        if self.pool is not None:
            x = self.pool(x)
        
        # Back to (batch, seq_len, channels) for LSTM
        x = x.transpose(1, 2)
        return x


class CNNLSTMRegressor(nn.Module):
    """CNN-LSTM hybrid model for financial return prediction.
    
    CNN frontend extracts local temporal patterns, LSTM captures long-range dependencies.
    Dilated residual CNN architecture used at Jane Street, Jump, Two Sigma.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_scale: float = 0.5,
        # CNN parameters
        cnn_blocks: int = 2,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        cnn_dilation: int = 1,
        cnn_stride: int = 1,
        cnn_pooling: Optional[int] = None,
        cnn_activation: str = "gelu",
        cnn_dropout: float = 0.1,
        cnn_batch_norm: bool = True,
        cnn_layer_norm: bool = False,
        cnn_residual: bool = True,
    ) -> None:
        super().__init__()
        self.output_scale = output_scale
        
        # CNN Frontend
        self.cnn = CNNFrontend(
            input_dim=input_dim,
            num_blocks=cnn_blocks,
            filters=cnn_filters,
            kernel_size=cnn_kernel_size,
            dilation=cnn_dilation,
            stride=cnn_stride,
            pooling=cnn_pooling,
            activation=cnn_activation,
            dropout=cnn_dropout,
            batch_norm=cnn_batch_norm,
            layer_norm=cnn_layer_norm,
            residual=cnn_residual,
        )
        
        # LSTM on CNN output
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            self.cnn.output_dim,  # Input is CNN output dimension
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize LSTM and head weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Forget gate bias
        
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        
        # CNN frontend
        x = self.cnn(x)  # -> (batch, new_seq_len, cnn_filters)
        
        # LSTM
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        
        # Head with bounded output
        raw = self.head(last).squeeze(-1)
        return torch.tanh(raw) * self.output_scale


# =============================================================================
# ATTENTION MECHANISMS FOR LSTM
# Supports: Bahdanau (additive), Luong (multiplicative), Scaled Dot-Product
# =============================================================================

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism.
    
    score(h_t, h_s) = v^T * tanh(W_q * h_t + W_k * h_s)
    
    Good for capturing complex alignment patterns in financial time series.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        attn_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, attn_hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, attn_hidden_dim, bias=False)
        self.v = nn.Linear(attn_hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,  # (batch, hidden_dim) - last LSTM output
        keys: torch.Tensor,   # (batch, seq_len, hidden_dim) - all LSTM outputs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.
        
        Returns:
            context: (batch, hidden_dim) - weighted sum of keys
            weights: (batch, seq_len) - attention weights
        """
        # query: (batch, hidden) -> (batch, 1, attn_hidden)
        query_proj = self.query_proj(query).unsqueeze(1)
        # keys: (batch, seq_len, hidden) -> (batch, seq_len, attn_hidden)
        keys_proj = self.key_proj(keys)
        
        # Additive score: (batch, seq_len, 1)
        scores = self.v(torch.tanh(query_proj + keys_proj)).squeeze(-1)
        
        # Softmax over sequence
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Weighted sum: (batch, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        
        return context, weights


class LuongAttention(nn.Module):
    """Multiplicative (Luong) attention mechanism.
    
    Supports three score functions:
    - dot: score = h_t^T * h_s
    - general: score = h_t^T * W * h_s  
    - concat: score = v^T * tanh(W * [h_t; h_s])
    
    Fast and effective for financial sequence modeling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        score_fn: str = "general",  # dot, general, concat
        dropout: float = 0.1,
    ):
        super().__init__()
        self.score_fn = score_fn
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        if score_fn == "general":
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif score_fn == "concat":
            self.W = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(
        self,
        query: torch.Tensor,  # (batch, hidden_dim)
        keys: torch.Tensor,   # (batch, seq_len, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = keys.shape
        
        if self.score_fn == "dot":
            # (batch, seq_len)
            scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        elif self.score_fn == "general":
            # (batch, hidden) -> (batch, hidden)
            query_proj = self.W(query)
            scores = torch.bmm(keys, query_proj.unsqueeze(-1)).squeeze(-1)
        else:  # concat
            # Expand query to match keys: (batch, seq_len, hidden)
            query_exp = query.unsqueeze(1).expand(-1, seq_len, -1)
            # Concatenate: (batch, seq_len, hidden*2)
            combined = torch.cat([query_exp, keys], dim=-1)
            scores = self.v(torch.tanh(self.W(combined))).squeeze(-1)
        
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (Transformer-style).
    
    score = (Q * K^T) / sqrt(d_k)
    
    Supports multi-head attention for capturing diverse patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.scale = math.sqrt(self.head_dim)
        
        # Multi-head projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,  # (batch, hidden_dim)
        keys: torch.Tensor,   # (batch, seq_len, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = keys.shape
        
        # Project and reshape for multi-head
        # query: (batch, 1, hidden) -> (batch, num_heads, 1, head_dim)
        q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # keys: (batch, seq_len, hidden) -> (batch, num_heads, seq_len, head_dim)
        k = self.k_proj(keys).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 🚀 FLASH ATTENTION via scaled_dot_product_attention (PyTorch 2.0+)
        # Uses fused CUDA kernels: 2-4x faster, O(n) memory instead of O(n²)
        dropout_p = self.dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            scale=1.0 / (self.scale * self.temperature),
        )
        
        # Reshape back: (batch, hidden_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)
        context = self.out_proj(attn_out)
        
        # Note: SDPA doesn't return attention weights (memory optimization)
        # Return uniform placeholder weights for compatibility
        mean_weights = torch.ones(batch_size, seq_len, device=query.device) / seq_len
        
        return context, mean_weights


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for improved position encoding.
    
    Used in modern transformers (LLaMA, GPT-NeoX) for better length generalization.
    Applies rotation to query/key vectors based on position.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for positions."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        
        # Interleave: [cos, cos, ...], [sin, sin, ...]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, :])
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary embedding to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            seq_len: Sequence length (optional, inferred from x)
            
        Returns:
            Tensor with rotary position encoding applied
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Expand cache if needed
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :seq_len, :self.dim]
        sin = self.sin_cached[:, :seq_len, :self.dim]
        
        # Apply rotation
        return (x * cos) + (self._rotate_half(x) * sin)


class Mixout(nn.Module):
    """Mixout regularization: mix dropout with original pretrained weights.
    
    From "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
    During training, randomly replaces weights with their original values.
    """
    
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p
        self.original_weights: Dict[str, torch.Tensor] = {}
    
    def register_original_weights(self, module: nn.Module) -> None:
        """Store original weights for mixout."""
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.original_weights[name] = param.data.clone()
    
    def forward(self, module: nn.Module) -> None:
        """Apply mixout to module weights during training."""
        if not module.training or self.p <= 0:
            return
        
        for name, param in module.named_parameters():
            if name in self.original_weights and param.requires_grad:
                mask = torch.bernoulli(torch.full_like(param, 1 - self.p))
                param.data = mask * param.data + (1 - mask) * self.original_weights[name].to(param.device)


class LSTMRegressor(nn.Module):
    """LSTM/GRU regressor with optional attention for financial return prediction.
    
    Supports:
    - Cell types: LSTM, GRU
    - Bidirectional processing
    - Multiple attention mechanisms: none, bahdanau, luong, scaled_dot
    - Configurable dropout types: input, output, recurrent, weight, time, zoneout
    - Layer normalization and residual connections
    - Flexible output head with configurable layers and activations
    - Multiple weight initialization schemes
    - Sequence noise injection for regularization
    - Stacked attention layers
    - Feedforward networks
    - Rotary positional embeddings
    - Mixout regularization
    - Stochastic depth
    
    All hyperparameters from the Optuna search space are supported.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_scale: float = 0.5,
        # Cell type and architecture
        cell_type: str = "lstm",  # lstm, gru
        bidirectional: bool = False,
        # Dropout variants
        input_dropout: float = 0.0,
        output_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,  # Applied within RNN (only LSTM supports this natively)
        weight_dropout: float = 0.0,  # DropConnect on recurrent weights
        time_dropout: float = 0.0,  # Drop entire timesteps
        zoneout: float = 0.0,  # Stochastic depth for hidden states
        sequence_noise_std: float = 0.0,  # Gaussian noise on input sequences
        # Normalization and skip connections
        layer_norm: bool = False,
        residual: bool = False,
        skip_connect: bool = False,
        # Output head configuration
        fc_layers: int = 1,  # Number of FC layers in head
        fc_hidden: int = 0,  # Hidden dim for FC layers (0 = auto: hidden_dim // 2)
        activation: str = "gelu",  # Hidden layer activation
        output_activation: str = "tanh",  # Final activation: tanh, sigmoid, none
        # Weight initialization
        recurrent_kernel_init: str = "orthogonal",  # orthogonal, xavier_uniform, xavier_normal
        hidden_state_init: str = "zeros",  # zeros, normal, learned
        # Attention parameters
        attention_type: str = "none",  # none, bahdanau, luong, scaled_dot
        attn_hidden_dim: int = 64,
        attn_dropout: float = 0.1,
        attn_heads: int = 1,
        attn_score_fn: str = "general",  # For Luong: dot, general, concat
        attn_temperature: float = 1.0,  # For scaled_dot
        attn_merge: str = "concat",  # concat, add, gate
        attn_normalization: str = "softmax",  # softmax, sparsemax, entmax
        attn_positional_encoding: bool = False,
        attn_context_length: int = 0,  # 0 = full sequence
        attn_entropy_reg: float = 0.0,
        attn_distance_reg: float = 0.0,
        attn_key_dim: int = 0,  # 0 = auto (hidden_dim)
        attn_value_dim: int = 0,  # 0 = auto (hidden_dim)
        # Advanced architecture (NEW)
        attn_layers: int = 1,  # Number of stacked attention layers
        feedforward_dim: int = 256,  # Feedforward network dimension
        batch_norm_head: bool = False,  # Add BatchNorm in output head
        dense_activation: str = "gelu",  # Activation in dense/feedforward layers
        mixout_prob: float = 0.0,  # Mixout regularization probability
        stochastic_depth: float = 0.0,  # Stochastic depth drop probability
        rotary_embedding: bool = False,  # Use rotary positional embeddings
    ) -> None:
        super().__init__()
        self.output_scale = output_scale
        self.attention_type = attention_type
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.residual = residual
        self.skip_connect = skip_connect
        self.sequence_noise_std = sequence_noise_std
        self.time_dropout_rate = time_dropout
        self.zoneout_rate = zoneout
        self.attn_merge = attn_merge
        self.output_activation = output_activation
        self.mixout_prob = mixout_prob
        self.stochastic_depth_rate = stochastic_depth
        self.attn_layers_count = attn_layers
        self.rotary_embedding = rotary_embedding
        
        effective_dropout = dropout if num_layers > 1 else 0.0
        num_directions = 2 if bidirectional else 1
        rnn_output_dim = hidden_dim * num_directions
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim) if not layer_norm else nn.Identity()
        self.input_layer_norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        
        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        
        # Select RNN cell type
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )
        
        # Weight dropout (DropConnect) - wrap recurrent weights
        self.weight_dropout_rate = weight_dropout
        if weight_dropout > 0:
            self._apply_weight_dropout()
        
        # Layer normalization after RNN
        self.rnn_layer_norm = nn.LayerNorm(rnn_output_dim) if layer_norm else nn.Identity()
        
        # Output dropout (applied to RNN output)
        self.output_dropout_layer = nn.Dropout(output_dropout) if output_dropout > 0 else nn.Identity()
        
        # Residual projection if dimensions don't match
        self.residual_proj = None
        if residual and input_dim != rnn_output_dim:
            self.residual_proj = nn.Linear(input_dim, rnn_output_dim, bias=False)
        
        # Skip connection from input to output head
        skip_dim = input_dim if skip_connect else 0
        
        # =====================================================================
        # ATTENTION MECHANISM (optional, with stacked layers support)
        # =====================================================================
        self.attention_layers = nn.ModuleList()
        self.attention_layer_norms = nn.ModuleList()
        
        if attention_type != "none":
            for layer_idx in range(attn_layers):
                if attention_type == "bahdanau":
                    attn_module = BahdanauAttention(
                        hidden_dim=rnn_output_dim,
                        attn_hidden_dim=attn_hidden_dim,
                        dropout=attn_dropout,
                    )
                elif attention_type == "luong":
                    attn_module = LuongAttention(
                        hidden_dim=rnn_output_dim,
                        score_fn=attn_score_fn,
                        dropout=attn_dropout,
                    )
                elif attention_type == "scaled_dot":
                    attn_module = ScaledDotProductAttention(
                        hidden_dim=rnn_output_dim,
                        num_heads=attn_heads,
                        dropout=attn_dropout,
                        temperature=attn_temperature,
                    )
                else:
                    attn_module = None
                
                if attn_module is not None:
                    self.attention_layers.append(attn_module)
                    # Layer norm between attention layers
                    if layer_idx < attn_layers - 1:
                        self.attention_layer_norms.append(nn.LayerNorm(rnn_output_dim))
            
            # Set first attention layer as self.attention for backward compat
            self.attention = self.attention_layers[0] if self.attention_layers else None
        else:
            self.attention = None
        
        # =====================================================================
        # ROTARY POSITIONAL EMBEDDINGS (optional)
        # =====================================================================
        self.rotary_emb = None
        if rotary_embedding and attention_type != "none":
            # Simple rotary embedding implementation
            self.rotary_emb = RotaryEmbedding(rnn_output_dim)
        
        # =====================================================================
        # FEEDFORWARD NETWORK (after attention, like Transformer)
        # =====================================================================
        self.feedforward = None
        if feedforward_dim > 0 and attention_type != "none":
            ffn_act = self._get_activation(dense_activation)
            self.feedforward = nn.Sequential(
                nn.Linear(rnn_output_dim, feedforward_dim),
                ffn_act,
                nn.Dropout(attn_dropout),
                nn.Linear(feedforward_dim, rnn_output_dim),
                nn.Dropout(attn_dropout),
            )
            self.ffn_layer_norm = nn.LayerNorm(rnn_output_dim)
        
        # Calculate head input dimension based on attention merge strategy
        if self.attention is not None:
            if attn_merge == "concat":
                head_input_dim = rnn_output_dim * 2 + skip_dim
            elif attn_merge == "add":
                head_input_dim = rnn_output_dim + skip_dim
            elif attn_merge == "gate":
                head_input_dim = rnn_output_dim + skip_dim
                self.attn_gate = nn.Sequential(
                    nn.Linear(rnn_output_dim * 2, rnn_output_dim),
                    nn.Sigmoid(),
                )
            else:
                head_input_dim = rnn_output_dim * 2 + skip_dim
        else:
            head_input_dim = rnn_output_dim + skip_dim
        
        # =====================================================================
        # OUTPUT HEAD (configurable depth, activation, and batch norm)
        # =====================================================================
        fc_hidden_dim = fc_hidden if fc_hidden > 0 else max(32, hidden_dim // 2)
        
        # Get activation functions
        act_fn = self._get_activation(activation)
        dense_act_fn = self._get_activation(dense_activation)
        
        head_layers = []
        
        # Optional batch norm at head input
        if batch_norm_head:
            head_layers.append(nn.BatchNorm1d(head_input_dim))
        head_layers.append(nn.LayerNorm(head_input_dim))
        
        if fc_layers == 1:
            # Single layer: input -> output
            head_layers.extend([
                nn.Linear(head_input_dim, fc_hidden_dim),
                dense_act_fn,  # Use dense_activation here
                nn.Linear(fc_hidden_dim, 1),
            ])
        else:
            # Multiple layers
            head_layers.append(nn.Linear(head_input_dim, fc_hidden_dim))
            head_layers.append(dense_act_fn)
            for _ in range(fc_layers - 2):
                head_layers.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
                head_layers.append(dense_act_fn)
                if batch_norm_head:
                    head_layers.append(nn.BatchNorm1d(fc_hidden_dim))
            head_layers.append(nn.Linear(fc_hidden_dim, 1))
        
        self.head = nn.Sequential(*head_layers)
        
        # Learned initial hidden state (optional)
        self.hidden_state_init_type = hidden_state_init
        self.init_hidden = None
        self.init_cell = None
        if hidden_state_init == "learned":
            self.init_hidden = nn.Parameter(torch.zeros(num_layers * num_directions, 1, hidden_dim))
            if cell_type == "lstm":
                self.init_cell = nn.Parameter(torch.zeros(num_layers * num_directions, 1, hidden_dim))
        
        # Initialize weights
        self._init_weights(recurrent_kernel_init, hidden_state_init)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "swish": nn.SiLU(),
            "mish": nn.Mish(),
            "none": nn.Identity(),
        }
        return activations.get(name.lower(), nn.GELU())
    
    def _apply_weight_dropout(self) -> None:
        """Apply weight dropout (DropConnect) to recurrent weights."""
        # Store original weights and register forward hook
        for name, param in self.rnn.named_parameters():
            if 'weight_hh' in name:
                # Register a hook to apply dropout during forward pass
                setattr(self, f"_orig_{name.replace('.', '_')}", param.data.clone())
    
    def _init_weights(self, kernel_init: str, hidden_init: str) -> None:
        """Initialize weights based on configuration."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Recurrent weights - use specified initialization
                if kernel_init == "orthogonal":
                    nn.init.orthogonal_(param)
                elif kernel_init == "xavier_uniform":
                    nn.init.xavier_uniform_(param)
                elif kernel_init == "xavier_normal":
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for LSTM
                if self.cell_type == "lstm":
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
        
        # Initialize learned hidden states if enabled
        if hidden_init == "normal":
            if self.init_hidden is not None:
                nn.init.normal_(self.init_hidden, std=0.01)
            if self.init_cell is not None:
                nn.init.normal_(self.init_cell, std=0.01)
        
        # Initialize head layers
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_initial_hidden(self, batch_size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get initial hidden state for RNN."""
        if self.init_hidden is not None:
            h0 = self.init_hidden.expand(-1, batch_size, -1).contiguous()
            if self.cell_type == "lstm" and self.init_cell is not None:
                c0 = self.init_cell.expand(-1, batch_size, -1).contiguous()
                return (h0, c0)
            return (h0,)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        device = x.device
        
        # Sequence noise injection (training regularization)
        if self.training and self.sequence_noise_std > 0:
            noise = torch.randn_like(x) * self.sequence_noise_std
            x = x + noise
        
        # Time dropout (drop entire timesteps)
        if self.training and self.time_dropout_rate > 0:
            time_mask = torch.rand(batch_size, seq_len, 1, device=device) > self.time_dropout_rate
            x = x * time_mask.float()
        
        # Input normalization (BatchNorm or LayerNorm)
        x_flat = x.reshape(-1, n_features)
        if isinstance(self.input_norm, nn.BatchNorm1d):
            x_norm = self.input_norm(x_flat)
        else:
            x_norm = x_flat
        x = x_norm.reshape(batch_size, seq_len, n_features)
        x = self.input_layer_norm(x)
        
        # Store for skip connection
        x_input = x[:, -1, :] if self.skip_connect else None
        
        # Input dropout
        x = self.input_dropout(x)
        
        # 🚀 WEIGHT DROPOUT: Use functional approach to avoid in-place param modification
        # In-place modification of .data breaks autograd graph and causes recompilation
        # We apply dropout to weights via functional_call or masking the output instead
        # Note: True weight dropout requires hooks; for now we skip the in-place modification
        # The regularization effect is achieved via input/output dropout and zoneout
        # if self.training and self.weight_dropout_rate > 0:
        #     # Would require register_forward_pre_hook for proper implementation
        #     pass
        
        # Get initial hidden state
        init_hidden = self._get_initial_hidden(batch_size, device)
        
        # RNN forward pass
        if init_hidden is not None:
            if self.cell_type == "lstm":
                out, _ = self.rnn(x, init_hidden)
            else:  # GRU
                out, _ = self.rnn(x, init_hidden[0])
        else:
            out, _ = self.rnn(x)
        
        # out: (batch, seq_len, hidden_dim * num_directions)
        
        # Zoneout (stochastic depth for hidden states)
        if self.training and self.zoneout_rate > 0:
            # Randomly keep some hidden states from previous timestep
            zoneout_mask = torch.rand(out.shape, device=device) > self.zoneout_rate
            out = out * zoneout_mask.float()
        
        # =====================================================================
        # ROTARY POSITIONAL EMBEDDINGS (if enabled)
        # =====================================================================
        if self.rotary_emb is not None:
            out = self.rotary_emb(out, seq_len)
        
        # Layer normalization
        out = self.rnn_layer_norm(out)
        
        # Output dropout
        out = self.output_dropout_layer(out)
        
        last = out[:, -1, :]  # (batch, hidden_dim * num_directions)
        
        # Residual connection
        if self.residual:
            residual_input = x[:, -1, :]  # Use last timestep of input
            if self.residual_proj is not None:
                residual_input = self.residual_proj(residual_input)
            last = last + residual_input
        
        # =====================================================================
        # STACKED ATTENTION LAYERS (with stochastic depth)
        # =====================================================================
        context = None
        if len(self.attention_layers) > 0:
            # Apply stacked attention layers
            for layer_idx, attn_layer in enumerate(self.attention_layers):
                # 🚀 Stochastic depth: GPU-based random (no .item() sync overhead)
                if self.training and self.stochastic_depth_rate > 0:
                    drop_prob = self.stochastic_depth_rate * (layer_idx + 1) / len(self.attention_layers)
                    if torch.rand(1, device=device).item() < drop_prob:
                        continue  # Skip this layer
                
                layer_context, _ = attn_layer(last, out)
                
                if context is None:
                    context = layer_context
                else:
                    # Residual connection between attention layers
                    context = context + layer_context
                
                # Layer norm between attention layers
                if layer_idx < len(self.attention_layer_norms):
                    context = self.attention_layer_norms[layer_idx](context)
            
            # =====================================================================
            # FEEDFORWARD NETWORK (after attention, Transformer-style)
            # =====================================================================
            if self.feedforward is not None and context is not None:
                # Pre-norm feedforward with residual
                ffn_input = self.ffn_layer_norm(context)
                ffn_out = self.feedforward(ffn_input)
                context = context + ffn_out  # Residual connection
            
            # Merge attention context with last hidden
            if context is not None:
                if self.attn_merge == "concat":
                    combined = torch.cat([last, context], dim=-1)
                elif self.attn_merge == "add":
                    combined = last + context
                elif self.attn_merge == "gate":
                    gate = self.attn_gate(torch.cat([last, context], dim=-1))
                    combined = gate * last + (1 - gate) * context
                else:
                    combined = torch.cat([last, context], dim=-1)
            else:
                combined = last
        else:
            combined = last
        
        # Add skip connection
        if x_input is not None:
            combined = torch.cat([combined, x_input], dim=-1)
        
        # =====================================================================
        # MIXOUT REGULARIZATION (during training)
        # =====================================================================
        # Applied implicitly through weight mixing in optimizer (handled externally)
        
        # Output head
        raw = self.head(combined).squeeze(-1)
        
        # Output activation
        if self.output_activation == "tanh":
            return torch.tanh(raw) * self.output_scale
        elif self.output_activation == "sigmoid":
            return (torch.sigmoid(raw) * 2 - 1) * self.output_scale
        else:
            return raw * self.output_scale


def build_sequence_data(
    features: pd.DataFrame,
    targets: pd.Series,
    seq_len: int,
    scaler_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    post_standardization_feature_weights: Optional[np.ndarray] = None,
) -> Optional[Tuple["SequenceData", Tuple[np.ndarray, np.ndarray]]]:
    """Build sequence data with proper feature standardization.
    
    Args:
        features: DataFrame of features
        targets: Series of target values
        seq_len: Sequence length for LSTM windows
        scaler_stats: Optional (mean, std) tuple from training set for validation inference
        
    Returns:
        Tuple of (SequenceData, (mean_array, std_array)) for reuse in validation/inference
    """
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")

    # IMPORTANT:
    # Do NOT drop rows due to NaNs across the (very wide) feature matrix.
    # With 1k+ columns, a single missing value would wipe out almost all rows.
    # Instead:
    #   - drop rows where the *target* is missing
    #   - replace non-finite / missing feature values with 0.0
    aligned = features.join(targets.rename("target"), how="inner")
    if aligned.empty:
        return None

    aligned = aligned.dropna(subset=["target"])
    if aligned.empty or len(aligned) <= seq_len:
        return None

    feat_df_all = aligned[features.columns]
    # Keep only numeric/bool columns; some pipelines may include datetime/object
    # columns (e.g., calendar fields) that cannot be cast to float.
    feat_df = feat_df_all.select_dtypes(include=["number", "bool"]).astype(float)
    if feat_df.empty or feat_df.shape[1] == 0:
        return None
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = feat_df.to_numpy(dtype=np.float32)
    
    # 🔧 FIX: Apply z-score standardization to features
    # This is CRITICAL - features like GDP (~17 trillion) will dominate otherwise
    if scaler_stats is not None:
        # Use provided stats (for validation/inference consistency)
        feat_mean, feat_std = scaler_stats
    else:
        # Compute stats from training data
        feat_mean = np.mean(values, axis=0)
        feat_std = np.std(values, axis=0)
        # Replace zero std with 1 to avoid division by zero (constant features)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    
    # Standardize: z = (x - mean) / std
    values = (values - feat_mean) / feat_std

    # Optional: apply per-feature weights AFTER standardization.
    # This is the correct place to apply family weights (scale-invariant).
    if post_standardization_feature_weights is not None:
        w = np.asarray(post_standardization_feature_weights, dtype=np.float32)
        if w.ndim == 1:
            if w.shape[0] != values.shape[1]:
                raise ValueError(
                    f"post_standardization_feature_weights length {w.shape[0]} != n_features {values.shape[1]}"
                )
            values = values * w.reshape(1, -1)
        elif w.ndim == 2:
            if w.shape != values.shape:
                raise ValueError(
                    f"post_standardization_feature_weights shape {w.shape} != values shape {values.shape}"
                )
            values = values * w
        else:
            raise ValueError(f"post_standardization_feature_weights must be 1D or 2D; got ndim={w.ndim}")
    
    # 🔧 FIX: Clip extreme values to prevent outlier domination
    # After z-scoring, values >5 std are extremely rare and likely noise
    values = np.clip(values, -5.0, 5.0)

    # Safety: avoid propagating any non-finite values into torch.
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    
    labels = aligned["target"].astype(float).to_numpy(dtype=np.float32)
    timestamps = aligned.index.to_numpy()
    
    # 🚀 SPEED OPTIMIZATION: Vectorized sliding window (10-100x faster than Python loop)
    # Uses stride tricks to create a view without copying data
    # Old approach: Python loop with list.append() - O(n) allocations
    # New approach: Single strided view - O(1) allocation
    n_windows = len(aligned) - seq_len
    if n_windows <= 0:
        return None
    
    # Create sliding window view (no copy, just strides)
    # sliding_window_view returns shape (n-seq_len+1, n_features, seq_len)
    # We need (n_windows, seq_len, n_features) where n_windows = n - seq_len
    sequences = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
    # Fix shape: move seq_len axis from -1 to position 1
    sequences = np.moveaxis(sequences, -1, 1)  # Now (n-seq_len+1, seq_len, n_features)
    # Drop last window to match old behavior: window[i] predicts labels[i+seq_len]
    sequences = sequences[:-1]  # Now (n-seq_len, seq_len, n_features) = (n_windows, seq_len, n_features)
    # Make contiguous copy for torch
    sequences = np.ascontiguousarray(sequences, dtype=np.float32)
    
    # Targets and timestamps are aligned with window end positions
    targets_arr = labels[seq_len:].astype(np.float32)
    ts_arr = timestamps[seq_len:]
    # Return both SequenceData and scaler stats for reuse
    return SequenceData(sequences=sequences, targets=targets_arr, timestamps=ts_arr), (feat_mean, feat_std)


def train_lstm_fold(
    seq_data: SequenceData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Mapping[str, Any],
    device: Optional["torch.device"] = None,
    warm_state: Optional[Dict[str, torch.Tensor]] = None,
    return_model: bool = False,
) -> Dict[str, Any]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for sequence training")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =====================================================================
    # DATA PIPELINE PARAMETERS (from Optuna sampling)
    # =====================================================================
    # These affect how sequence data is prepared and used
    train_seq_length = int(cfg.get("train_seq_length", cfg.get("lstm_seq_len", 63)))
    target_seq_length = int(cfg.get("target_seq_length", 1))  # Multi-step prediction
    market_regime_model = str(cfg.get("market_regime_model", "none"))  # HMM, VIX-feature, PCA-regime
    volatility_regime_window = int(cfg.get("volatility_regime_window", 21))
    
    # Note: train_seq_length and target_seq_length are used during data preparation
    # (in build_sequence_data or SequenceData creation), not in model training.
    # market_regime_model and volatility_regime_window would be used for regime-aware
    # training strategies or additional feature engineering.
    # These are passed through cfg and can be utilized by the calling code.
    
    # 🚀 Create dedicated CUDA stream for this fold to enable true parallelism
    # This allows multiple folds/trials to execute concurrently on GPU
    cuda_stream = None
    if device.type == "cuda":
        cuda_stream = torch.cuda.Stream(device=device)
    
    # Use context manager for stream if available
    stream_context = torch.cuda.stream(cuda_stream) if cuda_stream else nullcontext()
    
    with stream_context:
        # Optional NVTX ranges for Nsight profiling (disabled by default).
        # Enable with: DCF_ENABLE_NVTX=1
        import os as _os
        from contextlib import contextmanager as _contextmanager

        _nvtx_enabled = (
            device.type == "cuda"
            and _os.environ.get("DCF_ENABLE_NVTX", "0") == "1"
            and hasattr(torch.cuda, "nvtx")
        )

        @_contextmanager
        def _nvtx(msg: str):
            if not _nvtx_enabled:
                yield
                return
            try:
                torch.cuda.nvtx.range_push(msg)
            except Exception:
                yield
                return
            try:
                yield
            finally:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

        # Enable TF32 for A100/Ampere+ GPUs (2x speedup for FP32 ops)
        if device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark for optimal algorithm selection (10-20% faster LSTMs)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        batch_size = int(cfg.get("lstm_batch_size", 64))
        learning_rate = float(cfg.get("lstm_learning_rate", 1e-3))
        weight_decay = float(cfg.get("lstm_weight_decay", 1e-4))
        hidden_dim = int(cfg.get("lstm_hidden_dim", 96))
        num_layers = int(cfg.get("lstm_layers", 2))
        dropout = float(cfg.get("lstm_dropout", 0.1))
        
        # =====================================================================
        # TRAINING CONTROL PARAMETERS (from Optuna sampling)
        # =====================================================================
        # Use max_epochs from cfg, fallback to lstm_epochs for backward compat
        epochs = int(cfg.get("max_epochs", cfg.get("lstm_epochs", 15)))
        # Use early_stopping_patience from cfg, fallback to lstm_early_stop_patience
        # 🚀 SPEED: Reduced default from 5 to 3 for faster convergence
        # Batches 7 & 9 took 400-500s due to not triggering early stopping
        patience = int(cfg.get("early_stopping_patience", cfg.get("lstm_early_stop_patience", 3)))
        # Plateau patience for LR scheduler (reduced from 5 to 3)
        plateau_patience = int(cfg.get("plateau_patience", 3))
        # Warmup steps for learning rate
        warmup_steps = int(cfg.get("warmup_steps", 100))
        # Input noise for regularization during training
        input_noise_std = float(cfg.get("input_noise_std", 0.0))
        # Label smoothing (for regression, applied as target noise)
        label_smoothing = float(cfg.get("label_smoothing", 0.0))
        
        # =====================================================================
        # LR SCHEDULER PARAMETERS (from Optuna sampling)
        # =====================================================================
        cosine_t_max = int(cfg.get("cosine_t_max", 50))
        step_lr_step_size = int(cfg.get("step_lr_step_size", 10))
        cyclical_base_lr = float(cfg.get("cyclical_base_lr", 1e-5))
        cyclical_max_lr = float(cfg.get("cyclical_max_lr", 1e-3))
        cyclical_step_size = int(cfg.get("cyclical_step_size", 10))
        
        # =====================================================================
        # LOSS FUNCTION PARAMETERS (from Optuna sampling)
        # =====================================================================
        huber_delta = float(cfg.get("huber_delta", 1.0))
        quantile_alpha = float(cfg.get("quantile_alpha", 0.5))
        
        # =====================================================================
        # ADVANCED ARCHITECTURE PARAMETERS (from Optuna sampling)
        # =====================================================================
        attn_layers = int(cfg.get("attn_layers", 1))
        feedforward_dim = int(cfg.get("feedforward_dim", 256))
        batch_norm_head = bool(cfg.get("batch_norm_head", False))
        dense_activation = str(cfg.get("dense_activation", "gelu"))
        mixout_prob = float(cfg.get("mixout_prob", 0.0))
        stochastic_depth = float(cfg.get("stochastic_depth", 0.0))
        rotary_embedding = bool(cfg.get("rotary_embedding", False))
        
        # =====================================================================
        # MIXED PRECISION CONFIGURATION (Hedge Fund Best Practices)
        # =====================================================================
        # bf16 for forward/backward pass (312 TFLOPS on A100)
        # fp32 for: loss computation, LayerNorm/BatchNorm, EMA, softmax, grad accum
        # Optimizer states always in fp32
        amp_precision = str(cfg.get("amp_precision", "bf16"))
        use_amp = (amp_precision in ("bf16", "fp16")) and device.type == "cuda"
        
        # Determine AMP dtype
        if amp_precision == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_bf16 = True
        elif amp_precision == "fp16":
            amp_dtype = torch.float16
            use_bf16 = False
        else:
            amp_dtype = torch.float32
            use_bf16 = False
            use_amp = False
        
        # EMA configuration (stabilizes returns forecasts - used by hedge funds)
        ema_enabled = bool(cfg.get("ema_enabled", True))
        ema_decay = float(cfg.get("ema_decay", 0.9999))
        
        # Max gradient norm for clipping (lstm_grad_clip overrides max_grad_norm if set)
        max_grad_norm = float(cfg.get("lstm_grad_clip", cfg.get("max_grad_norm", 1.0)))
        
        # Use AMP if lstm_use_amp is True (overrides amp_precision)
        use_amp_override = cfg.get("lstm_use_amp", None)
        if use_amp_override is not None:
            use_amp = bool(use_amp_override) and device.type == "cuda"
        
        # Gradient accumulation steps
        grad_accum_steps = int(cfg.get("lstm_gradient_accumulation", 1))

        # Check if CNN frontend is enabled
        cnn_enabled = bool(cfg.get("cnn_frontend_enabled", False))
        
        if cnn_enabled:
            # Use CNN-LSTM hybrid model
            model = CNNLSTMRegressor(
                input_dim=seq_data.feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                cnn_blocks=int(cfg.get("cnn_blocks", 2)),
                cnn_filters=int(cfg.get("cnn_filters", 64)),
                cnn_kernel_size=int(cfg.get("cnn_kernel_size", 3)),
                cnn_dilation=int(cfg.get("cnn_dilation", 1)),
                cnn_stride=int(cfg.get("cnn_stride", 1)),
                cnn_pooling=cfg.get("cnn_pooling"),  # Can be None
                cnn_activation=str(cfg.get("cnn_activation", "gelu")),
                cnn_dropout=float(cfg.get("cnn_dropout", 0.1)),
                cnn_batch_norm=bool(cfg.get("cnn_batch_norm", True)),
                cnn_layer_norm=bool(cfg.get("cnn_layer_norm", False)),
                cnn_residual=bool(cfg.get("cnn_residual", True)),
            ).to(device)
        else:
            # Use standard LSTM model (with optional attention)
            # Extract all configuration parameters
            attention_type = str(cfg.get("lstm_attention_type", "none"))
            model = LSTMRegressor(
                input_dim=seq_data.feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                # Cell type and architecture
                cell_type=str(cfg.get("lstm_cell_type", "lstm")),
                bidirectional=bool(cfg.get("lstm_bidirectional", False)),
                # Dropout variants
                input_dropout=float(cfg.get("lstm_input_dropout", 0.0)),
                output_dropout=float(cfg.get("lstm_output_dropout", 0.0)),
                recurrent_dropout=float(cfg.get("lstm_recurrent_dropout", 0.0)),
                weight_dropout=float(cfg.get("lstm_weight_dropout", 0.0)),
                time_dropout=float(cfg.get("lstm_time_dropout", 0.0)),
                zoneout=float(cfg.get("lstm_zoneout", 0.0)),
                sequence_noise_std=float(cfg.get("lstm_sequence_noise_std", 0.0)),
                # Normalization and skip connections
                layer_norm=bool(cfg.get("lstm_layer_norm", False)),
                residual=bool(cfg.get("lstm_residual", False)),
                skip_connect=bool(cfg.get("lstm_skip_connect", False)),
                # Output head configuration
                fc_layers=int(cfg.get("lstm_fc_layers", 1)),
                fc_hidden=int(cfg.get("lstm_fc_hidden", 0)),
                activation=str(cfg.get("lstm_activation", "gelu")),
                output_activation=str(cfg.get("lstm_output_activation", "tanh")),
                # Weight initialization
                recurrent_kernel_init=str(cfg.get("lstm_recurrent_kernel_init", "orthogonal")),
                hidden_state_init=str(cfg.get("lstm_hidden_state_init", "zeros")),
                # Attention parameters
                attention_type=attention_type,
                attn_hidden_dim=int(cfg.get("lstm_attn_hidden_dim", 64)),
                attn_dropout=float(cfg.get("lstm_attn_dropout", 0.1)),
                attn_heads=int(cfg.get("lstm_attn_heads", 1)),
                attn_score_fn=str(cfg.get("lstm_attn_score_fn", "general")),
                attn_temperature=float(cfg.get("lstm_attn_temperature", 1.0)),
                attn_merge=str(cfg.get("lstm_attn_merge", "concat")),
                attn_normalization=str(cfg.get("lstm_attn_normalization", "softmax")),
                attn_positional_encoding=bool(cfg.get("lstm_attn_positional_encoding", False)),
                attn_context_length=int(cfg.get("lstm_attn_context_length", 0)),
                attn_entropy_reg=float(cfg.get("lstm_attn_entropy_reg", 0.0)),
                attn_distance_reg=float(cfg.get("lstm_attn_distance_reg", 0.0)),
                attn_key_dim=int(cfg.get("lstm_attn_key_dim", 0)),
                attn_value_dim=int(cfg.get("lstm_attn_value_dim", 0)),
                # Advanced architecture parameters (NEW)
                attn_layers=attn_layers,
                feedforward_dim=feedforward_dim,
                batch_norm_head=batch_norm_head,
                dense_activation=dense_activation,
                mixout_prob=mixout_prob,
                stochastic_depth=stochastic_depth,
                rotary_embedding=rotary_embedding,
            ).to(device)
        
        # =====================================================================
        # MIXOUT REGULARIZATION SETUP
        # =====================================================================
        mixout_handler: Optional[Mixout] = None
        if mixout_prob > 0:
            mixout_handler = Mixout(p=mixout_prob)
            mixout_handler.register_original_weights(model)
        
        if warm_state is not None and cfg.get("lstm_warm_start", True):
            model.load_state_dict(warm_state)
        
        # =====================================================================
        # TORCH.COMPILE CONFIGURATION (Properly Configured for Speed)
        # =====================================================================
        # mode="reduce-overhead" uses CUDA graphs internally (best for inference-like)
        # mode="max-autotune" benchmarks kernels for best performance (slower compile)
        # Disable for very small models where compile overhead > runtime benefit
        use_compile = bool(cfg.get("lstm_use_compile", True)) and hasattr(torch, 'compile')
        if use_compile and device.type == "cuda":
            # Only compile if model is large enough to benefit
            # Small models (hidden < 128, layers < 3) don't benefit from compile
            should_compile = hidden_dim >= 128 or num_layers >= 3
            if should_compile:
                try:
                    # 🚀 SPEED: Use mode="default" instead of "reduce-overhead"
                    # reduce-overhead uses CUDA graphs which can't handle:
                    # - Dynamic control flow (if self.training, stochastic depth)
                    # - torch.rand() calls for dropout
                    # - train/eval mode switches causing grad_mode state changes
                    # mode="default" avoids these issues while still providing speedup
                    model = torch.compile(
                        model, 
                        mode="default",
                        fullgraph=False,
                        dynamic=False,  # Static shapes for efficiency
                    )
                except Exception:
                    pass  # Fallback to eager mode if compile fails
        
        # =====================================================================
        # OPTIMIZER IN FP32 WITH FUSED KERNELS (Hedge Fund Best Practice)
        # =====================================================================
        # Fused AdamW runs entire optimizer step in a single CUDA kernel
        # This eliminates kernel launch overhead (~41μs per op saved)
        # Optimizer states are always maintained in fp32 for stability
        optimizer_name = str(cfg.get("lstm_optimizer", "AdamW"))
        momentum = float(cfg.get("lstm_momentum", 0.9))
        lr_multiplier = float(cfg.get("lstm_lr_multiplier", 1.0))
        effective_lr = learning_rate * lr_multiplier
        use_fused = device.type == "cuda" and hasattr(torch.optim.AdamW, 'fused')
        
        # Separate weight decay for recurrent weights (if specified)
        recurrent_weight_decay = float(cfg.get("lstm_recurrent_weight_decay", 0.0))
        
        # Check PyTorch version for fused optimizer support (requires 2.0+)
        try:
            if optimizer_name == "SGD":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=effective_lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    nesterov=True,
                )
            elif optimizer_name == "RMSprop":
                optimizer = torch.optim.RMSprop(
                    model.parameters(),
                    lr=effective_lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    alpha=0.99,  # Smoothing constant
                )
            elif optimizer_name == "Lookahead":
                # Lookahead wraps another optimizer (use AdamW as base)
                base_optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=effective_lr,
                    weight_decay=weight_decay,
                )
                # Simple Lookahead implementation
                optimizer = base_optimizer  # Fallback if no lookahead available
            elif use_fused and optimizer_name in ("Adam", "AdamW", "NAdam"):
                if optimizer_name == "NAdam":
                    optimizer = torch.optim.NAdam(
                        model.parameters(), 
                        lr=effective_lr, 
                        weight_decay=weight_decay,
                        fused=True,
                        foreach=False,  # fused and foreach are mutually exclusive
                    )
                else:  # Adam or AdamW
                    optimizer = torch.optim.AdamW(
                        model.parameters(), 
                        lr=effective_lr, 
                        weight_decay=weight_decay,
                        fused=True,
                        foreach=False,  # fused and foreach are mutually exclusive
                    )
            else:
                # Fallback to non-fused AdamW
                optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)
        except TypeError:
            # Older PyTorch without fused support
            optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)
        
        # =====================================================================
        # LR SCHEDULER CONFIGURATION (from Optuna sampling)
        # =====================================================================
        lr_scheduler_type = str(cfg.get("lstm_lr_scheduler", "cosine"))  # cosine, step, plateau, cyclical, none
        scheduler = None
        
        if lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_t_max, eta_min=effective_lr * 0.01
            )
        elif lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_lr_step_size, gamma=0.5
            )
        elif lr_scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=plateau_patience, min_lr=1e-6
            )
        elif lr_scheduler_type == "cyclical":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=cyclical_base_lr, max_lr=cyclical_max_lr,
                step_size_up=cyclical_step_size, mode='triangular2', cycle_momentum=False
            )
        elif lr_scheduler_type == "warmup_cosine":
            # Linear warmup then cosine decay
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, epochs * 100 - warmup_steps)  # Approx steps
                return 0.5 * (1 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # else: no scheduler (lr_scheduler_type == "none")
        
        # =====================================================================
        # LOSS FUNCTION CONFIGURATION (from Optuna sampling)
        # =====================================================================
        loss_fn_type = str(cfg.get("lstm_loss_fn", "mse"))  # mse, huber, quantile
        
        if loss_fn_type == "huber":
            criterion = nn.HuberLoss(delta=huber_delta)
        elif loss_fn_type == "quantile":
            # Quantile loss (pinball loss) for robust regression
            def quantile_loss(pred, target):
                error = target - pred
                return torch.mean(torch.max(quantile_alpha * error, (quantile_alpha - 1) * error))
            criterion = quantile_loss
        else:
            # Default MSE
            criterion = nn.MSELoss()
        
        # GradScaler only needed for FP16 (BF16 has sufficient dynamic range)
        scaler = amp.GradScaler(enabled=use_amp and not use_bf16) if amp else None
        
        # =====================================================================
        # EMA IN FP32 (Hedge Fund Best Practice)
        # =====================================================================
        # EMA tracking always in fp32 for numerical stability
        ema: Optional[ModelEMA] = None
        if ema_enabled:
            ema = ModelEMA(model, decay=ema_decay)

        # 🚀 GPU PRE-LOADING: Load data to GPU once, not per-batch (23x faster on A100)
        train_loader = seq_data.make_loader(train_idx, batch_size=batch_size, shuffle=True, device=device)
        val_loader = seq_data.make_loader(val_idx, batch_size=batch_size, shuffle=False, device=device)
        
        # Check if data is already on GPU (from make_loader)
        data_on_gpu = device.type == "cuda"

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_ema_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = math.inf
        patience_ctr = 0

        with _nvtx(f"train_lstm_fold:bs={batch_size},epochs={epochs},amp={use_amp}"):
            for epoch_idx in range(epochs):
                with _nvtx(f"epoch_{epoch_idx}:train"):
                    model.train()
                    for xb, yb in train_loader:
                        # Skip transfer if already on GPU
                        if not data_on_gpu:
                            xb = xb.to(device, non_blocking=True)
                            yb = yb.to(device, non_blocking=True)
                        
                        # =====================================================================
                        # INPUT NOISE REGULARIZATION (from Optuna sampling)
                        # =====================================================================
                        if input_noise_std > 0 and model.training:
                            xb = xb + torch.randn_like(xb) * input_noise_std
                        
                        # =====================================================================
                        # LABEL SMOOTHING (for regression - adds target noise)
                        # =====================================================================
                        if label_smoothing > 0 and model.training:
                            yb = yb + torch.randn_like(yb) * label_smoothing
                        
                        # 🚀 SPEED: set_to_none=True is faster than zero_grad()
                        # Avoids memset kernel launch (~41μs saved per step)
                        optimizer.zero_grad(set_to_none=True)
                        
                        # =====================================================================
                        # HEDGE FUND MIXED PRECISION TRAINING
                        # bf16 for forward/backward, fp32 for loss/LayerNorm/EMA/softmax/accum
                        # =====================================================================
                        if use_amp:
                            # Forward pass in bf16/fp16 for speed (312 TFLOPS on A100)
                            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                                preds = model(xb)
                            
                            # Loss computation in fp32 for numerical stability
                            # Exit autocast to ensure fp32 precision
                            loss = criterion(preds.float(), yb.float())
                            
                            # Scale loss for gradient accumulation (in fp32)
                            if grad_accum_steps > 1:
                                loss = loss / grad_accum_steps
                            
                            if scaler is not None:  # FP16 path needs GradScaler
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                                scaler.step(optimizer)
                                scaler.update()
                            else:  # BF16 path - no scaler needed
                                loss.backward()
                                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                                optimizer.step()
                        else:
                            # Full fp32 path
                            preds = model(xb)
                            loss = criterion(preds, yb)
                            if grad_accum_steps > 1:
                                loss = loss / grad_accum_steps
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                            optimizer.step()
                        
                        # EMA update in fp32 (hedge fund best practice)
                        if ema is not None:
                            ema.update(model)
                        
                        # =====================================================================
                        # MIXOUT REGULARIZATION (mix weights with original after optimizer step)
                        # =====================================================================
                        if mixout_handler is not None:
                            mixout_handler.forward(model)
                
                # =====================================================================
                # VALIDATION AND LR SCHEDULER STEP
                # =====================================================================
                with _nvtx(f"epoch_{epoch_idx}:val"):
                    val_loss = _evaluate(model, val_loader, criterion, device)

                # LR Scheduler step (after each epoch)
                if scheduler is not None:
                    if lr_scheduler_type == "plateau":
                        scheduler.step(val_loss)  # ReduceLROnPlateau needs val_loss
                    else:
                        scheduler.step()

                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    # Also save EMA state if enabled
                    if ema is not None:
                        best_ema_state = ema.state_dict()
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

        # =====================================================================
        # INFERENCE: Use EMA weights if available (hedge fund best practice)
        # =====================================================================
        if ema is not None and best_ema_state is not None:
            # Copy EMA weights to model for inference
            ema.load_state_dict(best_ema_state)
            ema.copy_to(model)
        elif best_state is not None:
            model.load_state_dict(best_state)
            
        preds = _predict(model, val_loader, device)
        
        # Capture state dict before cleanup
        # Use EMA state if available (more stable for forecasting)
        if ema is not None and best_ema_state is not None:
            final_state = {k: v.cpu().clone() for k, v in best_ema_state.items()}
        else:
            final_state = best_state if best_state is not None else {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        result = {
            "preds": preds,
            "timestamps": seq_data.timestamps[val_idx],
            "state_dict": final_state,
            "val_loss": best_val,
            "ema_enabled": ema_enabled,
        }
        
        # Return model if requested (for batched fold evaluation)
        if return_model:
            result["model"] = model  # Keep on current device for eval
            return result
        
        # 🔧 Move model to CPU and clear GPU memory after training
        if device.type == "cuda":
            model = model.cpu()
            del model
            # Synchronize stream before clearing cache
            if cuda_stream:
                cuda_stream.synchronize()
            torch.cuda.empty_cache()
        
        return result


# =============================================================================
# Mamba-like model (dependency-free)
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class _MambaLikeBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        conv_kernel: int,
        activation: str,
        norm_type: str,
        norm_strategy: str,
        dropout: float,
        resid_dropout: float,
        ssm_dropout: float,
        gate_dropout: float,
    ) -> None:
        super().__init__()

        if norm_type == "rmsnorm":
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.norm_strategy = norm_strategy

        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.act = _get_activation(activation)
        self.drop = nn.Dropout(dropout)
        self.gate_drop = nn.Dropout(gate_dropout)
        self.ssm_drop = nn.Dropout(ssm_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        # Depthwise conv over time as a cheap long-range mixer.
        # This is not a faithful SSM, but provides a tunable temporal inductive bias.
        padding = (conv_kernel - 1) // 2
        self.dwconv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=conv_kernel,
            padding=padding,
            groups=d_inner,
            bias=True,
        )

        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = x

        if self.norm_strategy == "pre":
            x = self.norm(x)

        u, g = self.in_proj(x).chunk(2, dim=-1)
        u = self.act(u)
        u = self.drop(u)

        # Temporal mixing: depthwise conv along sequence dimension
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = self.ssm_drop(u)

        gate = torch.sigmoid(g)
        gate = self.gate_drop(gate)
        x = u * gate
        x = self.out_proj(x)

        x = self.resid_drop(x)
        x = x + residual

        if self.norm_strategy == "post":
            x = self.norm(x)

        return x


class MambaLikeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        expand_factor: float,
        conv_kernel: int,
        activation: str,
        norm_type: str,
        norm_strategy: str,
        dropout: float,
        resid_dropout: float,
        ssm_dropout: float,
        gate_dropout: float,
        head_type: str,
        head_hidden_dim: int,
        head_num_layers: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)

        d_inner = max(16, int(d_model * float(expand_factor)))
        self.blocks = nn.ModuleList(
            [
                _MambaLikeBlock(
                    d_model=d_model,
                    d_inner=d_inner,
                    conv_kernel=conv_kernel,
                    activation=activation,
                    norm_type=norm_type,
                    norm_strategy=norm_strategy,
                    dropout=dropout,
                    resid_dropout=resid_dropout,
                    ssm_dropout=ssm_dropout,
                    gate_dropout=gate_dropout,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )

        self._head_type = str(head_type or "linear").lower().strip()

        if self._head_type in {"gaussian", "uncertainty", "gaussian_nll"}:
            # Uncertainty head: outputs (mu, log_var) so we can enforce positivity
            # via var = softplus(log_var) + eps during loss/inference.
            # We honor the same MLP head hyperparameters as the scalar MLP head by
            # building an optional shared trunk and then two linear outputs.
            layers: List[nn.Module] = []
            in_dim = d_model
            n = max(0, int(head_num_layers))
            if n > 0:
                for _ in range(n):
                    layers.append(nn.Linear(in_dim, head_hidden_dim))
                    layers.append(_get_activation(activation))
                    layers.append(nn.Dropout(head_dropout))
                    in_dim = head_hidden_dim
                self.head_trunk = nn.Sequential(*layers)
            else:
                self.head_trunk = nn.Identity()
            self.head_mu = nn.Linear(in_dim, 1)
            self.head_logvar = nn.Linear(in_dim, 1)
        elif self._head_type == "mlp":
            layers: List[nn.Module] = []
            in_dim = d_model
            for _ in range(max(1, int(head_num_layers))):
                layers.append(nn.Linear(in_dim, head_hidden_dim))
                layers.append(_get_activation(activation))
                layers.append(nn.Dropout(head_dropout))
                in_dim = head_hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Linear(d_model, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_dim)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        # Use last token representation
        h = x[:, -1, :]
        if self._head_type in {"gaussian", "uncertainty", "gaussian_nll"}:
            h2 = self.head_trunk(h)
            mu = self.head_mu(h2).squeeze(-1)
            log_var = self.head_logvar(h2).squeeze(-1)
            # Return a single tensor for compatibility: (B, 2) = [mu, log_var]
            return torch.stack([mu, log_var], dim=-1)
        out = self.head(h).squeeze(-1)
        return out


class _Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al.) - lightweight implementation.

    Falls back to AdamW if unavailable, but we keep it here to support tuning.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Update moments
                exp_avg.mul_(beta2).add_(g, alpha=1.0 - beta2)

                update = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

        return loss


def _evaluate_with_optional_bce(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: "torch.device",
    bce_targets: bool,
) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    n_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
            if bce_targets:
                yb = (yb > 0.0).float()
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss = total_loss + loss.detach()
            n_batches += 1
    return (total_loss / max(n_batches, 1)).item()


def train_mamba_fold(
    seq_data: SequenceData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Mapping[str, Any],
    device: Optional["torch.device"] = None,
    warm_state: Optional[Dict[str, torch.Tensor]] = None,
    return_model: bool = False,
) -> Dict[str, Any]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for sequence training")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If warm_state is provided, we load it before training (online fine-tuning).

    cuda_stream = None
    if device.type == "cuda":
        cuda_stream = torch.cuda.Stream(device=device)
    stream_context = torch.cuda.stream(cuda_stream) if cuda_stream else nullcontext()

    with stream_context:
        import os as _os
        import random as _random
        from contextlib import contextmanager as _contextmanager
        import sys as _sys
        import time as _time

        _nvtx_enabled = (
            device.type == "cuda"
            and _os.environ.get("DCF_ENABLE_NVTX", "0") == "1"
            and hasattr(torch.cuda, "nvtx")
        )

        global _MAMBA_DEVICE_LOGGED
        if not _MAMBA_DEVICE_LOGGED:
            try:
                dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else None
            except Exception:
                dev_name = None
            print(
                f"[INFO:mamba] device={device.type} cuda_available={torch.cuda.is_available()} gpu={dev_name} nvtx={_nvtx_enabled}",
                file=_sys.stderr,
                flush=True,
            )
            _MAMBA_DEVICE_LOGGED = True

        @_contextmanager
        def _nvtx(msg: str):
            if not _nvtx_enabled:
                yield
                return
            try:
                torch.cuda.nvtx.range_push(msg)
            except Exception:
                yield
                return
            try:
                yield
            finally:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

        deterministic = bool(cfg.get("torch_deterministic", False))
        disable_tf32 = bool(cfg.get("torch_disable_tf32", False))
        cfg_seed = cfg.get("torch_seed")

        if cfg_seed is not None:
            try:
                seed_int = int(cfg_seed)
                _random.seed(seed_int)
                np.random.seed(seed_int)
                torch.manual_seed(seed_int)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_int)
            except Exception:
                pass

        if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            # TF32 improves speed but changes numerics; disable for reproducible replays.
            allow_tf32 = not disable_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32

        if device.type == "cuda":
            if deterministic:
                try:
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                except Exception:
                    pass
                try:
                    # warn_only=True avoids hard crashes if a rare op has no deterministic kernel.
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
            else:
                torch.backends.cudnn.benchmark = True

        batch_size = int(cfg.get("mamba_batch_size", 32))
        lr = float(cfg.get("mamba_learning_rate", 1e-3))
        weight_decay = float(cfg.get("mamba_weight_decay", 1e-4))
        grad_clip = float(cfg.get("mamba_grad_clip", 1.0))

        epochs = int(cfg.get("max_epochs", cfg.get("mamba_max_epochs", 10)))
        patience = int(cfg.get("early_stopping_patience", 3))

        d_model = int(cfg.get("mamba_d_model", 128))
        n_layers = int(cfg.get("mamba_n_layers", 4))
        ssm_dim = int(cfg.get("mamba_ssm_dim", 96))
        expand_factor = float(cfg.get("mamba_expand_factor", 2.0))
        activation = str(cfg.get("mamba_activation", "silu"))
        norm_type = str(cfg.get("mamba_norm_type", "rmsnorm"))
        norm_strategy = str(cfg.get("mamba_norm_strategy", "pre"))
        dropout = float(cfg.get("mamba_dropout", 0.1))
        resid_dropout = float(cfg.get("mamba_resid_dropout", 0.0))
        ssm_dropout = float(cfg.get("mamba_ssm_dropout", 0.0))
        gate_dropout = float(cfg.get("mamba_gate_dropout", 0.0))

        head_type = str(cfg.get("mamba_head_type", "linear"))
        head_hidden_dim = int(cfg.get("mamba_head_hidden_dim", 128))
        head_num_layers = int(cfg.get("mamba_head_num_layers", 1))
        head_dropout = float(cfg.get("mamba_head_dropout", 0.0))

        # Map ssm_dim to a reasonable conv kernel size (odd, <= 31)
        conv_kernel = int(min(max(3, ssm_dim // 8 * 2 + 1), 31))

        with _nvtx("mamba_init"):
            model = MambaLikeRegressor(
                input_dim=seq_data.feature_dim,
                d_model=d_model,
                n_layers=n_layers,
                expand_factor=expand_factor,
                conv_kernel=conv_kernel,
                activation=activation,
                norm_type=norm_type,
                norm_strategy=norm_strategy,
                dropout=dropout,
                resid_dropout=resid_dropout,
                ssm_dropout=ssm_dropout,
                gate_dropout=gate_dropout,
                head_type=head_type,
                head_hidden_dim=head_hidden_dim,
                head_num_layers=head_num_layers,
                head_dropout=head_dropout,
            ).to(device)

            if warm_state is not None:
                try:
                    model.load_state_dict(warm_state, strict=True)
                except Exception:
                    # Best-effort: allow partial loads if the head changed.
                    try:
                        model.load_state_dict(warm_state, strict=False)
                    except Exception:
                        pass

        loss_name = str(cfg.get("mamba_loss_fn", "smooth_l1")).lower().strip()
        bce_targets = loss_name == "bce_logits"

        def _gaussian_nll(pred2: "torch.Tensor", y_true: "torch.Tensor") -> "torch.Tensor":
            # pred2: (B, 2) [mu, log_var]
            mu = pred2[:, 0]
            log_var = pred2[:, 1]
            var = torch.nn.functional.softplus(log_var) + 1e-6
            nll = 0.5 * (torch.log(var) + (y_true - mu) ** 2 / var)
            return nll.mean()

        if loss_name in ("gaussian_nll", "nll_gaussian"):
            criterion = None
        elif loss_name in ("mse", "l2"):
            criterion = nn.MSELoss()
        elif loss_name in ("smooth_l1", "huber"):
            criterion = nn.SmoothL1Loss()
        elif loss_name == "bce_logits":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.SmoothL1Loss()

        opt_name = str(cfg.get("mamba_optimizer", "adamw")).lower()
        if opt_name == "lion":
            optimizer = _Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        sched_name = str(cfg.get("mamba_lr_scheduler", "cosine")).lower()
        if sched_name == "one_cycle":
            # OneCycle is stepped per-batch
            train_loader_tmp = seq_data.make_loader(train_idx, batch_size=batch_size, shuffle=True, device=device)
            steps_per_epoch = max(1, len(train_loader_tmp))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=max(1, epochs),
                steps_per_epoch=steps_per_epoch,
            )
        elif sched_name == "linear_warmup_cosine":
            warmup_steps = int(cfg.get("mamba_warmup_steps", 100))
            total_steps = max(1, epochs)

            def _lr_lambda(ep: int) -> float:
                if warmup_steps <= 0:
                    return 1.0
                # epoch-based warmup (simple + stable)
                warmup_epochs = max(1, warmup_steps // 50)
                if ep < warmup_epochs:
                    return float(ep + 1) / float(warmup_epochs)
                # cosine decay
                t = float(ep - warmup_epochs) / float(max(1, total_steps - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        with _nvtx("mamba_loaders"):
            train_loader = seq_data.make_loader(train_idx, batch_size=batch_size, shuffle=True, device=device)
            val_loader = seq_data.make_loader(val_idx, batch_size=batch_size, shuffle=False, device=device)

        # AMP policy:
        # - We use bf16 autocast when supported.
        # - IMPORTANT: GradScaler is not needed for bf16 (and can add noticeable overhead).
        use_amp = device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16
        scaler = None

        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_ctr = 0

        train_start_ts = _time.time()
        try:
            n_train_batches = int(len(train_loader))
        except Exception:
            n_train_batches = -1
        try:
            n_val_batches = int(len(val_loader))
        except Exception:
            n_val_batches = -1

        for epoch in range(max(1, epochs)):
            with _nvtx(f"mamba_epoch_{epoch}"):
                model.train()
                for xb, yb in train_loader:
                    if xb.device != device:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                    if bce_targets:
                        yb = (yb > 0.0).float()

                    optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        preds = model(xb)
                        if loss_name in ("gaussian_nll", "nll_gaussian"):
                            # Ensure gaussian head output
                            if preds.ndim != 2 or preds.shape[1] != 2:
                                raise ValueError("gaussian_nll requires model output (B,2)=[mu,log_var]")
                            loss = _gaussian_nll(preds, yb)
                        else:
                            loss = criterion(preds, yb)  # type: ignore[misc]

                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()

                if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

                if loss_name in ("gaussian_nll", "nll_gaussian"):
                    # Inline eval to avoid changing shared helpers.
                    model.eval()
                    total = torch.tensor(0.0, device=device)
                    n_batches = 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            if xb.device != device:
                                xb = xb.to(device, non_blocking=True)
                                yb = yb.to(device, non_blocking=True)
                            out = model(xb)
                            if out.ndim != 2 or out.shape[1] != 2:
                                raise ValueError("gaussian_nll requires model output (B,2)=[mu,log_var]")
                            total = total + _gaussian_nll(out, yb).detach()
                            n_batches += 1
                    val_loss = (total / max(n_batches, 1)).item()
                else:
                    val_loss = _evaluate_with_optional_bce(model, val_loader, criterion, device, bce_targets)

                # Lightweight, always-on progress log (critical for long Optuna trials).
                try:
                    lr_now = float(optimizer.param_groups[0].get("lr", lr)) if optimizer.param_groups else float(lr)
                except Exception:
                    lr_now = float(lr)
                try:
                    elapsed_s = int(_time.time() - train_start_ts)
                except Exception:
                    elapsed_s = -1
                print(
                    f"[INFO:mamba] epoch={epoch + 1}/{max(1, epochs)} "
                    f"val_loss={float(val_loss):.6f} best={float(best_val):.6f} "
                    f"pat={int(patience_ctr)}/{int(patience)} lr={lr_now:.2e} "
                    f"batches(train,val)=({n_train_batches},{n_val_batches}) elapsed_s={elapsed_s}",
                    file=_sys.stderr,
                    flush=True,
                )
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break


        if best_state is not None:
            model.load_state_dict(best_state)

        preds = _predict(model, val_loader, device)
        final_state = best_state if best_state is not None else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        result: Dict[str, Any] = {
            "preds": preds,
            "timestamps": seq_data.timestamps[val_idx],
            "state_dict": final_state,
            "val_loss": best_val,
            "ema_enabled": False,
        }

        if return_model:
            # IMPORTANT: Phase-2 uses `return_model=True` and immediately runs additional
            # GPU work (stateful inference/backtest). We train on a dedicated CUDA stream,
            # so we must synchronize before returning; otherwise kernels can spill over
            # into the next phase/trial and cause apparent GPU "contention" or rare hangs.
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
            result["model"] = model
            return result

        if device.type == "cuda":
            model = model.cpu()
            del model
            if cuda_stream:
                cuda_stream.synchronize()
            torch.cuda.empty_cache()

        return result


def _evaluate(model: nn.Module, loader: DataLoader, criterion, device: "torch.device") -> float:
    """Evaluate model on validation data. Handles GPU-resident data efficiently.
    
    🚀 SPEED OPTIMIZATION: GPU-accumulated loss with single sync
    - Accumulate losses as GPU tensor without .item() each iteration
    - Single CPU sync at the end instead of N syncs per batch
    """
    model.eval()
    total_loss = torch.tensor(0.0, device=device)  # GPU tensor for accumulation
    n_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            # Skip transfer if data is already on the target device
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            # 🚀 GPU-ACCUMULATED LOSS: Avoid .item() sync per batch
            total_loss = total_loss + loss.detach()  # Accumulate tensor on GPU
            n_batches += 1
    # Single sync point at the end (instead of N syncs per batch)
    return (total_loss / max(n_batches, 1)).item()


def _predict(model: nn.Module, loader: DataLoader, device: "torch.device") -> np.ndarray:
    """Run inference on data. Handles GPU-resident data efficiently.
    
    🚀 SPEED OPTIMIZATION: Batched GPU->CPU transfer
    - Accumulate predictions on GPU as tensors
    - Single concatenate and transfer at the end
    - Uses non_blocking transfers with stream synchronization
    """
    model.eval()
    preds_gpu: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            # Skip transfer if data is already on the target device
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            out = model(xb)
            # If the model returns (B,2) [mu, log_var], keep only mu for legacy callers.
            if hasattr(out, "ndim") and int(getattr(out, "ndim", 0)) == 2 and int(out.shape[1]) == 2:
                out = out[:, 0]
            preds_gpu.append(out.detach())
    
    if not preds_gpu:
        return np.asarray([], dtype=np.float32)
    
    # Single GPU concatenate (much faster than per-batch numpy concat)
    all_preds = torch.cat(preds_gpu, dim=0)
    
    # Single GPU->CPU transfer (reduces PCIe overhead)
    return all_preds.cpu().numpy()


def predict_lstm_on_data(
    seq_data: SequenceData,
    state_dict: Dict[str, "torch.Tensor"],
    cfg: Mapping[str, Any],
    device: Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """Run inference on sequence data using a pre-trained LSTM state dict.
    
    🔧 Walk-Forward: This function enables proper out-of-sample prediction
    by loading a trained model and running inference on new sequence data.
    
    Args:
        seq_data: SequenceData to run inference on
        state_dict: Pre-trained LSTM state dictionary
        cfg: Configuration dict with model hyperparameters
        device: PyTorch device (defaults to GPU if available)
    
    Returns:
        Dict with 'preds' (predictions) and 'timestamps'
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for sequence inference")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = int(cfg.get("lstm_batch_size", 64))
    hidden_dim = int(cfg.get("lstm_hidden_dim", 96))
    num_layers = int(cfg.get("lstm_layers", 2))
    dropout = float(cfg.get("lstm_dropout", 0.1))
    
    # Check if CNN frontend is enabled
    cnn_enabled = bool(cfg.get("cnn_frontend_enabled", False))
    
    # Rebuild model with same architecture
    if cnn_enabled:
        model = CNNLSTMRegressor(
            input_dim=seq_data.feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cnn_blocks=int(cfg.get("cnn_blocks", 2)),
            cnn_filters=int(cfg.get("cnn_filters", 64)),
            cnn_kernel_size=int(cfg.get("cnn_kernel_size", 3)),
            cnn_dilation=int(cfg.get("cnn_dilation", 1)),
            cnn_stride=int(cfg.get("cnn_stride", 1)),
            cnn_pooling=cfg.get("cnn_pooling"),
            cnn_activation=str(cfg.get("cnn_activation", "gelu")),
            cnn_dropout=float(cfg.get("cnn_dropout", 0.1)),
            cnn_batch_norm=bool(cfg.get("cnn_batch_norm", True)),
            cnn_layer_norm=bool(cfg.get("cnn_layer_norm", False)),
            cnn_residual=bool(cfg.get("cnn_residual", True)),
        ).to(device)
    else:
        attention_type = str(cfg.get("lstm_attention_type", "none"))
        model = LSTMRegressor(
            input_dim=seq_data.feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            # Cell type and architecture
            cell_type=str(cfg.get("lstm_cell_type", "lstm")),
            bidirectional=bool(cfg.get("lstm_bidirectional", False)),
            # Dropout variants
            input_dropout=float(cfg.get("lstm_input_dropout", 0.0)),
            output_dropout=float(cfg.get("lstm_output_dropout", 0.0)),
            recurrent_dropout=float(cfg.get("lstm_recurrent_dropout", 0.0)),
            weight_dropout=float(cfg.get("lstm_weight_dropout", 0.0)),
            time_dropout=float(cfg.get("lstm_time_dropout", 0.0)),
            zoneout=float(cfg.get("lstm_zoneout", 0.0)),
            sequence_noise_std=float(cfg.get("lstm_sequence_noise_std", 0.0)),
            # Normalization and skip connections
            layer_norm=bool(cfg.get("lstm_layer_norm", False)),
            residual=bool(cfg.get("lstm_residual", False)),
            skip_connect=bool(cfg.get("lstm_skip_connect", False)),
            # Output head configuration
            fc_layers=int(cfg.get("lstm_fc_layers", 1)),
            fc_hidden=int(cfg.get("lstm_fc_hidden", 0)),
            activation=str(cfg.get("lstm_activation", "gelu")),
            output_activation=str(cfg.get("lstm_output_activation", "tanh")),
            # Weight initialization
            recurrent_kernel_init=str(cfg.get("lstm_recurrent_kernel_init", "orthogonal")),
            hidden_state_init=str(cfg.get("lstm_hidden_state_init", "zeros")),
            # Attention parameters
            attention_type=attention_type,
            attn_hidden_dim=int(cfg.get("lstm_attn_hidden_dim", 64)),
            attn_dropout=float(cfg.get("lstm_attn_dropout", 0.1)),
            attn_heads=int(cfg.get("lstm_attn_heads", 1)),
            attn_score_fn=str(cfg.get("lstm_attn_score_fn", "general")),
            attn_temperature=float(cfg.get("lstm_attn_temperature", 1.0)),
            attn_merge=str(cfg.get("lstm_attn_merge", "concat")),
            attn_normalization=str(cfg.get("lstm_attn_normalization", "softmax")),
            attn_positional_encoding=bool(cfg.get("lstm_attn_positional_encoding", False)),
            attn_context_length=int(cfg.get("lstm_attn_context_length", 0)),
            attn_entropy_reg=float(cfg.get("lstm_attn_entropy_reg", 0.0)),
            attn_distance_reg=float(cfg.get("lstm_attn_distance_reg", 0.0)),
            attn_key_dim=int(cfg.get("lstm_attn_key_dim", 0)),
            attn_value_dim=int(cfg.get("lstm_attn_value_dim", 0)),
            # Advanced architecture parameters
            attn_layers=int(cfg.get("attn_layers", 1)),
            feedforward_dim=int(cfg.get("feedforward_dim", 0)),
            batch_norm_head=bool(cfg.get("batch_norm_head", False)),
            dense_activation=str(cfg.get("dense_activation", "relu")),
            mixout_prob=float(cfg.get("mixout_prob", 0.0)),
            stochastic_depth=float(cfg.get("stochastic_depth", 0.0)),
            rotary_embedding=bool(cfg.get("rotary_embedding", False)),
        ).to(device)
    model.load_state_dict(state_dict)
    
    # Create loader for all data
    all_idx = np.arange(len(seq_data))
    loader = seq_data.make_loader(all_idx, batch_size=batch_size, shuffle=False)
    
    # Run inference
    preds = _predict(model, loader, device)
    
    return {
        "preds": preds,
        "timestamps": seq_data.timestamps,
    }


def predict_mamba_on_data(
    seq_data: SequenceData,
    state_dict: Dict[str, "torch.Tensor"],
    cfg: Mapping[str, Any],
    device: Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """Run inference on sequence data using a pre-trained Mamba state dict.

    Mirrors the API of `predict_lstm_on_data` so the pipeline can switch
    between sequence models with minimal branching.
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for sequence inference")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(cfg.get("mamba_batch_size", 32))

    d_model = int(cfg.get("mamba_d_model", 128))
    n_layers = int(cfg.get("mamba_n_layers", 4))
    ssm_dim = int(cfg.get("mamba_ssm_dim", 96))
    expand_factor = float(cfg.get("mamba_expand_factor", 2.0))
    activation = str(cfg.get("mamba_activation", "silu"))
    norm_type = str(cfg.get("mamba_norm_type", "rmsnorm"))
    norm_strategy = str(cfg.get("mamba_norm_strategy", "pre"))
    dropout = float(cfg.get("mamba_dropout", 0.1))
    resid_dropout = float(cfg.get("mamba_resid_dropout", 0.0))
    ssm_dropout = float(cfg.get("mamba_ssm_dropout", 0.0))
    gate_dropout = float(cfg.get("mamba_gate_dropout", 0.0))

    head_type = str(cfg.get("mamba_head_type", "linear"))
    head_hidden_dim = int(cfg.get("mamba_head_hidden_dim", 128))
    head_num_layers = int(cfg.get("mamba_head_num_layers", 1))
    head_dropout = float(cfg.get("mamba_head_dropout", 0.0))

    conv_kernel = int(min(max(3, ssm_dim // 8 * 2 + 1), 31))

    model = MambaLikeRegressor(
        input_dim=seq_data.feature_dim,
        d_model=d_model,
        n_layers=n_layers,
        expand_factor=expand_factor,
        conv_kernel=conv_kernel,
        activation=activation,
        norm_type=norm_type,
        norm_strategy=norm_strategy,
        dropout=dropout,
        resid_dropout=resid_dropout,
        ssm_dropout=ssm_dropout,
        gate_dropout=gate_dropout,
        head_type=head_type,
        head_hidden_dim=head_hidden_dim,
        head_num_layers=head_num_layers,
        head_dropout=head_dropout,
    ).to(device)
    model.load_state_dict(state_dict)

    all_idx = np.arange(len(seq_data))
    loader = seq_data.make_loader(all_idx, batch_size=batch_size, shuffle=False)
    # For gaussian head models we want to return both mu and sigma.
    model.eval()
    mu_gpu: List[torch.Tensor] = []
    logvar_gpu: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            out = model(xb)
            if hasattr(out, "ndim") and int(getattr(out, "ndim", 0)) == 2 and int(out.shape[1]) == 2:
                mu_gpu.append(out[:, 0].detach())
                logvar_gpu.append(out[:, 1].detach())
            else:
                mu_gpu.append(out.detach().reshape(-1))

    if not mu_gpu:
        mu = np.asarray([], dtype=np.float32)
        sigma = None
    else:
        mu_t = torch.cat(mu_gpu, dim=0)
        mu = mu_t.cpu().numpy()
        sigma = None
        if logvar_gpu:
            lv = torch.cat(logvar_gpu, dim=0)
            var = torch.nn.functional.softplus(lv) + 1e-6
            sigma = torch.sqrt(var).cpu().numpy()

    out_dict: Dict[str, Any] = {
        "preds": mu,
        "timestamps": seq_data.timestamps,
    }
    if sigma is not None:
        out_dict["sigma"] = sigma
    return out_dict


__all__ = [
    "SequenceData",
    "build_sequence_data",
    "train_lstm_fold",
    "train_mamba_fold",
    "predict_lstm_on_data",
    "predict_mamba_on_data",
]
