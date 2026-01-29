# Level 2: Cross-Section Mamba Implementation Plan

## ✅ IMPLEMENTATION STATUS

**Status: IMPLEMENTED** (2025-01-27)

### Components Implemented:
| Component | Location | Status |
|-----------|----------|--------|
| `CrossSectionSequenceData` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `build_cross_section_data()` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `CrossSectionAttention` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `CrossSectionMamba` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `train_cross_section_mamba()` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `CrossSectionInferenceBuffer` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `compute_vram_requirements()` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `probe_vram_for_cross_section()` | `src/stage_b/sequence_models.py` | ✅ Complete |
| `soft_rank()` | `src/stage_b/sequence_models.py` | ✅ Complete |
| Phase2 integration helpers | `src/stage_b_stateful/phase2_stateful.py` | ✅ Complete |

### Configuration Documented:
- `Phase2_Optuna_Search_Space.md` Section 12: Cross-Section Mamba parameters

### Usage:
```python
from src.stage_b.sequence_models import (
    CrossSectionSequenceData,
    build_cross_section_data,
    CrossSectionMamba,
    train_cross_section_mamba,
    CrossSectionInferenceBuffer,
    compute_vram_requirements,
)

# Enable in Phase2 config - see Section 13 for 3-stage tuning
cfg = {
    "cs_enabled": True,
    # Stage 1: Tune optimizer (freeze architecture)
    "cs_learning_rate": 3e-4,
    "cs_weight_decay": 1e-4,
    "cs_dropout": 0.1,
    "cs_grad_clip": 1.0,
    "cs_batch_size": 16,
    # Stage 2: Tune cross-section capacity
    "cs_cross_section_heads": 4,
    "cs_n_cross_section_layers": 2,
    "cs_adjacency_temperature": 1.0,
    # Stage 3: Tune temporal capacity + output discipline
    "cs_d_model": 128,
    "cs_n_temporal_layers": 4,
    "cs_sigma_floor": 0.01,
    "cs_sigma_ceiling": 2.0,
    # ... see Phase2_Optuna_Search_Space.md Section 13
}
```

---

## Overview

Transform from per-symbol models `[B, T, F]` to true multi-symbol models `[B, T, N, F]` that learn cross-section dynamics directly.

**Current (Level 1):**
```
For each symbol i:
  x_i: [B, T, F]  → MambaLikeRegressor → (mu_i, sigma_i)
```

**Target (Level 2):**
```
For batch of N symbols:
  X: [B, T, N, F] → CrossSectionMamba → (mu, sigma): [B, N]
```

---

## Architecture Design

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CrossSectionMamba                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: X ∈ R^[B, T, N, F]                                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ 1. TEMPORAL TOWER (per symbol, shared weights)             │     │
│  │    For each symbol i ∈ [1..N]:                             │     │
│  │      h_i = TemporalEncoder(X[:, :, i, :])  → [B, d_model]  │     │
│  │    Stack: H ∈ R^[B, N, d_model]                            │     │
│  └────────────────────────────────────────────────────────────┘     │
│                           │                                          │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ 2. CROSS-SECTION MESSAGE PASSING                          │     │
│  │    A_t = BuildAdjacency(H, mask) or use SGC adjacency      │     │
│  │    H' = CrossSectionAttention(A_t, H)  → [B, N, d_model]   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                           │                                          │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ 3. PREDICTION HEAD (per symbol)                            │     │
│  │    For each symbol i:                                      │     │
│  │      (mu_i, sigma_i) = GaussianHead(H'[:, i, :])          │     │
│  │    Output: mu ∈ R^[B, N], sigma ∈ R^[B, N]                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Multi-Symbol Dataset Builder

### File: `src/stage_b/sequence_models.py` (additions)

#### 1.1 CrossSectionSequenceData

```python
@dataclass
class CrossSectionSequenceData:
    """Dataset for cross-section learning with shape [B, T, N, F]."""
    
    sequences: np.ndarray      # [n_samples, T, N, F]
    targets: np.ndarray        # [n_samples, N] - per-symbol targets
    timestamps: np.ndarray     # [n_samples] - batch timestamp (end of sequence)
    symbol_order: List[str]    # [N] - which symbol is in each slot
    symbol_mask: np.ndarray    # [n_samples, N] - 1.0 if symbol valid, 0.0 if padded
    adjacency: Optional[np.ndarray] = None  # [n_samples, N, N] - precomputed adjacency
```

#### 1.2 Build Function

```python
def build_cross_section_data(
    X_by_symbol: Dict[str, np.ndarray],      # symbol -> [T_i, F]
    y_by_symbol: Dict[str, np.ndarray],      # symbol -> [T_i]
    ts_by_symbol: Dict[str, np.ndarray],     # symbol -> [T_i] timestamps
    seq_len: int,
    universe_size: int = 256,                 # N - fixed cross-section size
    min_symbols_per_batch: int = 32,          # Minimum symbols to form batch
    adjacency_builder: Optional[Callable] = None,  # Function to compute A_t
) -> CrossSectionSequenceData:
    """Build cross-section training data.
    
    Strategy:
    1. Find common timestamps where enough symbols have data
    2. For each valid timestamp t:
       - Select up to N symbols with data at [t-T+1, t]
       - Pad to N if fewer symbols available
       - Build adjacency A_t from historical data
    3. Return aligned [n_samples, T, N, F] tensor
    """
```

#### 1.3 Universe Selection Strategy

**Option A: Fixed Universe (simpler, recommended for v1)**
- Define N symbols at start of training period
- Use same N for all batches
- Pad symbols that delist/have gaps with zeros + mask

**Option B: Time-Varying Universe (more realistic, Phase 2)**
- Universe changes each week/month
- Need symbol→slot mapping per batch
- More complex but handles IPOs/delistings

**Recommendation:** Start with Option A, add Option B later.

---

## Phase 2: CrossSectionMamba Model

### File: `src/stage_b/sequence_models.py` (additions)

#### 2.1 Cross-Section Attention Module

```python
class CrossSectionAttention(nn.Module):
    """Graph attention over symbols at a single time step.
    
    Supports:
    - Learned adjacency (from symbol embeddings)
    - External adjacency (from SGC)
    - Multi-head attention with edge weights
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_learned_adjacency: bool = False,
        adjacency_temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Optional learned adjacency
        self.use_learned_adjacency = use_learned_adjacency
        if use_learned_adjacency:
            self.edge_proj = nn.Linear(d_model * 2, n_heads)  # Edge weight per head
        
        self.dropout = nn.Dropout(dropout)
        self.temperature = adjacency_temperature
        
    def forward(
        self,
        H: torch.Tensor,                      # [B, N, d_model]
        adjacency: Optional[torch.Tensor] = None,  # [B, N, N] or [N, N]
        mask: Optional[torch.Tensor] = None,  # [B, N] - 1 if valid, 0 if padded
    ) -> torch.Tensor:
        """
        Returns: H' ∈ R^[B, N, d_model]
        """
        B, N, D = H.shape
        
        # Compute Q, K, V
        Q = self.q_proj(H).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        K = self.k_proj(H).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(H).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, heads, N, N]
        
        # Apply external adjacency as edge weights (if provided)
        if adjacency is not None:
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0).expand(B, -1, -1)
            # Convert to attention bias (log-space)
            adj_bias = torch.log(adjacency.clamp(min=1e-6)).unsqueeze(1)  # [B, 1, N, N]
            scores = scores + adj_bias / self.temperature
        
        # Apply mask (set padded symbols to -inf)
        if mask is not None:
            # mask: [B, N] -> attn_mask: [B, 1, 1, N]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, V)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out
```

#### 2.2 CrossSectionMamba Model

```python
class CrossSectionMamba(nn.Module):
    """Multi-symbol Mamba with cross-section message passing.
    
    Input shape: [B, T, N, F]
    Output shape: (mu, sigma) each [B, N]
    """
    
    def __init__(
        self,
        input_dim: int,          # F - features per symbol
        d_model: int = 128,
        n_temporal_layers: int = 4,
        n_cross_section_layers: int = 2,
        expand_factor: float = 2.0,
        conv_kernel: int = 4,
        activation: str = "silu",
        norm_type: str = "rmsnorm",
        dropout: float = 0.1,
        # Cross-section parameters
        cross_section_heads: int = 4,
        cross_section_dropout: float = 0.1,
        use_learned_adjacency: bool = False,  # v1.5: learn residual
        adjacency_temperature: float = 1.0,
        # Temporal pooling for cross-section input
        temporal_pool_window: int = 1,  # 1 = T only, 5 = T-4..T pooling
        temporal_pool_mode: str = "last",  # "last", "mean", "attention"
        # Head parameters
        head_hidden_dim: int = 64,
        head_num_layers: int = 2,
        head_dropout: float = 0.0,
        # Regularization
        sigma_floor: float = 0.01,  # Minimum sigma to prevent collapse
        sigma_ceiling: float = 2.0,  # Maximum sigma to prevent overdispersion
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.temporal_pool_window = temporal_pool_window
        self.temporal_pool_mode = temporal_pool_mode
        self.sigma_floor = sigma_floor
        self.sigma_ceiling = sigma_ceiling
        
        # Input projection
        self.in_proj = nn.Linear(input_dim, d_model)
        
        # Temporal tower (shared across symbols)
        self.temporal_blocks = nn.ModuleList([
            _MambaLikeBlock(
                d_model=d_model,
                d_inner=int(d_model * expand_factor),
                conv_kernel=conv_kernel,
                activation=activation,
                norm_type=norm_type,
                norm_strategy="pre",
                dropout=dropout,
                resid_dropout=dropout,
                ssm_dropout=0.0,
                gate_dropout=0.0,
            )
            for _ in range(n_temporal_layers)
        ])
        
        # Optional temporal attention pooling
        if temporal_pool_mode == "attention" and temporal_pool_window > 1:
            self.temporal_pool_attn = nn.MultiheadAttention(
                d_model, num_heads=4, dropout=0.1, batch_first=True
            )
        
        # Cross-section message passing layers
        self.cross_section_layers = nn.ModuleList([
            CrossSectionAttention(
                d_model=d_model,
                n_heads=cross_section_heads,
                dropout=cross_section_dropout,
                use_learned_adjacency=use_learned_adjacency,
                adjacency_temperature=adjacency_temperature,
            )
            for _ in range(n_cross_section_layers)
        ])
        
        # Layer norms for cross-section
        self.cross_section_norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(n_cross_section_layers)
        ])
        
        # Gaussian prediction head (per-symbol)
        head_layers = []
        in_dim = d_model
        for _ in range(head_num_layers):
            head_layers.extend([
                nn.Linear(in_dim, head_hidden_dim),
                nn.SiLU(),
                nn.Dropout(head_dropout),
            ])
            in_dim = head_hidden_dim
        self.head_trunk = nn.Sequential(*head_layers) if head_layers else nn.Identity()
        self.head_mu = nn.Linear(in_dim, 1)
        self.head_logvar = nn.Linear(in_dim, 1)
        
    def _pool_temporal(self, h: torch.Tensor) -> torch.Tensor:
        """Pool temporal states for cross-section input.
        
        Args:
            h: [B*N, T, d_model]
        Returns:
            h_pooled: [B*N, d_model]
        """
        if self.temporal_pool_window <= 1 or self.temporal_pool_mode == "last":
            return h[:, -1, :]
        
        # Pool over last `window` steps
        window = min(self.temporal_pool_window, h.shape[1])
        h_window = h[:, -window:, :]  # [B*N, window, d_model]
        
        if self.temporal_pool_mode == "mean":
            return h_window.mean(dim=1)
        elif self.temporal_pool_mode == "attention":
            # Query is last state, keys/values are window
            query = h[:, -1:, :]  # [B*N, 1, d_model]
            out, _ = self.temporal_pool_attn(query, h_window, h_window)
            return out.squeeze(1)
        else:
            return h[:, -1, :]
        
    def forward(
        self,
        x: torch.Tensor,                      # [B, T, N, F]
        adjacency: Optional[torch.Tensor] = None,  # [B, N, N] or [N, N]
        mask: Optional[torch.Tensor] = None,  # [B, N] - 1 if valid, 0 if padded/inactive
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (mu, sigma) each [B, N]
        
        Note: Returns sigma (not logvar) with floor/ceiling enforced.
        """
        B, T, N, F = x.shape
        
        # 1. TEMPORAL TOWER: Process each symbol independently
        # Reshape to process all symbols in parallel: [B*N, T, F]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        
        # Input projection
        h = self.in_proj(x_flat)  # [B*N, T, d_model]
        
        # Temporal blocks (with gradient checkpointing for memory)
        for block in self.temporal_blocks:
            h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
        
        # Pool temporal states for cross-section input
        h_final = self._pool_temporal(h)  # [B*N, d_model]
        
        # Reshape back to [B, N, d_model]
        H = h_final.view(B, N, -1)
        
        # 2. CROSS-SECTION MESSAGE PASSING
        for norm, cross_attn in zip(self.cross_section_norms, self.cross_section_layers):
            H_normed = norm(H)
            H = H + cross_attn(H_normed, adjacency=adjacency, mask=mask)
        
        # 3. PREDICTION HEAD (per symbol)
        h_head = self.head_trunk(H)  # [B, N, head_hidden_dim]
        mu = self.head_mu(h_head).squeeze(-1)  # [B, N]
        logvar = self.head_logvar(h_head).squeeze(-1)  # [B, N]
        
        # Convert to sigma with floor/ceiling
        sigma = torch.sqrt(F.softplus(logvar) + 1e-6)
        sigma = sigma.clamp(min=self.sigma_floor, max=self.sigma_ceiling)
        
        return mu, sigma
```

---

## Phase 3: Training Loop Changes

### File: `src/stage_b/sequence_models.py` (modify `train_mamba_fold`)

#### 3.1 New Training Function

```python
def train_cross_section_mamba(
    cross_section_data: CrossSectionSequenceData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Mapping[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train CrossSectionMamba model.
    
    Key differences from train_mamba_fold:
    1. Loss is summed over symbols (with masking)
    2. Adjacency is passed per batch
    3. Validation metrics are per-symbol Sharpe/IR
    4. Includes rank stability and sigma discipline regularizers
    """
    
    # Extract config
    d_model = cfg.get("cs_d_model", 128)
    n_temporal_layers = cfg.get("cs_n_temporal_layers", 4)
    n_cross_section_layers = cfg.get("cs_n_cross_section_layers", 2)
    temporal_pool_window = cfg.get("cs_temporal_pool_window", 5)  # T-4..T
    temporal_pool_mode = cfg.get("cs_temporal_pool_mode", "mean")
    
    # Regularization weights
    rank_stability_weight = cfg.get("cs_rank_stability_weight", 0.01)
    sigma_floor = cfg.get("cs_sigma_floor", 0.01)
    sigma_ceiling = cfg.get("cs_sigma_ceiling", 2.0)
    
    # Build model
    model = CrossSectionMamba(
        input_dim=cross_section_data.sequences.shape[-1],  # F
        d_model=d_model,
        n_temporal_layers=n_temporal_layers,
        n_cross_section_layers=n_cross_section_layers,
        temporal_pool_window=temporal_pool_window,
        temporal_pool_mode=temporal_pool_mode,
        sigma_floor=sigma_floor,
        sigma_ceiling=sigma_ceiling,
        # ...
    ).to(device)
    
    # VRAM probe before training
    if device is not None and device.type == "cuda":
        probe_result = probe_vram_for_cross_section(
            model,
            B=cfg.get("cs_batch_size", 16),
            T=cross_section_data.sequences.shape[1],
            N=cross_section_data.sequences.shape[2],
            F=cross_section_data.sequences.shape[3],
            device=device,
        )
        if not probe_result["fits"]:
            logger.warning(f"[CS-Mamba] VRAM probe failed: {probe_result['peak_gb']:.1f}GB > {probe_result['available_gb']:.1f}GB")
            # Suggest smaller batch
            suggested_B = probe_result["max_B_estimate"]
            logger.warning(f"[CS-Mamba] Suggested batch size: {suggested_B}")
    
    # Previous batch predictions for rank stability
    prev_mu: Optional[torch.Tensor] = None
    
    # Training loop
    for epoch in range(max_epochs):
        model.train()
        for batch_idx in train_loader:
            x = torch.tensor(cross_section_data.sequences[batch_idx], device=device)  # [B, T, N, F]
            y = torch.tensor(cross_section_data.targets[batch_idx], device=device)    # [B, N]
            mask = torch.tensor(cross_section_data.symbol_mask[batch_idx], device=device)  # [B, N]
            adj = None
            if cross_section_data.adjacency is not None:
                adj = torch.tensor(cross_section_data.adjacency[batch_idx], device=device)
            
            with torch.cuda.amp.autocast():
                mu, sigma = model(x, adjacency=adj, mask=mask)
                
                # 1. Gaussian NLL loss (masked)
                nll = 0.5 * (torch.log(sigma**2) + (y - mu)**2 / (sigma**2))  # [B, N]
                nll_loss = (nll * mask).sum() / mask.sum()
                
                # 2. Rank stability regularizer (reduces day-to-day flip noise)
                rank_loss = torch.tensor(0.0, device=device)
                if prev_mu is not None and rank_stability_weight > 0:
                    # Spearman-like: penalize rank inversions between consecutive batches
                    # Use differentiable approximation via soft ranking
                    curr_ranks = soft_rank(mu, mask)  # [B, N]
                    prev_ranks = soft_rank(prev_mu, mask)  # [B, N]
                    rank_loss = ((curr_ranks - prev_ranks) ** 2 * mask).sum() / mask.sum()
                    rank_loss = rank_loss * rank_stability_weight
                
                # Total loss
                loss = nll_loss + rank_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Store for next batch
            prev_mu = mu.detach()
    
    return {"model": model, "metrics": {...}}


def soft_rank(x: torch.Tensor, mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable soft ranking for regularization.
    
    Args:
        x: [B, N] values to rank
        mask: [B, N] valid symbols
    Returns:
        ranks: [B, N] soft ranks in [0, 1]
    """
    # Mask invalid symbols with large negative value
    x_masked = x.clone()
    x_masked[mask == 0] = -1e9
    
    # Pairwise comparisons with softmax
    # rank_i = sum_j sigmoid((x_i - x_j) / temp)
    diff = x_masked.unsqueeze(-1) - x_masked.unsqueeze(-2)  # [B, N, N]
    ranks = torch.sigmoid(diff / temperature).sum(dim=-1)   # [B, N]
    
    # Normalize to [0, 1]
    n_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    ranks = ranks / n_valid
    
    return ranks
```

#### 3.2 Memory Optimization

For `[B, T, N, F]` with B=32, T=127, N=256, F=200:
- Input tensor: 32 × 127 × 256 × 200 × 4 bytes = **330 MB per batch**
- Need gradient checkpointing for temporal tower

```python
# In forward():
from torch.utils.checkpoint import checkpoint

# Checkpoint temporal blocks to reduce memory
for block in self.temporal_blocks:
    h = checkpoint(block, h, use_reentrant=False)
```

---

## Phase 4: Inference Loop Changes

### File: `src/stage_b_stateful/phase2_stateful.py` (modifications)

#### 4.1 Buffer Management

```python
class CrossSectionInferenceBuffer:
    """Maintains rolling buffers for all symbols in cross-section inference."""
    
    def __init__(
        self,
        symbols: List[str],
        seq_len: int,
        feature_dim: int,
        device: torch.device,
    ):
        self.symbols = symbols
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.device = device
        
        # Rolling buffer: [N, T, F]
        self.buffer = torch.zeros(
            len(symbols), seq_len, feature_dim,
            device=device, dtype=torch.float32
        )
        
        # Validity mask: [N, T] - which timesteps are valid
        self.valid_mask = torch.zeros(
            len(symbols), seq_len,
            device=device, dtype=torch.bool
        )
        
    def update(self, symbol: str, features: torch.Tensor) -> None:
        """Add new timestep for a symbol. features: [F]"""
        idx = self.symbol_to_idx.get(symbol)
        if idx is None:
            return
        
        # Roll buffer left and add new features at end
        self.buffer[idx, :-1] = self.buffer[idx, 1:].clone()
        self.buffer[idx, -1] = features
        
        self.valid_mask[idx, :-1] = self.valid_mask[idx, 1:].clone()
        self.valid_mask[idx, -1] = True
        
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cross-section batch.
        
        Returns:
            x: [1, T, N, F]
            mask: [1, N] - 1 if symbol has full history
        """
        x = self.buffer.unsqueeze(0).permute(0, 2, 1, 3)  # [1, T, N, F]
        mask = self.valid_mask.all(dim=1).unsqueeze(0).float()  # [1, N]
        return x, mask
```

#### 4.2 Modified Walk-Forward Loop

```python
def _cross_section_daily_inference(
    model: CrossSectionMamba,
    buffer: CrossSectionInferenceBuffer,
    adjacency: torch.Tensor,  # [N, N] from SGC
    device: torch.device,
) -> Dict[str, Tuple[float, float]]:
    """Run daily cross-section inference.
    
    Returns: {symbol: (mu, sigma)} for all symbols
    """
    model.eval()
    with torch.no_grad():
        x, mask = buffer.get_batch()  # [1, T, N, F], [1, N]
        mu, logvar = model(x, adjacency=adjacency, mask=mask)
        sigma = torch.sqrt(F.softplus(logvar) + 1e-6)
        
        predictions = {}
        for i, sym in enumerate(buffer.symbols):
            if mask[0, i] > 0.5:  # Valid symbol
                predictions[sym] = (mu[0, i].item(), sigma[0, i].item())
            else:
                predictions[sym] = (0.0, 1.0)  # Default for invalid
        
        return predictions
```

---

## Phase 5: Optuna Integration

### New Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `cs_enabled` | Bool | - | False | Enable cross-section model |
| `cs_universe_size` | Categorical | {64, 128, 256} | 128 | N - symbols per batch |
| `cs_n_temporal_layers` | Categorical | {2, 3, 4} | 3 | Temporal tower depth |
| `cs_n_cross_section_layers` | Categorical | {1, 2, 3} | 2 | Cross-section depth |
| `cs_d_model` | Categorical | {64, 128, 256} | 128 | Model dimension |
| `cs_cross_section_heads` | Categorical | {2, 4, 8} | 4 | Attention heads |
| `cs_use_learned_adjacency` | Bool | - | False | Learn adjacency vs use SGC |
| `cs_adjacency_temperature` | Float | [0.5, 2.0] | 1.0 | Softmax temperature |

---

## Critical Constraints & Risks

### 1. Universe Consistency
- **Decision:** Fixed N per fold + active_mask at runtime
- **Implementation:**
  - Define `UniverseFold` using only info at fold start (e.g., index constituent snapshot)
  - Maintain `active_mask[t, i]` for halted/delisted/missing/failing hygiene gates
  - N stays constant for tensors, but model/portfolio respect tradability

### 2. Leakage-Safe Adjacency
- **Decision:** Use SGC adjacency as v1, add learned residual in v1.5
- **v1:** Pass SGC adjacency directly as attention edge weights
- **v1.5:** Learn residual with SGC as mask:
  ```python
  A_eff = normalize(A_sgc * softplus(W_residual(h)))
  ```

### 3. Cross-Section Timing
- **Decision:** Message passing at T only (final step)
- **Enhancement:** Pool over T-4..T for recent shock propagation:
  ```python
  h_T = mean(h[T-4:T+1])  # or attention pool
  ```

### 4. Training Objective
- **Decision:** Pure NLL + two small regularizers
- **Regularizer 1:** Cross-sectional rank stability (reduce day-to-day flip noise)
- **Regularizer 2:** Sigma floor discipline (prevent collapse/overdispersion)

### 5. Memory Requirements - VRAM Probe Formula
```python
def compute_vram_requirements(
    B: int,        # Batch size
    T: int,        # Sequence length  
    N: int,        # Universe size
    F: int,        # Feature dimension
    d_model: int,  # Model dimension
    n_layers: int, # Number of temporal layers
    reserve_pct: float = 0.10,  # 10% reserve
) -> Dict[str, float]:
    """Compute VRAM requirements in GB with 10% reserve."""
    
    bytes_per_float = 4  # float32
    
    # Input tensor: [B, T, N, F]
    input_bytes = B * T * N * F * bytes_per_float
    
    # Temporal tower activations (per layer, need for backward)
    # Each layer stores: [B*N, T, d_model]
    temporal_act_bytes = n_layers * (B * N * T * d_model * bytes_per_float)
    
    # Cross-section attention: [B, heads, N, N] scores + [B, N, d_model] output
    n_heads = 4
    cross_section_bytes = B * n_heads * N * N * bytes_per_float + B * N * d_model * bytes_per_float
    
    # Gradients (roughly 2x activations)
    gradient_multiplier = 2.0
    
    # Model parameters (estimate)
    param_bytes = (F * d_model + n_layers * 4 * d_model * d_model + N * N) * bytes_per_float
    
    total_bytes = (input_bytes + temporal_act_bytes + cross_section_bytes) * gradient_multiplier + param_bytes
    
    # Add reserve
    total_with_reserve = total_bytes / (1.0 - reserve_pct)
    
    return {
        "input_gb": input_bytes / 1e9,
        "activations_gb": (temporal_act_bytes + cross_section_bytes) / 1e9,
        "total_gb": total_bytes / 1e9,
        "total_with_reserve_gb": total_with_reserve / 1e9,
        "max_batch_for_24gb": int(24 * (1 - reserve_pct) * 1e9 / (total_bytes / B)),
        "max_batch_for_16gb": int(16 * (1 - reserve_pct) * 1e9 / (total_bytes / B)),
    }

# Example for typical config:
# B=32, T=127, N=200, F=200, d_model=128, n_layers=4
# >>> compute_vram_requirements(32, 127, 200, 200, 128, 4)
# {'input_gb': 0.65, 'activations_gb': 5.2, 'total_gb': 11.7, 
#  'total_with_reserve_gb': 13.0, 'max_batch_for_24gb': 44, 'max_batch_for_16gb': 29}
```

### 6. Runtime VRAM Probe
```python
def probe_vram_for_cross_section(
    model: nn.Module,
    B: int, T: int, N: int, F: int,
    device: torch.device,
    reserve_pct: float = 0.10,
) -> Dict[str, Any]:
    """Probe actual VRAM usage with forward+backward pass.
    
    Call this once before training to validate config fits in memory.
    """
    import torch
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Allocate dummy batch
    x = torch.randn(B, T, N, F, device=device, dtype=torch.float32)
    mask = torch.ones(B, N, device=device, dtype=torch.float32)
    adj = torch.eye(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    
    # Forward pass
    with torch.cuda.amp.autocast():
        mu, logvar = model(x, adjacency=adj, mask=mask)
        loss = mu.sum() + logvar.sum()  # Dummy loss
    
    # Backward pass
    loss.backward()
    
    peak_bytes = torch.cuda.max_memory_allocated(device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    available_bytes = total_bytes * (1 - reserve_pct)
    
    # Cleanup
    del x, mask, adj, mu, logvar, loss
    torch.cuda.empty_cache()
    
    fits = peak_bytes < available_bytes
    headroom_pct = (available_bytes - peak_bytes) / total_bytes * 100
    
    return {
        "peak_gb": peak_bytes / 1e9,
        "total_gb": total_bytes / 1e9,
        "available_gb": available_bytes / 1e9,
        "fits": fits,
        "headroom_pct": headroom_pct,
        "max_B_estimate": int(B * available_bytes / peak_bytes) if peak_bytes > 0 else B,
    }
```

---

## Implementation Order

| Step | Component | Effort | Dependency |
|------|-----------|--------|------------|
| 1 | `CrossSectionSequenceData` dataclass | 2h | None |
| 2 | `build_cross_section_data()` | 4h | Step 1 |
| 3 | `CrossSectionAttention` module | 3h | None |
| 4 | `CrossSectionMamba` model | 4h | Step 3 |
| 5 | `train_cross_section_mamba()` | 4h | Steps 2, 4 |
| 6 | Memory optimization (checkpointing) | 2h | Step 5 |
| 7 | `CrossSectionInferenceBuffer` | 2h | None |
| 8 | Phase2 inference integration | 4h | Steps 4, 7 |
| 9 | Optuna parameter registration | 2h | Step 5 |
| 10 | Testing & validation | 4h | All |

**Total:** ~31 hours

---

## Design Decisions (Resolved)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Universe Strategy | Fixed N per fold + `active_mask[t,i]` | Stable tensor shapes; mask handles halts/delists |
| 2 | Adjacency Source | SGC as v1; learn residual in v1.5 | SGC is point-in-time; learned-from-scratch overfits |
| 3 | Cross-Section Timing | T only (with T-4..T pooling option) | Memory; temporal tower already captures dynamics |
| 4 | Training Objective | NLL + rank stability + sigma discipline | Debuggable; regularizers reduce turnover |
| 5 | GPU Memory | 10% reserve; probe before training | Formula-based + runtime validation |
