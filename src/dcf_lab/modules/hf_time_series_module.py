"""Reusable HFTimeSeriesModule.

Implements the canonical high-frequency module requested by Stage A/B pipelines.
The network ingests sliding windows of OHLCV-derived features and predicts the
probability that the 63-day forward log return will be positive.  Outputs are
exposed as:

* ``hf_score`` in ``[-1, 1]`` — directional conviction (0 = flat)
* ``hf_conf`` in ``[0, 1]`` — reliability of the score, tied to the distance
  from the decision boundary

Key design points (see HF design spec):

* Input projection Linea r(D_in → d_model) with LayerNorm
* 2-layer bidirectional GRU (hidden size = 64) with dropout
* Attention pooling over the temporal dimension
* GELU + dropout head that produces a binary logit
* BCEWithLogitsLoss for optimisation, AdamW + cosine schedule

The module is intentionally lightweight so it can be embedded inside different
feature families without re-implementing the sequence stack.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:  # pragma: no cover - torch may be optional in some deployments
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
    from torch.optim import AdamW, Optimizer
    from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                          ReduceLROnPlateau, SequentialLR,
                                          _LRScheduler)
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Tensor = None  # type: ignore
    AdamW = Optimizer = CosineAnnealingLR = LinearLR = SequentialLR = ReduceLROnPlateau = _LRScheduler = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class HFTimeSeriesConfig:
    """Configuration for :class:`HFTimeSeriesModule`.

    Micro-improvements (v2):
      - attn_temperature: Softmax temperature τ ∈ [0.7, 1.2] to prevent attention
        collapse onto single days. Lower τ = sharper, higher τ = smoother.
      - conf_ema_alpha: EMA smoothing for hf_conf to dampen single-day conviction
        spikes. α = 0.0 disables (raw |score|), α = 0.3 recommended.
    """

    input_dim: int
    window: int
    horizon: int = 63
    d_model: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True
    device: Optional[str] = None
    # v2 micro-improvements
    attn_temperature: float = 1.0  # τ ∈ [0.7, 1.2], prevents attention collapse
    conf_ema_alpha: float = 0.3   # EMA smoothing for hf_conf, 0 = disabled

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.window <= 1:
            raise ValueError("window must be > 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if not 0.0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be in [0, 0.5]")
        if not 0.5 <= self.attn_temperature <= 2.0:
            raise ValueError("attn_temperature must be in [0.5, 2.0]")
        if not 0.0 <= self.conf_ema_alpha <= 0.9:
            raise ValueError("conf_ema_alpha must be in [0.0, 0.9]")

    @property
    def encoder_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)


class HFTimeSeriesModule(nn.Module):
    """Canonical GRU + attention encoder used by HF families."""

    def __init__(self, config: HFTimeSeriesConfig) -> None:
        if torch is None or nn is None:  # pragma: no cover - dependency guard
            raise ImportError(
                "PyTorch is required to instantiate HFTimeSeriesModule. "
                "Install torch>=2.0 and retry."
            )
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)

        self.encoder = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        attn_dim = config.encoder_dim
        self.attn_proj = nn.Linear(attn_dim, attn_dim)
        self.attn_vector = nn.Linear(attn_dim, 1, bias=False)

        self.head = nn.Sequential(
            nn.Linear(attn_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # v2 micro-improvements: attention temperature and confidence EMA state
        self.attn_temperature = config.attn_temperature
        self.conf_ema_alpha = config.conf_ema_alpha
        # Running EMA state for confidence calibration (batch-level, reset per forward)
        self._conf_ema_state: Optional[Tensor] = None

        self.to(config.device or self._auto_device())

    @staticmethod
    def _auto_device() -> str:
        if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda"
        return "cpu"

    def forward(
        self,
        inputs: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run the sequence encoder.

        Args:
            inputs: Tensor shaped (batch, window, input_dim)
            attention_mask: Optional tensor shaped (batch, window) where 1
                marks valid timesteps and 0 masks padding.
        Returns:
            Dict with logits, probabilities, hf_score, hf_conf, attention weights,
            and pooled representation.
        """

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, window, input_dim)")
        if inputs.size(1) != self.config.window:
            raise ValueError(
                f"expected window={self.config.window}, got {inputs.size(1)}"
            )

        x = self.input_proj(inputs)
        x = self.input_norm(x)
        x = F.dropout(x, p=self.config.dropout, training=self.training)

        enc_out, _ = self.encoder(x)
        attn_hidden = torch.tanh(self.attn_proj(enc_out))
        scores = self.attn_vector(attn_hidden).squeeze(-1)

        if attention_mask is not None:
            if attention_mask.shape != scores.shape:
                raise ValueError("attention_mask must match (batch, window)")
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # (A) Attention temperature scaling: softmax(e_t / τ)
        # τ < 1.0 = sharper attention, τ > 1.0 = smoother distribution
        # Prevents collapse onto single day, improves robustness in volatile regimes
        scaled_scores = scores / self.attn_temperature
        weights = torch.softmax(scaled_scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(weights * enc_out, dim=1)

        logits = self.head(pooled).squeeze(-1)
        prob_up = torch.sigmoid(logits)
        hf_score = (prob_up * 2.0) - 1.0

        # (B) Confidence calibration via EMA: dampens single-day conviction spikes
        # hf_conf_t = α × hf_conf_{t-1} + (1 - α) × |hf_score_t|
        raw_conf = torch.abs(hf_score)
        if self.conf_ema_alpha > 0.0 and not self.training:
            # Only apply EMA during inference (training uses raw for gradient flow)
            if self._conf_ema_state is None or self._conf_ema_state.shape != raw_conf.shape:
                self._conf_ema_state = raw_conf.clone()
            else:
                self._conf_ema_state = (
                    self.conf_ema_alpha * self._conf_ema_state +
                    (1.0 - self.conf_ema_alpha) * raw_conf
                )
            hf_conf = self._conf_ema_state.clone()
        else:
            hf_conf = raw_conf

        return {
            "logits": logits,
            "p_up": prob_up,
            "hf_score": hf_score,
            "hf_conf": hf_conf,
            "hf_conf_raw": raw_conf,  # Unsmoothed for diagnostics
            "attention": weights.squeeze(-1),
            "pooled_state": pooled,
            "attn_entropy": self._attention_entropy(weights.squeeze(-1)),  # Diagnostic
        }

    @staticmethod
    def _attention_entropy(weights: Tensor) -> Tensor:
        """Compute attention entropy for diagnostic purposes.

        Higher entropy = more uniform attention (good).
        Low entropy = attention collapsed onto few days (potential overfit).
        Max entropy for window=40 is log(40) ≈ 3.69.
        """
        # Avoid log(0) with small epsilon
        log_weights = torch.log(weights + 1e-10)
        entropy = -torch.sum(weights * log_weights, dim=1)
        return entropy

    def reset_ema_state(self) -> None:
        """Reset confidence EMA state (call between symbols or time series)."""
        self._conf_ema_state = None

    @staticmethod
    def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
        """Binary cross entropy with logits."""
        targets = targets.float().view_as(logits)
        return F.binary_cross_entropy_with_logits(logits, targets)

    @torch.no_grad()
    def infer(
        self, inputs: Tensor, *, attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Convenience wrapper returning (hf_score, hf_conf)."""
        self.eval()
        outputs = self.forward(inputs, attention_mask=attention_mask)
        return outputs["hf_score"], outputs["hf_conf"]


def configure_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Optimizer:
    """AdamW per canonical spec."""
    if AdamW is None:  # pragma: no cover
        raise ImportError("PyTorch optimizers not available")
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def configure_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float = 0.1,
    mode: str = "cosine",
) -> _LRScheduler:
    """Create cosine or plateau scheduler with optional warmup."""
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if mode == "cosine":
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    if mode == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=False)

    raise ValueError("mode must be 'cosine' or 'plateau'")


def clip_gradients(model: nn.Module, max_norm: float = 2.0) -> None:
    """Safely clip gradients."""
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch not available")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def training_step(
    model: HFTimeSeriesModule,
    batch_inputs: Tensor,
    batch_targets: Tensor,
    *,
    attention_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Single optimisation step helper."""
    outputs = model(batch_inputs, attention_mask=attention_mask)
    loss = model.compute_loss(outputs["logits"], batch_targets)
    return {
        "loss": loss,
        "hf_score": outputs["hf_score"].detach(),
        "hf_conf": outputs["hf_conf"].detach(),
        "prob_up": outputs["p_up"].detach(),
    }


def prepare_supervised_targets(prices: Tensor, horizon: int) -> Tensor:
    """Generate binary targets: forward log return over ``horizon`` > 0."""
    if prices.ndim != 2:
        raise ValueError("prices must have shape (batch, time)")
    forward = prices[:, horizon:] / prices[:, :-horizon]
    log_ret = torch.log(forward + 1e-12)
    target = (log_ret > 0).float()
    # Align to leading portion (drop last horizon samples)
    pad = torch.zeros(target.shape[0], horizon, device=target.device)
    return torch.cat([target, pad], dim=1)


if __name__ == "__main__":  # pragma: no cover - quick smoke test
    if torch is None:
        raise SystemExit("torch is required for the HFTimeSeriesModule smoke test")

    cfg = HFTimeSeriesConfig(input_dim=32, window=40)
    model = HFTimeSeriesModule(cfg)
    dummy = torch.randn(8, cfg.window, cfg.input_dim)
    target = torch.randint(0, 2, (8,), dtype=torch.float32)

    out = model(dummy)
    print("logits shape", out["logits"].shape)
    print("hf_score range", out["hf_score"].min().item(), out["hf_score"].max().item())

    loss = HFTimeSeriesModule.compute_loss(out["logits"], target)
    print("loss", float(loss))
