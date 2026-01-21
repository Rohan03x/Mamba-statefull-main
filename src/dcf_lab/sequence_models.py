"""Reusable PyTorch sequence model wrappers for adaptive systems.

This module replaces the legacy TensorFlow sequence stack with PyTorch Lightning
implementations and optional Temporal Fusion Transformer support via
``pytorch-forecasting``. The intent is to expose a uniform API that returns
probabilistic forecasts (0-1) for binary direction targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

try:  # Optional TFT support
    from pytorch_forecasting import TimeSeriesDataSet  # type: ignore
    from pytorch_forecasting.models.temporal_fusion_transformer import (  # type: ignore
        TemporalFusionTransformer,
    )
    from pytorch_forecasting.metrics import QuantileLoss  # type: ignore

    _HAS_PF = True
except Exception:  # pragma: no cover - optional dependency
    TimeSeriesDataSet = None  # type: ignore
    TemporalFusionTransformer = None  # type: ignore
    QuantileLoss = None  # type: ignore
    _HAS_PF = False

_LOG = logging.getLogger(__name__)


class SequenceModelError(RuntimeError):
    """Raised when sequence model preparation or training fails."""


class _SeqDataset(Dataset):
    """Tiny helper dataset that always returns (x, y) tensors."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        if y is None:
            y = np.zeros((len(self.X),), dtype=np.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
        return self.X[idx], self.y[idx]


class _LSTMClassifier(pl.LightningModule):
    """Simple LSTM + sigmoid head for binary classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - Lightning forwards
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(self.dropout(last)).squeeze(-1)
        return torch.sigmoid(logits)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage == "train"))
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):  # pragma: no cover - Lightning boilerplate
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)


def _extract_prediction_batches(batches: Sequence) -> np.ndarray:
    """Convert Lightning ``trainer.predict`` batches into a flat probability array."""

    def _first_tensor(node: object) -> Optional[torch.Tensor]:
        if isinstance(node, torch.Tensor):
            return node
        if isinstance(node, dict):
            ordered: Iterable = (
                [node.get("prediction"), node.get("predictions")] + list(node.values())
            )
        elif isinstance(node, (list, tuple)):
            ordered = node
        elif node is None:
            ordered = ()
        else:
            try:
                return torch.as_tensor(node, dtype=torch.float32)
            except Exception:
                ordered = ()
        for item in ordered:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None

    tensors = [t for t in (_first_tensor(batch) for batch in batches) if t is not None]
    if not tensors:
        return np.array([])
    concat = torch.cat([t.reshape(t.shape[0], -1) for t in tensors], dim=0)
    if concat.shape[1] > 1:
        col = min(1, concat.shape[1] - 1)
        concat = concat[:, col]
    else:
        concat = concat.squeeze(-1)
    return torch.clamp(concat, 0.0, 1.0).detach().cpu().numpy()


@dataclass
class SequenceModelConfig:
    model_type: str
    sequence_length: int = 1
    hidden_size: int = 64
    dropout: float = 0.2
    num_layers: int = 1
    max_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    attention_head_size: int = 2  # used by TFT
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)


class SequenceModelWrapper:
    """Uniform interface around PyTorch sequence models (LSTM / TFT)."""

    def __init__(self, config: SequenceModelConfig) -> None:
        self.config = config
        self._trainer: Optional[pl.Trainer] = None
        self._model: Optional[pl.LightningModule] = None
        self._device = "gpu" if torch.cuda.is_available() else "cpu"
        self._group_id = "seq"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the configured model and return validation/test probabilities."""

        if self.config.model_type == "lstm":
            return self._fit_predict_lstm(x_train, y_train, x_val, y_val, x_test)
        if self.config.model_type == "tft":
            try:
                return self._fit_predict_tft(x_train, y_train, x_val, y_val, x_test)
            except Exception as exc:
                _LOG.warning("TFT wrapper fallback to LSTM due to error: %s", exc)
                return self._fit_predict_lstm(x_train, y_train, x_val, y_val, x_test)
        raise SequenceModelError(f"Unsupported sequence model type: {self.config.model_type}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = max(1, int(self.config.sequence_length))
        if seq_len <= 1:
            return X.reshape(len(X), 1, X.shape[1]).astype(np.float32), y.astype(np.float32)
        if len(X) < seq_len:
            raise SequenceModelError(
                f"Not enough samples ({len(X)}) for requested sequence length {seq_len}"
            )
        seq_x, seq_y = [], []
        for idx in range(seq_len - 1, len(X)):
            start = idx - seq_len + 1
            seq_x.append(X[start : idx + 1])
            seq_y.append(y[idx])
        return np.stack(seq_x).astype(np.float32), np.asarray(seq_y, dtype=np.float32)

    def _prepare_trainer(self, enable_progress: bool = False) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=self.config.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=enable_progress,
            accelerator=self._device,
            devices=1,
            deterministic=True,
        )

    def _predict_loader(self, loader: Optional[DataLoader]) -> np.ndarray:
        if loader is None:
            return np.array([])
        batches = self._trainer.predict(self._model, loader)  # type: ignore[arg-type]
        return _extract_prediction_batches(batches)

    # ------------------------------------------------------------------
    # LSTM target
    # ------------------------------------------------------------------
    def _fit_predict_lstm(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_tr_seq, y_tr_seq = self._build_sequences(x_train, y_train)
        x_val_seq, y_val_seq = self._build_sequences(x_val, y_val)
        # For test we do not necessarily need labels, but we pass zeros to match dataset signature
        x_test_seq, _ = self._build_sequences(x_test, np.zeros_like(x_test[:, 0]))

        if x_tr_seq.size == 0:
            raise SequenceModelError("Empty training dataset for sequence model")

        train_loader = DataLoader(
            _SeqDataset(x_tr_seq, y_tr_seq),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        val_loader = None
        if len(x_val_seq) > 0:
            val_loader = DataLoader(
                _SeqDataset(x_val_seq, y_val_seq),
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
        test_loader = DataLoader(
            _SeqDataset(x_test_seq, None),
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        self._model = _LSTMClassifier(
            input_size=x_tr_seq.shape[-1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            lr=self.config.learning_rate,
        )
        self._trainer = self._prepare_trainer()
        self._trainer.fit(self._model, train_loader, val_loader)

        val_probs = self._predict_loader(val_loader)
        test_probs = self._predict_loader(test_loader)
        return val_probs, test_probs

    # ------------------------------------------------------------------
    # TFT target (pytorch-forecasting)
    # ------------------------------------------------------------------
    def _fit_predict_tft(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not _HAS_PF or TimeSeriesDataSet is None or TemporalFusionTransformer is None:
            raise SequenceModelError("pytorch-forecasting is not available")

        feature_cols = [f"feat_{i}" for i in range(x_train.shape[1])]
        quantiles = tuple(float(q) for q in self.config.quantiles)
        encoder_len = max(1, min(self.config.sequence_length, len(x_train)))

        def _frame(X: np.ndarray, y: np.ndarray, start_idx: int) -> "pd.DataFrame":
            data = {col: X[:, idx] for idx, col in enumerate(feature_cols)}
            data["target"] = y.astype(np.float32)
            data["time_idx"] = np.arange(start_idx, start_idx + len(X), dtype=np.int64)
            data["group"] = np.zeros(len(X), dtype=np.int64)
            return pd.DataFrame(data)

        train_df = _frame(x_train, y_train, 0)
        val_df = _frame(x_val, y_val, len(x_train))
        test_df = _frame(x_test, np.zeros(len(x_test), dtype=np.float32), len(x_train) + len(x_val))

        train_dataset = TimeSeriesDataSet(
            data=train_df,
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            max_encoder_length=encoder_len,
            max_prediction_length=1,
            min_encoder_length=max(1, encoder_len // 2),
            min_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=feature_cols,
            time_varying_unknown_reals=["target"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            target_normalizer=None,
            allow_missing_timesteps=True,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=False, stop_randomization=True)
        test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_df, predict=False, stop_randomization=True)

        train_loader = train_dataset.to_dataloader(train=True, batch_size=self.config.batch_size, num_workers=0)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=self.config.batch_size, num_workers=0)
        test_loader = test_dataset.to_dataloader(train=False, batch_size=self.config.batch_size, num_workers=0)

        self._model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_size,
            output_size=len(quantiles),
            loss=QuantileLoss(quantiles=list(quantiles)),
            log_interval=-1,
            reduce_on_plateau_patience=2,
        )
        self._trainer = self._prepare_trainer()
        self._trainer.fit(self._model, train_loader, val_loader)

        val_probs = _extract_prediction_batches(self._trainer.predict(self._model, val_loader))
        test_probs = _extract_prediction_batches(self._trainer.predict(self._model, test_loader))
        return val_probs, test_probs


__all__ = [
    "SequenceModelConfig",
    "SequenceModelError",
    "SequenceModelWrapper",
]
