"""
PyTorch-based forecasters: a lightweight LSTM quantile forecaster and an optional TFT wrapper.

This integrates with the existing QuantileForecaster contract from forecast_models.py
without introducing new dependencies beyond torch and pytorch-lightning (already present).
If pytorch_forecasting is available, TFTForecaster can be used; otherwise use LSTMQuantileForecaster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
try:
    from lightning.pytorch import Trainer as LTrainer  # modern Lightning
    _HAS_LTRAINER = True
except Exception:  # pragma: no cover
    LTrainer = None  # type: ignore
    _HAS_LTRAINER = False

try:
    # Core classes from pytorch_forecasting
    from pytorch_forecasting import TimeSeriesDataSet  # type: ignore
    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer  # type: ignore
    from pytorch_forecasting.metrics import QuantileLoss  # type: ignore
    _HAS_PF = True
except Exception:
    TimeSeriesDataSet = None  # type: ignore
    TemporalFusionTransformer = None  # type: ignore
    QuantileLoss = None  # type: ignore
    _HAS_PF = False

# Import base classes
from .forecast_models import QuantileForecaster, ForecastConfig, MODEL_NOT_FITTED


def _pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    # y_pred: [B, Q], y_true: [B], quantiles: [Q]
    y_true = y_true.unsqueeze(-1).expand_as(y_pred)
    err = y_true - y_pred
    q = quantiles.view(1, -1).to(y_pred.device)
    loss = torch.max(q * err, (q - 1) * err)
    return loss.mean()


class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _LSTMModule(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, quantiles: List[float], lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=[quantiles])
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.lr = lr
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        q = self.head(last)
        return q

    def training_step(self, batch, batch_idx):
        x, y = batch
        q_pred = self(x)
        loss = _pinball_loss(q_pred, y, self.quantiles)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        q_pred = self(x)
        loss = _pinball_loss(q_pred, y, self.quantiles)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)


@dataclass
class LSTMConfig:
    lookback: int = 30
    hidden_size: int = 64
    batch_size: int = 64
    max_epochs: int = 5
    lr: float = 1e-3


class LSTMQuantileForecaster(QuantileForecaster):
    """
    Simple LSTM forecaster that predicts multiple quantiles with pinball loss.
    Expects sequential features shaped as [time, features]. Helper prepare_sequences is provided below.
    """

    def __init__(self, config: Optional[ForecastConfig] = None, lstm: Optional[LSTMConfig] = None):
        super().__init__(config)
        self.lcfg = lstm or LSTMConfig()
        self.model: Optional[_LSTMModule] = None
        self.fitted_scalers: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @staticmethod
    def prepare_sequences(series: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        vals = series.values.astype(np.float32)
        X, y = [], []
        for i in range(len(vals) - lookback):
            X.append(vals[i:i+lookback][:, None])
            y.append(vals[i+lookback])
        return np.stack(X), np.array(y)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMQuantileForecaster':
        # Expect X to include at least one column; use the first as the target series base
        series = X.iloc[:, 0]
        lookback = self.lcfg.lookback
        x_arr, y_arr = self.prepare_sequences(series, lookback)

        # standardize
        mu, sigma = x_arr.mean(), x_arr.std() + 1e-8
        x_arr = (x_arr - mu) / sigma
        y_arr = (y_arr - mu) / sigma
        self.fitted_scalers = (float(mu), float(sigma))

        # split
        n = len(x_arr)
        split = max(int(n * 0.8), 1)
        train_ds = _SeqDataset(x_arr[:split], y_arr[:split])
        val_ds = _SeqDataset(x_arr[split:], y_arr[split:])
        # Windows-safe DataLoaders (num_workers=0). Use pin_memory when CUDA is available for speed.
        use_cuda = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.lcfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.lcfg.batch_size,
            num_workers=0,
            pin_memory=use_cuda,
        )

        self.model = _LSTMModule(
            input_size=x_arr.shape[-1],
            hidden_size=self.lcfg.hidden_size,
            quantiles=self.config.quantiles,
            lr=self.lcfg.lr,
        )
        # Prefer GPU if available; Lightning will handle device transfer.
        trainer = pl.Trainer(
            max_epochs=self.lcfg.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator=("gpu" if use_cuda else "cpu"),
            devices=1,
            deterministic=True,
        )
        trainer.fit(self.model, train_loader, val_loader if len(val_ds) > 0 else None)
        self.is_fitted = True
        return self

    def predict_quantiles(self, X: pd.DataFrame, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        if not self.is_fitted or self.model is None or self.fitted_scalers is None:
            raise ValueError(MODEL_NOT_FITTED)
        q_list = quantiles or self.config.quantiles
        series = X.iloc[:, 0]
        lookback = self.lcfg.lookback
        x_arr, _ = self.prepare_sequences(series, lookback)
        mu, sigma = self.fitted_scalers
        x_arr = (x_arr - mu) / sigma
        ds = _SeqDataset(x_arr, np.zeros((len(x_arr),), dtype=np.float32))
        use_cuda = torch.cuda.is_available()
        dl = DataLoader(ds, batch_size=self.lcfg.batch_size, num_workers=0, pin_memory=use_cuda)
        self.model.eval()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(device)
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(device, non_blocking=use_cuda)
                q_pred = self.model(xb)
                preds.append(q_pred.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        # denormalize
        preds = preds * sigma + mu
        trained_cols = [f"q{int(q*100)}" for q in self.config.quantiles]
        pred_df = pd.DataFrame(preds, columns=trained_cols)
        cols = [f"q{int(q*100)}" for q in q_list]
        out = pred_df[cols] if all(c in pred_df.columns for c in cols) else pred_df
        out.index = X.index[lookback:]
        return out


@dataclass
class TFTConfig:
    lookback: int = 60
    prediction_length: int = 1
    hidden_size: int = 32
    batch_size: int = 64
    max_epochs: int = 3
    lr: float = 1e-3


class TFTForecaster(QuantileForecaster):
    """Wrapper around pytorch_forecasting TemporalFusionTransformer."""
    def __init__(self, config: Optional[ForecastConfig] = None, tft: Optional[TFTConfig] = None):
        super().__init__(config)
        if not _HAS_PF:
            raise ImportError("pytorch_forecasting not available; use LSTMQuantileForecaster instead.")
        self.tcfg = tft or TFTConfig()
        self._training_ds = None
        self._model = None
        self._symbol = None
        self._index = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TFTForecaster':
        # Use first column as the target series; align index to pandas Date index for reporting
        series = X.iloc[:, 0].dropna()
        self._index = series.index
        self._symbol = getattr(X, 'attrs', {}).get('symbol', None) or 'SYMBOL'
        df = pd.DataFrame({
            'time_idx': np.arange(len(series), dtype=np.int64),
            'symbol': self._symbol,
            'target': series.values.astype(np.float32)
        })

        max_encoder_length = int(self.tcfg.lookback)
        max_prediction_length = int(self.tcfg.prediction_length)

        # Training dataset excludes last prediction horizon implicitly via dataset split
        training_cutoff = df['time_idx'].max() - max_prediction_length
        training = TimeSeriesDataSet(
            df[df.time_idx <= training_cutoff],
            time_idx='time_idx',
            target='target',
            group_ids=['symbol'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=['target'],
        )
        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

        train_loader = training.to_dataloader(train=True, batch_size=self.tcfg.batch_size, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=self.tcfg.batch_size, num_workers=0)

        qloss = QuantileLoss(quantiles=self.config.quantiles)
        self._model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.tcfg.lr,
            hidden_size=self.tcfg.hidden_size,
            loss=qloss,
            dropout=0.1,
            output_size=len(self.config.quantiles),
            attention_head_size=1,
        )
        trainer_cls = LTrainer if _HAS_LTRAINER else pl.Trainer
        use_cuda = torch.cuda.is_available()
        trainer = trainer_cls(
            max_epochs=self.tcfg.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator=("gpu" if use_cuda else "cpu"),
            devices=1,
            deterministic=True,
        )
        trainer.fit(self._model, train_loader, val_loader)
        self._training_ds = training
        self.is_fitted = True
        return self

    def predict_quantiles(self, X: pd.DataFrame, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        if not self.is_fitted or self._model is None or self._training_ds is None:
            raise ValueError(MODEL_NOT_FITTED)
        q_list = quantiles or self.config.quantiles
        # Rebuild df matching fit
        series = X.iloc[:, 0].dropna()
        df = pd.DataFrame({
            'time_idx': np.arange(len(series), dtype=np.int64),
            'symbol': self._symbol or 'SYMBOL',
            'target': series.values.astype(np.float32)
        })
        pred_ds = TimeSeriesDataSet.from_dataset(self._training_ds, df, predict=True, stop_randomization=True)
        pred_loader = pred_ds.to_dataloader(train=False, batch_size=self.tcfg.batch_size, num_workers=0)
        preds = self._model.predict(pred_loader)
        # preds: [N, prediction_length] or [N, prediction_length, Q] depending on config; request quantiles via loss
        # TemporalFusionTransformer.predict returns point forecasts unless loss is QuantileLoss (then quantiles in last dim)
        arr = preds.cpu().numpy()
        # If shape is (N, pl, Q) reduce prediction_length dimension by taking last step
        if arr.ndim == 3:
            arr = arr[:, -1, :]  # last step
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = np.repeat(arr, len(q_list), axis=1)
        columns = [f"q{int(q*100)}" for q in q_list]
        out = pd.DataFrame(arr, columns=columns)
        # Align index to X (skip encoder-length warmup)
        lookback = self.tcfg.lookback
        out.index = X.index[lookback:lookback + len(out)] if hasattr(X, 'index') else pd.RangeIndex(len(out))
        return out


__all__ = [
    'LSTMQuantileForecaster', 'LSTMConfig', 'TFTForecaster', 'TFTConfig'
]
