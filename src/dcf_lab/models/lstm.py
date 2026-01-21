"""
LSTM model implementation using PyTorch Lightning.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class PriceLSTM(pl.LightningModule):
    """LSTM model for time series forecasting with PyTorch Lightning."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        loss: str = "mse",
        bidirectional: bool = False,
        num_outputs: int = 1
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            loss: Loss function ('mse' or 'huber')
            bidirectional: Whether to use bidirectional LSTM
            num_outputs: Number of outputs (horizon length)
        """
        super().__init__()
        self.save_hyperparameters()

        # Model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.num_outputs = num_outputs

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output layer
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, num_outputs)
        )

        # Loss function
        self.loss_fn = nn.MSELoss() if loss == "mse" else nn.HuberLoss()

        # Metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use last output
        last_out = lstm_out[:, -1, :]

        # Linear layer
        output = self.fc(last_out)

        return output

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Tuple of (X, y)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.training_step_outputs.append(loss)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Tuple of (X, y)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.validation_step_outputs.append(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        """Calculate and log epoch-level training metrics."""
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Calculate and log epoch-level validation metrics."""
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class LSTMForecaster:
    """Wrapper class for LSTM model with data preprocessing."""

    def __init__(
        self,
        model_params: Dict,
        batch_size: int = 32,
        max_epochs: int = 100,
        early_stopping_patience: int = 20,
        device: str = "auto"
    ):
        """
        Initialize LSTM forecaster.

        Args:
            model_params: Dictionary of model parameters
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            device: Device to use for training ("auto", "cpu", or "cuda")
        """
        self.model_params = model_params
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.trainer = None
        self.scaler = None

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create PyTorch DataLoader.

        Args:
            X: Input features
            y: Target values
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader
        """
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        # Create dataset
        dataset = TensorDataset(X, y)

        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        # Create model
        self.model = PriceLSTM(
            input_size=X_train.shape[2],
            num_outputs=y_train.shape[1],
            **self.model_params
        )

        # Create dataloaders
        train_loader = self._create_dataloader(X_train, y_train)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience
                )
            ],
            enable_progress_bar=True
        )

        # Train model
        self.trainer.fit(
            self.model,
            train_loader,
            val_loader
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fit before predicting")

        # Create dataloader
        loader = self._create_dataloader(
            X, np.zeros((len(X), 1)), shuffle=False)

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                y_hat = self.model(x)
                predictions.append(y_hat.cpu().numpy())

        return np.vstack(predictions)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and parameters
        torch.save({
            "model_state": self.model.state_dict(),
            "model_params": self.model_params,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LSTMForecaster":
        """
        Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path)

        # Create instance
        instance = cls(
            model_params=checkpoint["model_params"],
            batch_size=checkpoint["batch_size"],
            max_epochs=checkpoint["max_epochs"],
            early_stopping_patience=checkpoint["early_stopping_patience"],
            device=checkpoint["device"]
        )

        # Create model and load state
        instance.model = PriceLSTM(**checkpoint["model_params"])
        instance.model.load_state_dict(checkpoint["model_state"])

        return instance
