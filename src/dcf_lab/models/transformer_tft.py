"""
Temporal Fusion Transformer implementation using pytorch-forecasting.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

logger = logging.getLogger(__name__)


class TFTModel:
    """Wrapper for Temporal Fusion Transformer model."""

    def __init__(
        self,
        hidden_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        early_stopping_patience: int = 20,
        gpus: int = 1 if torch.cuda.is_available() else 0,
        gradient_clip_val: float = 0.1
    ):
        """
        Initialize TFT model.

        Args:
            hidden_size: Hidden size of network
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for processing continuous variables
            learning_rate: Learning rate
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            gpus: Number of GPUs to use
            gradient_clip_val: Gradient clipping threshold
        """
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gpus = gpus
        self.gradient_clip_val = gradient_clip_val

        self.model = None
        self.training = None
        self.validation = None

    def _create_datasets(
        self,
        df: pd.DataFrame,
        target: str,
        time_idx: str,
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_known_categoricals: List[str],
        time_varying_known_reals: List[str],
        time_varying_unknown_categoricals: List[str],
        time_varying_unknown_reals: List[str],
        group_ids: List[str],
        max_encoder_length: int,
        max_prediction_length: int,
        min_encoder_length: Optional[int] = None,
        min_prediction_length: Optional[int] = None,
        train_val_split: float = 0.2
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Create training and validation datasets.

        Args:
            df: DataFrame with data
            target: Target variable
            time_idx: Time index column
            static_categoricals: Static categorical variables
            static_reals: Static real variables
            time_varying_known_categoricals: Time-varying known categorical variables
            time_varying_known_reals: Time-varying known real variables
            time_varying_unknown_categoricals: Time-varying unknown categorical variables
            time_varying_unknown_reals: Time-varying unknown real variables
            group_ids: Group IDs for time series
            max_encoder_length: Maximum encoder length
            max_prediction_length: Maximum prediction length
            min_encoder_length: Minimum encoder length
            min_prediction_length: Minimum prediction length
            train_val_split: Train-validation split ratio

        Returns:
            Training and validation datasets
        """
        # Create training dataset
        training = TimeSeriesDataSet(
            df
            [lambda x: x.time_idx <= x.time_idx.max() * (1 - train_val_split)],
            time_idx=time_idx, target=target, group_ids=group_ids,
            static_categoricals=static_categoricals, static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_encoder_length=min_encoder_length,
            min_prediction_length=min_prediction_length,
            target_normalizer=GroupNormalizer(
                groups=group_ids, transformation="softplus"),
            add_relative_time_idx=True, add_target_scales=True,
            add_encoder_length=True)

        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df,
            min_prediction_idx=training.index.time.max() + 1,
            stop_randomization=True
        )

        return training, validation

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        time_idx: str,
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_known_categoricals: List[str],
        time_varying_known_reals: List[str],
        time_varying_unknown_categoricals: List[str],
        time_varying_unknown_reals: List[str],
        group_ids: List[str],
        max_encoder_length: int,
        max_prediction_length: int,
        min_encoder_length: Optional[int] = None,
        min_prediction_length: Optional[int] = None,
        train_val_split: float = 0.2
    ) -> None:
        """
        Fit TFT model.

        Args:
            See _create_datasets for parameter descriptions
        """
        # Create datasets
        self.training, self.validation = self._create_datasets(
            df=df, target=target, time_idx=time_idx,
            static_categoricals=static_categoricals, static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            group_ids=group_ids, max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_encoder_length=min_encoder_length,
            min_prediction_length=min_prediction_length,
            train_val_split=train_val_split)

        # Create dataloaders
        train_dataloader = self.training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=0
        )
        val_dataloader = self.validation.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=0
        )

        # Create model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.gpus > 0 else "cpu",
            devices=self.gpus,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience
                )
            ]
        )

        # Train model
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

    def predict(
        self,
        df: pd.DataFrame,
        return_quantiles: bool = True,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate predictions.

        Args:
            df: DataFrame with features
            return_quantiles: Whether to return prediction quantiles
            quantiles: Quantiles to return

        Returns:
            If return_quantiles=False: Array of predictions
            If return_quantiles=True: Dictionary with quantile predictions
        """
        if self.model is None:
            raise ValueError("Model must be fit before predicting")

        # Create dataloader
        dataloader = self.validation.to_dataloader(
            df,
            train=False,
            batch_size=self.batch_size,
            num_workers=0
        )

        # Make predictions
        predictions = self.model.predict(
            dataloader,
            mode="prediction",
            return_x=False,
            return_index=False,
            return_quantiles=return_quantiles,
            quantiles=quantiles
        )

        if return_quantiles:
            # Convert quantile predictions to dictionary
            result = {}
            for i, q in enumerate(quantiles):
                result[f"q{int(q*100)}"] = predictions[:, :, i]
            return result

        return predictions.numpy()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state": self.model.state_dict(),
            "training": self.training,
            "validation": self.validation,
            "params": {
                "hidden_size": self.hidden_size,
                "attention_head_size": self.attention_head_size,
                "dropout": self.dropout,
                "hidden_continuous_size": self.hidden_continuous_size,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "gpus": self.gpus,
                "gradient_clip_val": self.gradient_clip_val
            }
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TFTModel":
        """
        Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path)

        # Create instance
        instance = cls(**checkpoint["params"])
        instance.training = checkpoint["training"]
        instance.validation = checkpoint["validation"]

        # Create and load model
        instance.model = TemporalFusionTransformer.from_dataset(
            instance.training,
            learning_rate=instance.learning_rate,
            hidden_size=instance.hidden_size,
            attention_head_size=instance.attention_head_size,
            dropout=instance.dropout,
            hidden_continuous_size=instance.hidden_continuous_size
        )
        instance.model.load_state_dict(checkpoint["model_state"])

        return instance
