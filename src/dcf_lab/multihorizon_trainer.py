"""
Main training function for the multi-horizon r    # Set seed for reproducibility
    torch.manual_seed(42)
    # Use modern numpy random generator instead of deprecated global state
    _ = np.random.default_rng(42)  # Create generator for future useprediction model
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .ai_price_forecast import TimeSeriesTransformer
from .direct_forecast import (
    evaluate_model_metrics,
    prepare_log_returns_data,
    prepare_torch_datasets,
)
from .losses import CRPSLoss


def train_multihorizon_forecast_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    horizon: int = 10,
    sequence_length: int = 60,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    model_config: Optional[Dict] = None
) -> Tuple[torch.nn.Module, Dict, Dict]:
    """
    Train a multi-horizon log returns forecasting model with quantile outputs

    Args:
        df: DataFrame with features and prices
        feature_columns: List of feature columns
        horizon: Forecast horizon
        sequence_length: Input sequence length
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        model_config: Optional model configuration

    Returns:
        Tuple of (trained model, training history, validation metrics)
    """
    # Set default model configuration
    if model_config is None:
        model_config = {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 3,
            'dim_feedforward': 128,
            'dropout': 0.3,
            'quantiles': [0.1, 0.5, 0.9]
        }

    # Set seed for reproducibility
    torch.manual_seed(42)
    # Use modern numpy random generator instead of deprecated global state
    np.random.default_rng(42)
    # Also set legacy global seed for compatibility
    np.random.seed(42)

    # Prepare data
    print("Preparing log returns data...")
    _, X, y = prepare_log_returns_data(
        df, feature_columns, horizon, sequence_length
    )

    # Convert to torch datasets
    print("Creating torch datasets...")
    x_train, y_train, x_val, y_val = prepare_torch_datasets(
        X, y, val_split=0.2)

    # Create model
    print("Creating multi-horizon quantile model...")
    model = TimeSeriesTransformer(
        input_dim=len(feature_columns),
        forecast_horizon=horizon,
        quantiles=model_config['quantiles'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_encoder_layers=model_config['num_encoder_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        dropout=model_config['dropout']
    )

    # Set up loss function - use CRPS for quantile predictions
    criterion = CRPSLoss(quantiles=model_config['quantiles'])

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5)

    # Learning rate scheduler for adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("Starting training...")
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('in')
    patience_counter = 0
    early_stopping_patience = 10
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Create batches with random permutation
        permutation = torch.randperm(x_train.size(0))

        for i in range(0, x_train.size(0), batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = epoch_loss / max(1, (x_train.size(0) // batch_size))
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val).item()

        history['val_loss'].append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'Epoch {
                    epoch+1}/{epochs}, Train Loss: {
                    avg_train_loss:.6f}, Val Loss: {
                    val_loss:.6f}')

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Calculate validation metrics
    print("Calculating validation metrics...")
    val_metrics = evaluate_model_metrics(model, x_val, y_val)

    print("Training complete.")
    print(f"Final validation loss: {best_val_loss:.6f}")
    print(f"Direction accuracy: {val_metrics['direction_accuracy']:.4f}")

    return model, history, val_metrics
