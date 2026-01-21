"""
Implementation of quantile loss (pinball loss) for PyTorch.

This module contains implementations for quantile loss functions
used in the direct multi-horizon forecasting models.
"""

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    Quantile loss, also known as pinball loss.

    This loss is used for quantile regression, where we want to predict
    a range of quantiles (e.g., 10%, 50%, 90%) for uncertainty estimation.

    Args:
        quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
    """

    def __init__(self, quantiles=None):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [
            0.1, 0.5, 0.9]

    def forward(self, preds, target):
        """
        Calculate the quantile loss.

        Args:
            preds: Tensor of shape (batch_size, num_quantiles, horizon)
                  or (batch_size, horizon) if single quantile
            target: Tensor of shape (batch_size, horizon)

        Returns:
            Quantile loss value
        """
        if len(preds.shape) == 2:
            # Single quantile case (assuming median)
            return self._quantile_loss(preds, target, 0.5)

        losses = []
        for i, q in enumerate(self.quantiles):
            q_pred = preds[:, i]
            losses.append(self._quantile_loss(q_pred, target, q))

        # Return the mean of all quantile losses
        return torch.mean(torch.stack(losses))

    def _quantile_loss(self, pred, target, quantile):
        """Calculate the quantile loss for a single quantile"""
        error = target - pred
        return torch.mean(torch.max(quantile * error, (quantile - 1) * error))


class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) loss.

    CRPS is a proper scoring rule for evaluating probabilistic forecasts.
    This implementation uses a discretized approximation based on multiple quantiles.

    Args:
        quantiles: List of quantiles to predict
    """

    def __init__(self, quantiles=None):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [
            0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        self.quantile_loss = QuantileLoss(quantiles=quantiles)

    def forward(self, preds, target):
        """
        Calculate the CRPS loss based on multiple quantile predictions.

        Args:
            preds: Tensor of shape (batch_size, num_quantiles, horizon)
            target: Tensor of shape (batch_size, horizon)

        Returns:
            CRPS loss value
        """
        # Use the average of all quantile losses as an approximation to CRPS
        return self.quantile_loss(preds, target)
