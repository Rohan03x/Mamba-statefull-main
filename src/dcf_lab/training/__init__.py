"""
Training module for financial ML models.

Provides comprehensive training, evaluation, and validation pipelines
specifically designed for financial time series forecasting.
"""

from .baseline_trainer import (
    TrainingConfig,
    ModelResult,
    MetricsCalculator,
    CalibrationAnalyzer,
    ModelFactory,
    BaselineTrainer
)

__all__ = [
    'TrainingConfig',
    'ModelResult',
    'MetricsCalculator',
    'CalibrationAnalyzer',
    'ModelFactory',
    'BaselineTrainer'
]