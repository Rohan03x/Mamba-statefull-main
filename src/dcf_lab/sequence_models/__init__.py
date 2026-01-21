"""
Multi-horizon Sequence Model Training Framework

This module implements Task 8: "Multi-horizon sequence model training (TFT/Transformer)"
with comprehensive support for:

1. Supervised sliding windows (60-120 days) → multiple horizons (1/5/20/60 days)
2. TFT best practices: static, known-future (calendars, expiries), observed-past (prices, vol, news)
3. Curriculum learning: start shorter horizons → extend; freeze embeddings when stable
4. Interpretability checks: attention/variable importance validation
5. Robust, interpretable multi-horizon models complementing tree & AR families

Key components:
- tft_model.py: Temporal Fusion Transformer implementation
- data_preparation.py: Feature splitting and window creation
- curriculum_learning.py: Progressive horizon training
- interpretability.py: Attention analysis and variable importance
- multi_horizon_trainer.py: Complete training pipeline
"""

# Core sequence model components
from .data_preparation import (
    SequenceDataProcessor,
    FeatureSplitter,
    WindowGenerator,
    FeatureConfig,
    WindowConfig,
    MultiHorizonTargets
)

from .tft_model import (
    TemporalFusionTransformer,
    TFTConfig,
    TFTPredictor,
    TFTTrainer,
    VariableSelectionNetwork,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention
)

from .curriculum_learning import (
    CurriculumScheduler,
    CurriculumConfig,
    HorizonProgression,
    EmbeddingFreezer,
    StabilityMonitor
)

from .interpretability import (
    ModelExplainer,
    InterpretabilityConfig,
    AttentionAnalyzer,
    VariableImportanceScorer,
    InterpretabilityChecker
)

from .multi_horizon_trainer import (
    MultiHorizonTrainer,
    SequenceModelConfig,
    TrainingResults,
    ModelEvaluator
)

__all__ = [
    # Data preparation
    'SequenceDataProcessor',
    'FeatureSplitter', 
    'WindowGenerator',
    'FeatureConfig',
    'WindowConfig',
    'MultiHorizonTargets',
    
    # TFT model
    'TemporalFusionTransformer',
    'TFTConfig',
    'TFTPredictor',
    'TFTTrainer',
    'VariableSelectionNetwork',
    'GatedResidualNetwork',
    'InterpretableMultiHeadAttention',
    
    # Curriculum learning
    'CurriculumScheduler',
    'CurriculumConfig',
    'HorizonProgression',
    'EmbeddingFreezer',
    'StabilityMonitor',
    
    # Interpretability
    'ModelExplainer',
    'InterpretabilityConfig',
    'AttentionAnalyzer',
    'VariableImportanceScorer',
    'InterpretabilityChecker',
    
    # Training framework
    'MultiHorizonTrainer',
    'SequenceModelConfig',
    'TrainingResults',
    'ModelEvaluator'
]