"""
Ensemble Learning Module for Financial ML

Provides sophisticated ensemble methods with regime awareness and proper leakage prevention:
- Base learner training with OOF predictions
- Meta-learner stacking
- Regime detection with HMM
- Options-anchored expert
- Intelligent gating networks
"""

from .ensemble_framework import (
    EnsembleConfig,
    BaseModelTrainer,
    RegimeDetector,
    OptionsExpert
)

from .meta_learner import (
    MetaLearner,
    GatingNetwork,
    EnsembleStacker
)

__all__ = [
    # Configuration
    'EnsembleConfig',
    
    # Core ensemble components
    'BaseModelTrainer',
    'RegimeDetector', 
    'OptionsExpert',
    'MetaLearner',
    'GatingNetwork',
    
    # Main ensemble class
    'EnsembleStacker'
]