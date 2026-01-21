"""
Options-anchored Learning Loop Framework

This module implements a sophisticated options-anchored learning system that:

1. Computes implied-move baseline from near-ATM straddle/IV (1σ neutral prior)
2. Blends options prior with technical/macro/news features via small learner
3. Targets realized move/return for prediction
4. Implements IV extremeness gating to up-weight options expert
5. Integrates with ensemble stacking for end-to-end learning

The framework leverages market-implied expectations as a strong baseline
and learns to combine them optimally with other information sources.

Components:
- implied_moves.py: Computing implied volatility and moves from options
- options_prior.py: Creating neutral priors from options market data
- blending_learner.py: Small learner for combining priors with features
- iv_gating.py: IV extremeness detection and gating mechanisms
- options_ensemble.py: Integration with ensemble framework
"""

# Core options-anchored learning components
from .implied_moves import (
    ImpliedVolatilityCalculator,
    ATMStraddle,
    ImpliedMoveComputer,
    OptionsDataProcessor
)

from .options_prior import (
    OptionsPrior,
    NeutralPriorBuilder,
    ImpliedDistribution,
    PriorCalibrator
)

from .blending_learner import (
    OptionsBlendingLearner,
    FeatureBlender,
    PriorWeightingNetwork,
    BlendingOptimizer
)

from .iv_gating import (
    IVExtremenessDetector,
    AdaptiveGatingNetwork,
    OptionsExpertGate,
    ExtremenessMetrics,
    GatingDecision
)

# Ensemble integration
from .options_ensemble import (
    OptionsExpert,
    OptionsExpertConfig,
    OptionsExpertPrediction,
    OptionsAnchoredEnsemble,
    EndToEndOptionsLearner,
    OptionsIntegration
)


# Configuration and utilities
from .config import OptionsConfig, BlendingConfig, GatingConfig
from .utils import (
    validate_options_data,
    compute_options_features,
    calculate_iv_percentiles,
    format_options_results
)

__all__ = [
    # Implied moves and volatility
    'ImpliedVolatilityCalculator',
    'ATMStraddle', 
    'ImpliedMoveComputer',
    'OptionsDataProcessor',
    
    # Options priors
    'OptionsPrior',
    'NeutralPriorBuilder',
    'ImpliedDistribution',
    'PriorCalibrator',
    
    # Blending learner
    'OptionsBlendingLearner',
    'FeatureBlender',
    'PriorWeightingNetwork',
    'BlendingOptimizer',
    
    # IV gating
    'IVExtremenessDetector',
    'AdaptiveGatingNetwork',
    'OptionsExpertGate',
    'ExtremenessMetrics',
    
    # Ensemble integration
    'OptionsAnchoredEnsemble',
    'OptionsExpert',
    'EndToEndOptionsLearner',
    'OptionsIntegration',
    
    # Configuration
    'OptionsConfig',
    'BlendingConfig',
    'GatingConfig',
    
    # Utilities
    'validate_options_data',
    'compute_options_features',
    'calculate_iv_percentiles',
    'format_options_results'
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Financial ML Framework"
__description__ = "Options-anchored learning loop for financial ML"

print("📊 Options-anchored Learning Loop Framework Initialized")
print("   ✓ Implied volatility and move computation")
print("   ✓ Options-based neutral priors")
print("   ✓ Feature blending with small learners")
print("   ✓ IV extremeness gating mechanisms")
print("   ✓ End-to-end ensemble integration")