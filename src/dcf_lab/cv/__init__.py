"""
Cross-validation module for financial time series.

This module provides leakage-safe cross-validation utilities specifically
designed for financial machine learning applications.
"""

from .time_series_cv import (
    CVConfig,
    PurgedTimeSeriesSplit,
    WalkForwardValidator,
    ConformalPredictor,
    CalibrationWrapper,
    create_target_variable
)

from .enhanced_cv import (
    EnhancedCVConfig,
    RigorousTimeSeriesSplit,
    PerFoldStandardizer,
    MissingDataHandler,
    create_enhanced_target_variable
)

from .walk_forward_cv import (
    WalkForwardConfig,
    WalkForwardSplit,
    create_walk_forward_splitter
)

__all__ = [
    # Original CV utilities
    'CVConfig',
    'PurgedTimeSeriesSplit',
    'WalkForwardValidator',
    'ConformalPredictor',
    'CalibrationWrapper',
    'create_target_variable',
    
    # Enhanced CV with rigorous leakage protection
    'EnhancedCVConfig',
    'RigorousTimeSeriesSplit', 
    'PerFoldStandardizer',
    'MissingDataHandler',
    'create_enhanced_target_variable',
    
    # Walk-forward CV with purged embargo (López de Prado)
    'WalkForwardConfig',
    'WalkForwardSplit',
    'create_walk_forward_splitter'
]