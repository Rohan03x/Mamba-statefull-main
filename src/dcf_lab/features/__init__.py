"""
Features module for DCF Lab

Provides feature engineering and validation functionality for financial ML:
- Feature cleaning and leakage prevention
- Automated feature validation
- Lag-based feature creation
"""

from .feature_cleaner import (
    FeatureCleaner,
    create_clean_feature_set,
    calculate_rsi
)

__all__ = [
    'FeatureCleaner',
    'create_clean_feature_set', 
    'calculate_rsi'
]
