"""
Feature engineering modules for DCF Suite.

This package contains specialized feature bundles for different market conditions.
"""

from .downside_v1 import DownsideFeaturesV1, get_feature_bundle

__all__ = ['DownsideFeaturesV1', 'get_feature_bundle']
