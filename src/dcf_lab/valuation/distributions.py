"""
Distribution specifications for DCF Monte Carlo simulations.

This module re-exports DistributionSpec and related functions from dcf_mc.py
to provide consistent import structure across the codebase.
"""

# Import distribution specification and helper functions
from .dcf_mc import (
    DistributionSpec,
    constant_value,
    custom_distribution,
    normal_distribution,
    triangular_distribution,
    uniform_distribution,
)

__all__ = [
    'DistributionSpec',
    'constant_value',
    'normal_distribution',
    'uniform_distribution',
    'triangular_distribution',
    'custom_distribution'
]
