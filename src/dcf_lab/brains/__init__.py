"""Module registry helpers for aggregator-aware brains."""

from __future__ import annotations

from .family_brains import (
    DEFAULT_FAMILIES,
    FamilyModule,
    build_family_modules,
    register_families_as_modules,
)

__all__ = [
    "DEFAULT_FAMILIES",
    "FamilyModule",
    "build_family_modules",
    "register_families_as_modules",
]
