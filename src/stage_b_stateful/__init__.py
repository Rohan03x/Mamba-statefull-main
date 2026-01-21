"""Stage B Phase-2 (stateful) runner.

This package implements the Phase-2 "stateful" evaluation loop:
- Load a frozen Stage-B best-trial parameter dict.
- Re-tune only a small refinement set.
- Train on a fixed train block.
- Evaluate on a fixed OOS block with burn-in and NO state resets.

The implementation is intentionally minimal and purpose-built for the spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["Phase2RefinementSpec", "Phase2Result", "evaluate_phase2_stateful_once", "run_phase2_stateful_optuna"]

if TYPE_CHECKING:
    from .phase2_stateful import Phase2RefinementSpec, Phase2Result, evaluate_phase2_stateful_once, run_phase2_stateful_optuna


def __getattr__(name: str):
    if name in __all__:
        from . import phase2_stateful as _p2

        return getattr(_p2, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
