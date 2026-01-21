"""Stage B orchestration package (Mamba-only mode).

Keep this module *lightweight*.

Historically this file eagerly imported `pipeline` and `meta_optimizer`, which
pulled in heavy optional dependencies (e.g. sklearn/scipy) even when callers
only needed small constants like `DEFAULT_CANDIDATE_UNIVERSE` from
`src.stage_b.universe_selector`. That caused slow startups and, on some pods,
apparent "hangs" before logging was configured.

We provide lazy attribute access via `__getattr__` so existing imports like:

    from src.stage_b import StageBRunner

continue to work, without paying import costs unless those symbols are
actually accessed.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING


_PIPELINE_EXPORTS = {
    "FamilySpec",
    "PrepFamiliesSettings",
    "StageBConfig",
    "StageBPipeline",
    "StageBRunner",
    "StageBResult",
    "TierTwoResult",
    "ModelCandidate",
    "TrackData",
    "FAMILIES",
}

_META_OPT_EXPORTS = {
    "MetaOptimizer",
    "MetaOptimizerConfig",
    "TrialMemory",
    "AttributionEngine",
    "InteractionLearner",
    "SearchSpaceUpdater",
    "AdaptedParam",
    "MetaLearningFeedback",
    "FeedbackAction",
    "create_meta_optimizer",
}

__all__ = sorted(_PIPELINE_EXPORTS | _META_OPT_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _PIPELINE_EXPORTS:
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in _META_OPT_EXPORTS:
        from . import meta_optimizer as _meta

        return getattr(_meta, name)
    raise AttributeError(f"module 'src.stage_b' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _PIPELINE_EXPORTS | _META_OPT_EXPORTS)


if TYPE_CHECKING:
    from .meta_optimizer import (  # noqa: F401
        AttributionEngine,
        AdaptedParam,
        FeedbackAction,
        InteractionLearner,
        MetaLearningFeedback,
        MetaOptimizer,
        MetaOptimizerConfig,
        SearchSpaceUpdater,
        TrialMemory,
        create_meta_optimizer,
    )
    from .pipeline import (  # noqa: F401
        FAMILIES,
        FamilySpec,
        ModelCandidate,
        PrepFamiliesSettings,
        StageBConfig,
        StageBPipeline,
        StageBResult,
        StageBRunner,
        TierTwoResult,
        TrackData,
    )
