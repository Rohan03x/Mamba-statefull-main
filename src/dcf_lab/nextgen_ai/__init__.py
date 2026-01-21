"""Next-Gen AI modules (optional, feature-gated).
This package provides optional plug-ins for:
- Sequence Transformers (TFT/Time-Series Transformer)
- Multimodal encoders (text/audio/tabular)
- Bayesian probabilistic calibration
- Causal graph filters
- Meta-learning / AutoML ensemble tuners
- Evolutionary search for feature/model combos

All modules are safe to import when dependencies are missing. Each exposes a
is_available() method and no-op fallbacks.
"""

from .feature_flags import NextGenFlags
from .transformers import SequenceTransformerModel
from .multimodal import MultimodalEncoder
from .bayesian import BayesianCalibrator
from .causal import CausalFilter
from .meta_automl import MetaAutoML
from .evolutionary import EvolutionarySearch
from .decision_policy import DecisionPolicy

__all__ = [
    "NextGenFlags",
    "SequenceTransformerModel",
    "MultimodalEncoder",
    "BayesianCalibrator",
    "CausalFilter",
    "MetaAutoML",
    "EvolutionarySearch",
    "DecisionPolicy",
]
