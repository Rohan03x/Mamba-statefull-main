from dataclasses import dataclass

@dataclass
class NextGenFlags:
    enable_transformers: bool = False
    enable_multimodal: bool = False
    enable_bayesian: bool = False
    enable_causal: bool = False
    enable_meta_automl: bool = False
    enable_evolutionary: bool = False
    enable_decision_policy: bool = False

    @staticmethod
    def defaults():
        return NextGenFlags()
