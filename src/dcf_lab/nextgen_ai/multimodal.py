from typing import Dict
import numpy as np

class MultimodalEncoder:
    """Encodes text/news/transcripts into vectors and fuses with tabular/price.
    Placeholder-only: returns zeros when unavailable.
    """
    def __init__(self):
        self._available = self._check_deps()

    def _check_deps(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    def encode_texts(self, texts: Dict[str, str]) -> Dict[str, np.ndarray]:
        if not self._available:
            return {k: np.zeros(384, dtype=float) for k in texts}
        return {k: np.zeros(384, dtype=float) for k in texts}

    def fuse(self, tabular: np.ndarray, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        if embeddings:
            emb = np.mean(np.stack(list(embeddings.values())), axis=0)
            emb = emb.reshape(1, -1).repeat(len(tabular), axis=0)
            return np.concatenate([tabular, emb], axis=1)
        return tabular
