import numpy as np


class SequenceTransformerModel:
    """Lightweight Time-Series Transformer (TST/TFT-style) wrapper using PyTorch.

    Designed to be fast and optional; if torch is unavailable, it degrades to a no-op.
    Trains a tiny encoder over very short sequences (seq_len may be 1 in current pipeline).
    """

    def __init__(self, input_dim: int, seq_len: int, hidden_size: int = 64, n_heads: int = 1, n_layers: int = 1, epochs: int = 5, lr: float = 1e-3):
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self._available = self._check_deps()
        self._fitted = False
        self._device = 'cpu'
        self._model = None

    def _check_deps(self) -> bool:
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    def _build_model(self):
        import torch.nn as nn

        d_model = max(8, self.input_dim)

        class TinyTST(nn.Module):
            def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int):
                super().__init__()
                self.proj = nn.Linear(input_dim, d_model)
                enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
                self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 1)
                )

            def forward(self, x):
                # x: (B, T, F)
                h = self.proj(x)
                h = self.enc(h)
                # mean-pool over time
                h = h.mean(dim=1)
                out = self.head(h).squeeze(-1)
                return out

        self._model = TinyTST(self.input_dim, d_model, self.n_heads, self.n_layers).to(self._device)

    def fit(self, x_seq: np.ndarray, y: np.ndarray, **kwargs):
        if not self._available:
            return
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            x = torch.tensor(x_seq, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[2] != self.input_dim:
                # attempt to reshape single-step
                B = x.shape[0]
                x = x.view(B, 1, -1)
            self._device = 'cpu'
            self._build_model()
            model = self._model
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
            model.train()
            epochs = int(kwargs.get('epochs', self.epochs))
            for _ in range(max(1, epochs)):
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            self._fitted = True
        except Exception:
            # keep silent; fall back to uniform proba
            self._fitted = False

    def predict_proba(self, x_seq: np.ndarray) -> np.ndarray:
        if not self._available or not self._fitted or self._model is None:
            return np.full((len(x_seq),), 0.5, dtype=float)
        try:
            import torch
            self._model.eval()
            x = torch.tensor(x_seq, dtype=torch.float32)
            if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[2] != self.input_dim:
                B = x.shape[0]
                x = x.view(B, 1, -1)
            with torch.no_grad():
                logits = self._model(x)
                proba = torch.sigmoid(logits).cpu().numpy().astype(float)
            return proba
        except Exception:
            return np.full((len(x_seq),), 0.5, dtype=float)
