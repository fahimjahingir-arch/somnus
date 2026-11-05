"""Two-stage sleep stager: a CNN reads each 30 s epoch, a transformer reads the night around it."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EpochEncoder(nn.Module):
    """1-D CNN that turns one 30 s EEG epoch (3000 samples @ 100 Hz) into a feature vector."""

    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()

        def block(c_in: int, c_out: int, k: int, stride: int = 1) -> list[nn.Module]:
            return [
                nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            ]

        self.net = nn.Sequential(
            *block(1, 32, 50, stride=6),  # ~0.5 s receptive field: slow waves, spindles
            nn.MaxPool1d(8),
            nn.Dropout(dropout),
            *block(32, 64, 8),
            *block(64, 64, 8),
            *block(64, d_model, 8),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, 1, T) -> (N, d)
        return self.net(x).squeeze(-1)


class SleepTransformer(nn.Module):
    """Sequence-to-sequence stager over a window of consecutive epochs."""

    def __init__(
        self,
        n_classes: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
        **_,
    ):
        super().__init__()
        self.encoder = EpochEncoder(d_model, dropout)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, T) -> (B, L, C)
        b, seq, t = x.shape
        z = self.encoder(x.reshape(b * seq, 1, t)).reshape(b, seq, -1)
        z = self.context(z + self.pos[:, :seq])
        return self.head(z)


@torch.no_grad()
def predict_recording(
    model: nn.Module, x: np.ndarray, seq_len: int, stride: int, batch_size: int = 64
) -> np.ndarray:
    """Slide a window over a whole night and average the overlapping logits per epoch."""
    from somnus.dataset import window_starts

    model.eval()
    n = len(x)
    starts = window_starts(n, seq_len, stride)
    total = np.zeros((n, model.head[-1].out_features), dtype=np.float64)
    hits = np.zeros(n, dtype=np.float64)
    xt = torch.from_numpy(x.astype(np.float32))
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        batch = torch.stack([xt[s : s + seq_len] for s in chunk])
        logits = model(batch).numpy()
        for s, lg in zip(chunk, logits, strict=True):
            total[s : s + seq_len] += lg
            hits[s : s + seq_len] += 1
    return (total / hits[:, None]).astype(np.float32)
