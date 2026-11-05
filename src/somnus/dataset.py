"""PyTorch dataset of fixed-length windows of consecutive 30-second epochs."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def window_starts(n: int, seq_len: int, stride: int) -> list[int]:
    """Window start indices covering [0, n), always including a window that ends at n."""
    if n < seq_len:
        return []
    starts = list(range(0, n - seq_len + 1, stride))
    if starts[-1] != n - seq_len:
        starts.append(n - seq_len)
    return starts


class SequenceDataset(Dataset):
    def __init__(
        self,
        recordings: list[tuple[np.ndarray, np.ndarray]],
        seq_len: int,
        stride: int,
        augment: bool = False,
    ):
        self.recordings = recordings
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.index = [
            (i, s)
            for i, (_, y) in enumerate(recordings)
            for s in window_starts(len(y), seq_len, stride)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, s = self.index[k]
        x, y = self.recordings[i]
        if self.augment:  # random shift so each pass sees different window alignments
            s = int(np.clip(s + np.random.randint(-self.stride // 2, self.stride // 2 + 1), 0, len(y) - self.seq_len))
        xs = torch.from_numpy(x[s : s + self.seq_len].astype(np.float32))
        ys = torch.from_numpy(y[s : s + self.seq_len].astype(np.int64))
        if self.augment:
            xs = xs * torch.empty(1).uniform_(0.8, 1.2) + 0.05 * torch.randn_like(xs)
            if torch.rand(1) < 0.5:
                xs = -xs  # EEG polarity flip
        return xs, ys
