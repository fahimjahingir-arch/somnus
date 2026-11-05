"""Scoring utilities: agreement metrics, calibration and temperature scaling."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from somnus import STAGES


def scored(y_true: np.ndarray, *arrays: np.ndarray):
    """Drop unscored (-1) epochs from labels and any aligned arrays."""
    keep = y_true >= 0
    return (y_true[keep], *(a[keep] for a in arrays))


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(STAGES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            st: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": int(s[i])}
            for i, st in enumerate(STAGES)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter temperature scaling fitted by NLL on held-out (validation) data."""
    lt = torch.from_numpy(logits.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[list[dict], float]:
    """Reliability-diagram bins and expected calibration error (ECE)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - c)
        rows.append({"bin_lo": lo, "bin_hi": hi, "confidence": c, "accuracy": acc, "n": int(m.sum())})
    return rows, float(ece)
