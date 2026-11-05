"""Training loop with subject-wise splits, class-balanced loss and early stopping on Cohen's kappa."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from somnus import STAGES
from somnus.config import run_dir, seed_everything
from somnus.data import load_recordings
from somnus.dataset import SequenceDataset
from somnus.metrics import scored, summary
from somnus.model import SleepTransformer, predict_recording

AGE_BINS = [0, 40, 65, 80, 200]


def subject_split(
    subjects: list[int], n_val: int, n_test: int, seed: int, groups: dict[int, int] | None = None
) -> dict[str, list[int]]:
    """Assign whole subjects (both nights) to train / val / test so nobody leaks across splits.

    With `groups` (subject -> group id, e.g. age band) subjects are shuffled within each group and
    then interleaved, so test and validation sets draw evenly from every group.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        order = [int(s) for s in rng.permutation(sorted(subjects))]
    else:
        buckets = [
            [int(s) for s in rng.permutation(sorted(s for s in subjects if groups[s] == g))]
            for g in sorted(set(groups[s] for s in subjects))
        ]
        order = [b[i] for i in range(max(map(len, buckets))) for b in buckets if i < len(b)]
    return {
        "test": sorted(order[:n_test]),
        "val": sorted(order[n_test : n_test + n_val]),
        "train": sorted(order[n_test + n_val :]),
    }


def recs_for(meta: pd.DataFrame, subjects: list[int]) -> list[str]:
    return meta.loc[meta.subject.isin(subjects), "rec_id"].tolist()


def class_weights(recordings, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes)
    for _, y in recordings:
        counts += np.bincount(y[y >= 0], minlength=n_classes)
    w = np.sqrt(counts.sum() / (n_classes * np.maximum(counts, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def evaluate_recordings(model, recordings, seq_len: int, stride: int) -> dict:
    ys, ps = [], []
    for x, y in recordings:
        logits = predict_recording(model, x, seq_len, stride)
        yt, lg = scored(y.astype(np.int64), logits)
        ys.append(yt)
        ps.append(lg.argmax(1))
    return summary(np.concatenate(ys), np.concatenate(ps))


def train(cfg: dict, log=print) -> Path:
    tcfg, mcfg = cfg["train"], cfg["model"]
    seed_everything(tcfg["seed"])
    torch.set_num_threads(os.cpu_count() or 4)
    out = run_dir(cfg)
    processed = cfg["data"]["processed_dir"]

    meta = pd.read_parquet(Path(processed) / "metadata.parquet")
    meta = meta[meta.subject.isin(cfg["data"]["subjects"])]
    age_group = meta.groupby("subject").age.first().pipe(pd.cut, AGE_BINS, labels=False)
    split = subject_split(
        meta.subject.unique().tolist(),
        cfg["split"]["val_subjects"],
        cfg["split"]["test_subjects"],
        cfg["split"]["seed"],
        groups={int(k): int(v) for k, v in age_group.items()},
    )
    split_recs = {k: recs_for(meta, v) for k, v in split.items()}
    (out / "split.json").write_text(json.dumps({"subjects": split, "recordings": split_recs}, indent=2))
    log(f"subjects  train={split['train']}  val={split['val']}  test={split['test']}")

    train_recs = load_recordings(processed, split_recs["train"])
    val_recs = load_recordings(processed, split_recs["val"])

    seq_len = mcfg["seq_len"]
    loader = DataLoader(
        SequenceDataset(train_recs, seq_len, tcfg["train_stride"], augment=True),
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    model = SleepTransformer(n_classes=len(STAGES), **mcfg)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{len(loader.dataset)} training windows | {n_params / 1e3:.0f}k parameters")

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights(train_recs, len(STAGES)), ignore_index=-1, label_smoothing=0.05
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(loader)
    warmup = tcfg["warmup_epochs"] * len(loader)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history, best_kappa, stale = [], -1.0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits.reshape(-1, len(STAGES)), yb.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(xb)
            n += len(xb)

        val = evaluate_recordings(model, val_recs, seq_len, cfg["eval"]["stride"])
        row = {
            "epoch": epoch,
            "train_loss": running / n,
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_kappa": val["kappa"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        improved = val["kappa"] > best_kappa
        log(
            f"epoch {epoch:2d}  loss {row['train_loss']:.3f}  val acc {val['accuracy']:.3f}  "
            f"F1 {val['macro_f1']:.3f}  kappa {val['kappa']:.3f}  ({row['seconds']:.0f}s)"
            + ("  *" if improved else "")
        )
        if improved:
            best_kappa, stale = val["kappa"], 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out / "best.pt")
        else:
            stale += 1
            if stale >= tcfg["patience"]:
                log(f"early stop: no val kappa improvement for {stale} epochs")
                break
    return out
