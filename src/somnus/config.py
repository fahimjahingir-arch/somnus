from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir(cfg: dict, root: str | Path = "runs") -> Path:
    d = Path(root) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
