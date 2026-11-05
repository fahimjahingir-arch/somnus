"""Command-line interface: `somnus prepare | train | evaluate | predict`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from tqdm import tqdm

from somnus import EPOCH_SEC, STAGES

app = typer.Typer(add_completion=False, help="Sleep staging from single-channel EEG.")
console = Console()


@app.command()
def prepare(config: Path = typer.Option("configs/default.yaml", help="Experiment config.")):
    """Download Sleep-EDF recordings from PhysioNet and build 30 s epoch arrays."""
    from somnus.config import load_config
    from somnus.data import build_dataset

    meta = build_dataset(load_config(config), progress=lambda it: tqdm(it, desc="recordings"))
    console.print(f"[green]{len(meta)} nights, {meta.n_epochs.sum():,} epochs[/green]")
    console.print(meta[list(STAGES)].sum().to_string())


@app.command()
def train(config: Path = typer.Option("configs/default.yaml")):
    """Train the CNN + transformer stager with subject-wise validation."""
    from somnus.config import load_config
    from somnus.train import train as run_train

    out = run_train(load_config(config), log=console.print)
    console.print(f"[green]best checkpoint: {out / 'best.pt'}[/green]")


@app.command()
def evaluate(run: Path = typer.Argument(Path("runs/transformer-fpzcz"))):
    """Score the held-out test nights and export tidy CSVs for the R figures."""
    from somnus.evaluate import evaluate as run_eval

    run_eval(run, log=console.print)


@app.command()
def predict(
    psg: Path = typer.Argument(..., help="Sleep-EDF style PSG .edf file"),
    checkpoint: Path = typer.Option(Path("runs/transformer-fpzcz/best.pt")),
    out: Path = typer.Option(Path("hypnogram.csv")),
):
    """Stage a whole night from an EDF file and write a hypnogram CSV."""
    from somnus.data import preprocess_recording
    from somnus.evaluate import load_model
    from somnus.metrics import softmax
    from somnus.model import predict_recording

    model, cfg = load_model(checkpoint)
    d = cfg["data"]
    rec = preprocess_recording(str(psg), None, d["channel"], d["l_freq"], d["h_freq"], 0)
    probs = softmax(predict_recording(model, rec["x"], cfg["model"]["seq_len"], cfg["eval"]["stride"]))
    df = pd.DataFrame({"start_sec": np.arange(len(probs)) * EPOCH_SEC,
                       "stage": [STAGES[i] for i in probs.argmax(1)],
                       **{f"p_{s}": probs[:, i].round(4) for i, s in enumerate(STAGES)}})
    df.to_csv(out, index=False)
    console.print(f"[green]{len(df)} epochs -> {out}[/green]")
    console.print(df.stage.value_counts().mul(EPOCH_SEC / 60).round(1).rename("minutes").to_string())


if __name__ == "__main__":
    app()
