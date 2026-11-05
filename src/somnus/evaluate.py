"""Held-out evaluation: metrics, per-night agreement, calibration and frequency-band attribution.

Everything is exported as tidy CSV so the R figures in `r/` can be rebuilt without Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from somnus import EPOCH_LEN, SFREQ, STAGES
from somnus.data import load_recordings
from somnus.metrics import fit_temperature, reliability, scored, softmax, summary
from somnus.model import SleepTransformer, predict_recording

BANDS = {
    "delta (0.5-4 Hz)": (0.5, 4.0),
    "theta (4-8 Hz)": (4.0, 8.0),
    "alpha (8-12 Hz)": (8.0, 12.0),
    "sigma (12-15 Hz)": (12.0, 15.0),
    "beta (15-30 Hz)": (15.0, 30.0),
}
FREQS = np.fft.rfftfreq(EPOCH_LEN, 1 / SFREQ)


def load_model(ckpt_path: str | Path) -> tuple[SleepTransformer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SleepTransformer(n_classes=len(STAGES), **cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def band_stop(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Remove one frequency band from every epoch (used for occlusion-style attribution)."""
    spec = np.fft.rfft(x.astype(np.float32), axis=-1)
    spec[:, (FREQS >= lo) & (FREQS < hi)] = 0
    return np.fft.irfft(spec, n=EPOCH_LEN, axis=-1).astype(np.float32)


def epoch_spectrum(x: np.ndarray, fmax: float = 30.0, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Log power per epoch in `step`-Hz bins up to `fmax` -> (n_epochs, n_bins), bin centres."""
    power = np.abs(np.fft.rfft(x.astype(np.float32), axis=-1)) ** 2
    edges = np.arange(0.5, fmax + step, step)
    out = np.stack(
        [power[:, (FREQS >= lo) & (FREQS < hi)].mean(1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
        axis=1,
    )
    return 10 * np.log10(out + 1e-6), (edges[:-1] + edges[1:]) / 2


def relative_band_power(x: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.rfft(x.astype(np.float32), axis=-1)) ** 2
    bands = np.stack([power[:, (FREQS >= lo) & (FREQS < hi)].sum(1) for lo, hi in BANDS.values()], 1)
    return bands / bands.sum(1, keepdims=True)


def evaluate(run: str | Path, log=print) -> dict:
    run = Path(run)
    model, cfg = load_model(run / "best.pt")
    split = json.loads((run / "split.json").read_text())["recordings"]
    processed, seq_len, stride = cfg["data"]["processed_dir"], cfg["model"]["seq_len"], cfg["eval"]["stride"]

    # 1. temperature scaling fitted on validation nights only
    val_logits, val_y = [], []
    for x, y in load_recordings(processed, split["val"]):
        yt, lg = scored(y.astype(np.int64), predict_recording(model, x, seq_len, stride))
        val_logits.append(lg)
        val_y.append(yt)
    temperature = fit_temperature(np.concatenate(val_logits), np.concatenate(val_y))
    log(f"temperature (fit on validation) = {temperature:.3f}")

    # 2. held-out test nights
    test_ids = split["test"]
    test_recs = load_recordings(processed, test_ids)
    pred_rows, per_rec, all_y, all_logits, band_rows = [], [], [], [], []
    for rec_id, (x, y) in zip(test_ids, test_recs, strict=True):
        logits = predict_recording(model, x, seq_len, stride)
        probs = softmax(logits, temperature)
        y = y.astype(np.int64)
        pred_rows.append(
            pd.DataFrame(
                {
                    "rec_id": rec_id,
                    "epoch": np.arange(len(y)),
                    "true": y,
                    "pred": probs.argmax(1),
                    **{f"p_{s}": probs[:, i] for i, s in enumerate(STAGES)},
                }
            )
        )
        yt, lg = scored(y, logits)
        s = summary(yt, lg.argmax(1))
        per_rec.append(
            {"rec_id": rec_id, "subject": int(rec_id[3:5]), "night": int(rec_id[5]),
             "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "kappa": s["kappa"],
             **{f"f1_{k}": v["f1"] for k, v in s["per_class"].items()}}
        )
        all_y.append(yt)
        all_logits.append(lg)
        rbp = relative_band_power(x)
        for b, name in enumerate(BANDS):
            band_rows.append(pd.DataFrame({"rec_id": rec_id, "true": y, "band": name, "rel_power": rbp[:, b]}))

    y_all, logits_all = np.concatenate(all_y), np.concatenate(all_logits)
    probs_raw, probs_cal = softmax(logits_all), softmax(logits_all, temperature)
    metrics = summary(y_all, probs_cal.argmax(1))
    rel_raw, ece_raw = reliability(probs_raw, y_all, cfg["eval"]["reliability_bins"])
    rel_cal, ece_cal = reliability(probs_cal, y_all, cfg["eval"]["reliability_bins"])
    metrics.update({"temperature": temperature, "ece_uncalibrated": ece_raw, "ece_calibrated": ece_cal,
                    "n_test_nights": len(test_ids), "n_test_epochs": int(len(y_all))})
    log(f"TEST  acc {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
        f"kappa {metrics['kappa']:.3f}  ECE {ece_raw:.3f} -> {ece_cal:.3f}")

    # 3. frequency-band occlusion: how much does removing each band hurt the true-stage probability?
    occl = []
    for name, (lo, hi) in BANDS.items():
        for (x, y), base in zip(test_recs, pred_rows, strict=True):
            p = softmax(predict_recording(model, band_stop(x, lo, hi), seq_len, stride), temperature)
            y = y.astype(np.int64)
            keep = y >= 0
            base_p = base[[f"p_{s}" for s in STAGES]].to_numpy()
            idx = np.flatnonzero(keep)
            drop = base_p[idx, y[idx]] - p[idx, y[idx]]
            occl.append(pd.DataFrame({"true": y[idx], "band": name, "prob_drop": drop}))
    occl_df = pd.concat(occl).groupby(["true", "band"], as_index=False)["prob_drop"].mean()
    occl_df["stage"] = occl_df["true"].map(dict(enumerate(STAGES)))

    # 4. spectrogram for the median-kappa test night (the README hero figure)
    meta = pd.read_parquet(Path(processed) / "metadata.parquet")[["rec_id", "age", "sex"]]
    per_rec_df = pd.DataFrame(per_rec).merge(meta, on="rec_id", how="left")
    per_rec_df = per_rec_df.sort_values("kappa").reset_index(drop=True)
    showcase = per_rec_df.iloc[len(per_rec_df) // 2]["rec_id"]
    spec, centres = epoch_spectrum(test_recs[test_ids.index(showcase)][0])
    spec_df = pd.DataFrame(spec, columns=centres).reset_index(names="epoch").melt(
        id_vars="epoch", var_name="freq", value_name="db")

    # 5. write everything
    out = run / "eval"
    out.mkdir(exist_ok=True)
    preds = pd.concat(pred_rows)
    preds.to_csv(out / "predictions.csv", index=False)
    per_rec_df.to_csv(out / "per_night.csv", index=False)
    cm = np.array(metrics["confusion"])
    pd.DataFrame([{"true": STAGES[i], "pred": STAGES[j], "n": int(cm[i, j])}
                  for i in range(len(STAGES)) for j in range(len(STAGES))]).to_csv(out / "confusion.csv", index=False)
    pd.concat([pd.DataFrame(rel_raw).assign(model="uncalibrated"),
               pd.DataFrame(rel_cal).assign(model="temperature-scaled")]).to_csv(out / "reliability.csv", index=False)
    occl_df.to_csv(out / "band_importance.csv", index=False)
    bands = pd.concat(band_rows)
    bands = bands[bands.true >= 0].assign(stage=lambda d: d.true.map(dict(enumerate(STAGES))))
    bands.to_csv(out / "band_power.csv", index=False)
    spec_df.assign(rec_id=showcase).to_csv(out / "showcase_spectrogram.csv", index=False)
    preds[preds.rec_id == showcase].to_csv(out / "showcase_hypnogram.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"wrote evaluation artefacts to {out} (showcase night: {showcase})")
    return metrics
