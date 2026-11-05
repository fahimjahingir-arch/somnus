"""Download Sleep-EDF Expanded recordings and turn them into labelled 30-second EEG epochs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from somnus import EPOCH_LEN, EPOCH_SEC, SFREQ, STAGES

# R&K scoring in Sleep-EDF -> AASM 5-class scheme (stages 3 and 4 merge into N3).
ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
UNSCORED = -1

_SC_NAME = re.compile(r"SC4(\d{2})(\d)")


def fetch_recordings(
    subjects: list[int], recordings: list[int], path: str | None = None
) -> list[tuple[str, str]]:
    """Download (or reuse cached) PSG + hypnogram EDF pairs from PhysioNet via MNE."""
    from mne.datasets.sleep_physionet.age import fetch_data

    files = fetch_data(subjects=subjects, recording=recordings, path=path, on_missing="warn")
    return [(str(psg), str(hyp)) for psg, hyp in files]


def parse_recording_id(psg_path: str | Path) -> tuple[str, int, int]:
    """'SC4012E0-PSG.edf' -> ('SC4012', subject=1, night=2)."""
    name = Path(psg_path).name
    m = _SC_NAME.match(name)
    if not m:
        raise ValueError(f"Not a Sleep-EDF SC filename: {name}")
    return name[:6], int(m.group(1)), int(m.group(2))


def labels_from_annotations(
    onsets: np.ndarray, durations: np.ndarray, descriptions: list[str], n_epochs: int
) -> np.ndarray:
    """Expand hypnogram annotations (onset/duration in seconds) into one label per 30 s epoch.

    Epochs with no stage, movement time or '?' stay UNSCORED (-1) and are masked out of the loss.
    """
    labels = np.full(n_epochs, UNSCORED, dtype=np.int8)
    for onset, duration, desc in zip(onsets, durations, descriptions, strict=True):
        stage = ANNOTATION_MAP.get(desc)
        if stage is None:
            continue
        start = int(round(onset / EPOCH_SEC))
        stop = min(n_epochs, start + int(round(duration / EPOCH_SEC)))
        if start < n_epochs and stop > start:
            labels[max(start, 0) : stop] = stage
    return labels


def sleep_window(labels: np.ndarray, edge_epochs: int) -> tuple[int, int]:
    """Indices covering sleep onset -> final awakening, padded with `edge_epochs` of wake."""
    asleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    if asleep.size == 0:
        return 0, len(labels)
    return max(0, asleep[0] - edge_epochs), min(len(labels), asleep[-1] + 1 + edge_epochs)


def robust_scale(x: np.ndarray) -> np.ndarray:
    """Per-recording median/IQR scaling so amplitude differences between people don't dominate."""
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) / 1.349 or 1.0
    return np.clip((x - med) / scale, -20, 20)


def load_eeg(psg_path: str, channel: str, l_freq: float, h_freq: float):
    import mne

    raw = mne.io.read_raw_edf(psg_path, include=[channel], preload=True, verbose="error")
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose="error")
    raw.filter(l_freq, h_freq, verbose="error")
    return raw


def preprocess_recording(
    psg_path: str,
    hyp_path: str | None,
    channel: str,
    l_freq: float,
    h_freq: float,
    edge_minutes: int,
) -> dict:
    """Return epochs `x` (n, 3000) float16, labels `y` (n,), and the start offset in epochs."""
    import mne

    raw = load_eeg(psg_path, channel, l_freq, h_freq)
    signal = raw.get_data(picks=[channel])[0] * 1e6  # volts -> microvolts
    n_epochs = len(signal) // EPOCH_LEN

    if hyp_path is not None:
        raw.set_annotations(mne.read_annotations(hyp_path), emit_warning=False)
        ann = raw.annotations
        y = labels_from_annotations(
            ann.onset - raw.first_time, ann.duration, list(ann.description), n_epochs
        )
        start, stop = sleep_window(y, edge_minutes * 60 // EPOCH_SEC)
    else:
        y = np.full(n_epochs, UNSCORED, dtype=np.int8)
        start, stop = 0, n_epochs

    x = signal[start * EPOCH_LEN : stop * EPOCH_LEN].reshape(-1, EPOCH_LEN)
    x = robust_scale(x).astype(np.float16)
    return {"x": x, "y": y[start:stop], "start_epoch": start}


def build_dataset(cfg: dict, progress=None) -> pd.DataFrame:
    """Download + preprocess every configured recording; returns the metadata table."""
    dcfg = cfg["data"]
    out = Path(dcfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pairs = fetch_recordings(dcfg["subjects"], dcfg["recordings"], dcfg.get("raw_dir"))

    rows = []
    for psg, hyp in progress(pairs) if progress else pairs:
        rec_id, subject, night = parse_recording_id(psg)
        target = out / f"{rec_id}.npz"
        if target.exists():
            y = np.load(target)["y"]
        else:
            rec = preprocess_recording(
                psg, hyp, dcfg["channel"], dcfg["l_freq"], dcfg["h_freq"], dcfg["edge_minutes"]
            )
            np.savez_compressed(target, x=rec["x"], y=rec["y"], start_epoch=rec["start_epoch"])
            y = rec["y"]
        counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
        rows.append(
            {"rec_id": rec_id, "subject": subject, "night": night, "n_epochs": len(y), **counts}
        )

    meta = pd.DataFrame(rows).sort_values("rec_id").reset_index(drop=True)
    meta = meta.merge(subject_demographics(), on="subject", how="left")
    meta.to_parquet(out / "metadata.parquet", index=False)
    meta.to_csv(out / "metadata.csv", index=False)
    return meta


def subject_demographics() -> pd.DataFrame:
    """Age and sex per Sleep-EDF subject (shipped with MNE's record table)."""
    from mne.datasets.sleep_physionet._utils import AGE_SLEEP_RECORDS

    table = pd.read_csv(AGE_SLEEP_RECORDS)
    return table.groupby("subject", as_index=False)[["age", "sex"]].first()


def load_recordings(processed_dir: str | Path, rec_ids: list[str]) -> list[tuple[np.ndarray, ...]]:
    recs = []
    for rec_id in rec_ids:
        with np.load(Path(processed_dir) / f"{rec_id}.npz") as f:
            recs.append((f["x"], f["y"]))
    return recs
