import numpy as np
import torch

from somnus import EPOCH_LEN
from somnus.data import UNSCORED, labels_from_annotations, parse_recording_id, sleep_window
from somnus.dataset import SequenceDataset, window_starts
from somnus.evaluate import band_stop, relative_band_power
from somnus.metrics import fit_temperature, reliability, softmax
from somnus.model import SleepTransformer, predict_recording
from somnus.train import subject_split


def test_labels_from_annotations_maps_and_merges_stages():
    labels = labels_from_annotations(
        np.array([0.0, 60.0, 120.0, 150.0, 180.0]),
        np.array([60.0, 60.0, 30.0, 30.0, 60.0]),
        ["Sleep stage W", "Sleep stage 2", "Sleep stage 3", "Sleep stage 4", "Movement time"],
        n_epochs=8,
    )
    assert labels.tolist() == [0, 0, 2, 2, 3, 3, UNSCORED, UNSCORED]


def test_sleep_window_pads_with_wake():
    y = np.array([0] * 10 + [2] * 5 + [0] * 10, dtype=np.int8)
    assert sleep_window(y, edge_epochs=3) == (7, 18)


def test_parse_recording_id():
    assert parse_recording_id("/x/SC4012E0-PSG.edf") == ("SC4012", 1, 2)


def test_window_starts_cover_the_end():
    assert window_starts(10, 4, 3) == [0, 3, 6]
    assert window_starts(11, 4, 3) == [0, 3, 6, 7]
    assert window_starts(3, 4, 1) == []


def test_subject_split_is_disjoint():
    split = subject_split(list(range(20)), n_val=3, n_test=4, seed=0)
    sets = [set(v) for v in split.values()]
    assert sum(map(len, sets)) == 20
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_stratified_split_samples_every_group():
    groups = {s: s // 5 for s in range(20)}  # four groups of five
    split = subject_split(list(range(20)), n_val=4, n_test=4, seed=3, groups=groups)
    assert sorted(groups[s] for s in split["test"]) == [0, 1, 2, 3]
    assert sorted(groups[s] for s in split["val"]) == [0, 1, 2, 3]


def _toy_recording(n=30):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, EPOCH_LEN)).astype(np.float16), rng.integers(-1, 5, n).astype(np.int8)


def test_dataset_and_model_shapes():
    ds = SequenceDataset([_toy_recording()], seq_len=7, stride=5, augment=True)
    xb, yb = ds[0]
    assert xb.shape == (7, EPOCH_LEN) and yb.shape == (7,)
    model = SleepTransformer(d_model=32, n_heads=2, n_layers=1)
    assert model(xb.unsqueeze(0)).shape == (1, 7, 5)


def test_predict_recording_covers_every_epoch():
    x, _ = _toy_recording(23)
    torch.manual_seed(0)
    logits = predict_recording(SleepTransformer(d_model=32, n_heads=2, n_layers=1), x, 7, 3)
    assert logits.shape == (23, 5) and np.isfinite(logits).all()


def test_band_stop_removes_band():
    t = np.arange(EPOCH_LEN) / 100
    x = (np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 10 * t))[None]
    rbp = relative_band_power(band_stop(x, 0.5, 4.0))
    assert rbp[0, 0] < 1e-6 and rbp[0, 2] > 0.99


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 2000)
    logits = rng.standard_normal((2000, 5)) * 0.5
    logits[np.arange(2000), y] += 1.0
    logits *= 6  # deliberately overconfident
    t = fit_temperature(logits, y)
    assert t > 1.5
    _, ece_raw = reliability(softmax(logits), y)
    _, ece_cal = reliability(softmax(logits, t), y)
    assert ece_cal < ece_raw
