"""Somnus: sleep staging from single-channel EEG."""

STAGES = ["W", "N1", "N2", "N3", "REM"]
EPOCH_SEC = 30
SFREQ = 100
EPOCH_LEN = EPOCH_SEC * SFREQ

__version__ = "0.1.0"
