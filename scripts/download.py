"""Parallel, checksum-verified download of Sleep-EDF cassette recordings.

PhysioNet throttles each connection (~0.3-0.5 MB/s), so fetching files concurrently is ~10x faster
than MNE's sequential fetcher. Files land where MNE expects them, so `somnus prepare` reuses them.

    python scripts/download.py --subjects 0-23 --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from mne.datasets.sleep_physionet._utils import AGE_SLEEP_RECORDS

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(fname: str, sha: str, dest: Path, retries: int = 6) -> str:
    target = dest / fname
    if target.exists() and sha1(target) == sha:
        return f"cached  {fname}"
    tmp = target.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(BASE + fname, tmp)
            if sha1(tmp) == sha:
                tmp.replace(target)
                return f"done    {fname}" + (f" (attempt {attempt})" if attempt > 1 else "")
        except OSError:
            pass  # PhysioNet resets long connections now and then; back off and retry
        time.sleep(5 * attempt)
    tmp.unlink(missing_ok=True)
    return f"FAILED  {fname}"


def parse_range(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="0-23")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    root = Path(os.environ.get("MNE_DATA", Path.home() / "mne_data")) / "physionet-sleep-data"
    root.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(AGE_SLEEP_RECORDS)
    todo = table[table.subject.isin(parse_range(args.subjects))]
    print(f"{len(todo)} files -> {root}")
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fetch, r.fname, r.sha, root) for r in todo.itertuples()]
        for i, job in enumerate(as_completed(jobs), 1):
            print(f"[{i:3d}/{len(jobs)}] {job.result()}", flush=True)


if __name__ == "__main__":
    main()
