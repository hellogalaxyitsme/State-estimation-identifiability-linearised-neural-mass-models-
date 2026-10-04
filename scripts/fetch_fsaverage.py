#!/usr/bin/env python3
"""Download the MNE fsaverage template head into data/subjects (used by the network and working-memory analyses)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-neural-observability")

import mne


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS_DIR = ROOT / "data" / "subjects"


def main() -> None:
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SUBJECTS_DIR"] = str(SUBJECTS_DIR)
    fsaverage = mne.datasets.fetch_fsaverage(subjects_dir=SUBJECTS_DIR)
    print(f"fsaverage is ready at {fsaverage}")


if __name__ == "__main__":
    main()
