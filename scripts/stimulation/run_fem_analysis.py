#!/usr/bin/env python3
"""Run the finite-element analysis of the stimulation cohort.

Requires the post-CHARM eligibility table, the head-model QC records and the
nominal-conductivity lead fields (see run_segmentation.py and run_lead_fields.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_observability.stimulation.cohort import load_config
from neural_observability.stimulation.fem_analysis import run_fem_analysis

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stimulation_fem.json")
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    results, path = run_fem_analysis(config, project_root)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
