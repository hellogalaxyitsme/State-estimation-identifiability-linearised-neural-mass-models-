#!/usr/bin/env python3
"""Fit and score the generative state-space mapping in every working-memory session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_observability.working_memory.generative_analysis import load_config, run_generative_analysis

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "working_memory_generative.json")
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    result, path = run_generative_analysis(config, project_root, max_subjects=args.max_subjects)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
