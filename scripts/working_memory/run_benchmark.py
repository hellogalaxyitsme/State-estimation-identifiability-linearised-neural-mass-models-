#!/usr/bin/env python3
"""Compare the static, lagged linear and generative scalp mappings across participants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_observability.working_memory.benchmark import load_config, run_benchmark

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "working_memory_benchmark.json")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--bootstraps", type=int, default=None)
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    result, path = run_benchmark(
        config, project_root, max_subjects=args.max_subjects, bootstrap_resamples=args.bootstraps
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
