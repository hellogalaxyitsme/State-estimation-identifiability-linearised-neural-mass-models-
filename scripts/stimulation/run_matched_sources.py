#!/usr/bin/env python3
"""Compare each stimulated site with matched wrong sources and the nearest other site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_observability.stimulation.matched_sources import load_config, run_matched_sources

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stimulation_matched_sources.json")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--bootstraps", type=int, default=None)
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    result, path = run_matched_sources(
        config,
        project_root,
        max_subjects=args.max_subjects,
        permutation_count=args.permutations,
        bootstrap_count=args.bootstraps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
