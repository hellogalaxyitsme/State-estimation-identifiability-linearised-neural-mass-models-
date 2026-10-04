#!/usr/bin/env python3
"""Run the spherical head-model analysis of the stimulation cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_observability.stimulation.spherical import (
    load_config,
    run_spherical_analysis,
    write_eligibility,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stimulation_spherical.json")
    parser.add_argument("--eligibility-only", action="store_true",
                        help="read geometry and array headers only and write the eligibility table")
    parser.add_argument("--max-subjects", type=int, default=None, help="limit the number of subjects")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.eligibility_only:
        results, path = write_eligibility(config, project_root)
    else:
        results, path = run_spherical_analysis(config, project_root, max_subjects=args.max_subjects)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
