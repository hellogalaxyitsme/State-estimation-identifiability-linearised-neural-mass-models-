#!/usr/bin/env python3
"""Run CHARM segmentation and automated head-model QC for the eligible cohort.

After visual review of the QC overlays (recorded in headmodel_qc/visual_qc.csv),
run with --post-charm-eligibility to write the eligibility table used by the
lead-field and outcome analyses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from neural_observability.head_models.fem_qc import make_qc_review_index, write_qc_cohort_summary
from neural_observability.stimulation.cohort import (
    load_config,
    load_eligible_runs,
    write_post_charm_eligibility,
    write_pre_charm_eligibility,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stimulation_fem.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--post-charm-eligibility", action="store_true",
                        help="only write the post-CHARM eligibility table")
    return parser.parse_args()


def segment(config: dict, config_path: Path, output: Path, subject: str, threads: int) -> dict:
    cache = Path(config["cache_root"])
    segmentations = cache / "segmentations"
    logs = cache / "logs"
    segmentations.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    charm = Path(config["software"]["simnibs_root"]) / "simnibs_env" / "bin" / "charm"
    t1 = Path(config["dataset"]["root"]) / subject / "anat" / f"{subject}_T1w.nii"
    mesh = segmentations / f"m2m_{subject}" / f"{subject}.msh"
    log = logs / f"charm_{subject}.log"
    if not mesh.is_file():
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(threads)
        environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                [str(charm), subject, str(t1), "--forceqform"],
                cwd=segmentations, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0 or not mesh.is_file():
            return {"status": "charm_failed", "log": str(log)}
    qc_path = output / "headmodel_qc" / f"{subject}_automated_qc.json"
    if not qc_path.is_file():
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "stimulation" / "run_head_model.py"), subject,
             "--config", str(config_path), "--qc-only"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            return {"status": "qc_failed", "error": completed.stderr[-2000:]}
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    return {"status": "complete", "automated_qc_pass": qc.get("automated_qc_pass") is True}


def main() -> None:
    args = parse_args()
    config, project_root = load_config(args.config)
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = project_root / output
    if not args.post_charm_eligibility:
        write_pre_charm_eligibility(config, project_root)
        subjects = list(load_eligible_runs(output / "pre_charm_eligibility.csv"))
        status = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(segment, config, args.config.resolve(), output, s, args.threads_per_worker): s for s in subjects}
            for future in as_completed(futures):
                status[futures[future]] = future.result()
        write_qc_cohort_summary(output / "headmodel_qc")
        make_qc_review_index(output / "headmodel_qc")
        print(json.dumps(status, indent=2, sort_keys=True))
    summary, path = write_post_charm_eligibility(config, project_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
