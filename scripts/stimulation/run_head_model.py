#!/usr/bin/env python3
"""Prepare inputs, run automated QC or compute the lead field of one head model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from neural_observability.head_models.fem import (
    automated_head_model_qc,
    compute_subject_fem_leadfield,
    prepare_subject_forward_inputs,
)
from neural_observability.head_models.fem_qc import (
    make_electrode_projection_overlay,
    make_segmentation_overlay,
)
from neural_observability.stimulation.cohort import (
    load_config,
    load_eligible_runs,
    load_visual_qc_passes,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", help="BIDS subject identifier, for example sub-01")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stimulation_fem.json")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare-inputs", action="store_true")
    actions.add_argument("--qc-only", action="store_true")
    actions.add_argument("--leadfield", action="store_true")
    parser.add_argument("--skull-conductivity", type=float, default=None)
    parser.add_argument("--cpus", type=int, default=1)
    return parser.parse_args()


def run_qc(config: dict, output: Path, subject: str) -> tuple[dict, Path]:
    result = automated_head_model_qc(config, subject)
    qc_dir = output / "headmodel_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = Path(config["cache_root"]) / "segmentations" / f"m2m_{subject}"
    make_segmentation_overlay(
        subject_dir / "T1.nii.gz",
        subject_dir / "final_tissues.nii.gz",
        qc_dir / f"{subject}_segmentation_overlay.png",
        subject,
    )
    geometry = result.pop("electrode_projection_geometry")
    make_electrode_projection_overlay(
        np.asarray(geometry["scalp_display_vertices_mm"], dtype=float),
        np.asarray(geometry["raw_positions_mm"], dtype=float),
        np.asarray(geometry["projected_positions_mm"], dtype=float),
        np.asarray(geometry["raw_landmark_positions_mm"], dtype=float),
        np.asarray(geometry["projected_landmark_positions_mm"], dtype=float),
        list(geometry["landmark_names"]),
        qc_dir / f"{subject}_electrode_projection_overlay.png",
        subject,
    )
    shutil.copy2(subject_dir / "charm_report.html", qc_dir / f"{subject}_charm_report.html")
    path = qc_dir / f"{subject}_automated_qc.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, path


def run_leadfield(config: dict, output: Path, subject: str, conductivity: float, cpus: int) -> tuple[dict, Path]:
    eligible = load_eligible_runs(output / "post_charm_eligibility.csv", "post_charm_eligible")
    if subject not in eligible:
        raise SystemExit(f"{subject} is not in the post-CHARM cohort")
    qc = json.loads((output / "headmodel_qc" / f"{subject}_automated_qc.json").read_text(encoding="utf-8"))
    if qc.get("automated_qc_pass") is not True:
        raise SystemExit(f"Automated head-model QC did not pass for {subject}")
    if subject not in load_visual_qc_passes(output / "headmodel_qc" / "visual_qc.csv"):
        raise SystemExit(f"Visual head-model QC did not pass for {subject}")
    record, compact = compute_subject_fem_leadfield(config, subject, conductivity, cpus=cpus)
    target = output / "leadfields" / subject / f"skull_{conductivity:.4f}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compact, target / compact.name)
    shutil.copy2(record["solver_validation"]["log_npz_path"], target / f"{subject}_solver_log.npz")
    (target / f"{subject}_leadfield_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record, target / compact.name


def main() -> None:
    args = parse_args()
    config, project_root = load_config(args.config)
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = project_root / output
    if args.prepare_inputs:
        result = prepare_subject_forward_inputs(
            config["dataset"]["root"],
            args.subject,
            Path(config["cache_root"]) / "forward_inputs" / args.subject,
            float(config["head_model"]["source_triangle_offset_mm"]),
        )
        path = Path(result["source_mesh_path"]).with_name(f"{args.subject}_forward_input_manifest.json")
    elif args.qc_only:
        result, path = run_qc(config, output, args.subject)
    else:
        conductivity = args.skull_conductivity
        if conductivity is None:
            conductivity = float(config["head_model"]["nominal_skull_conductivity_s_per_m"])
        result, path = run_leadfield(config, output, args.subject, conductivity, args.cpus)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
