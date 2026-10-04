#!/usr/bin/env python3
"""Compute the finite-element lead fields of the cohort and summarise their validation.

Every eligible subject is solved at the nominal skull conductivity, and a
seeded subset of subjects at the two other conductivities.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from neural_observability.stimulation.cohort import (
    conductivity_sensitivity_subjects,
    load_config,
    load_eligible_runs,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stimulation_fem.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    return parser.parse_args()


def _directory(output: Path, subject: str, conductivity: float) -> Path:
    return output / "leadfields" / subject / f"skull_{conductivity:.4f}"


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def conductivity_comparison(subject: str, conductivity: float, nominal: dict, alternative: dict) -> dict[str, Any]:
    for name in ["eeg_channel_names", "source_labels", "source_positions_mm", "source_orientations", "source_depth_mm"]:
        if not np.array_equal(nominal[name], alternative[name]):
            raise RuntimeError(f"{subject}: source or sensor geometry differs across conductivities: {name}")
    reference = np.asarray(nominal["oriented_lead_field"], dtype=float)
    compared = np.asarray(alternative["oriented_lead_field"], dtype=float)
    reference_norm = np.linalg.norm(reference, axis=0)
    compared_norm = np.linalg.norm(compared, axis=0)
    cosine = np.sum(reference * compared, axis=0) / (reference_norm * compared_norm)
    squared_cosine = np.clip(cosine, -1.0, 1.0) ** 2
    norm_ratio = compared_norm / reference_norm
    return {
        "subject": subject,
        "skull_conductivity_s_per_m": float(conductivity),
        "source_count": int(reference.shape[1]),
        "squared_topographic_cosine": {
            "minimum": float(np.min(squared_cosine)),
            "p05": float(np.quantile(squared_cosine, 0.05)),
            "median": float(np.median(squared_cosine)),
            "mean": float(np.mean(squared_cosine)),
        },
        "source_column_norm_ratio_to_nominal": {
            "minimum": float(np.min(norm_ratio)),
            "p05": float(np.quantile(norm_ratio, 0.05)),
            "median": float(np.median(norm_ratio)),
            "p95": float(np.quantile(norm_ratio, 0.95)),
            "maximum": float(np.max(norm_ratio)),
        },
        "relative_frobenius_change_from_nominal": float(
            np.linalg.norm(compared - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny)
        ),
    }


def write_validation(config: dict, output: Path, sensitivity: list[str], tasks: list[tuple[str, float]]) -> Path:
    nominal = float(config["head_model"]["nominal_skull_conductivity_s_per_m"])
    rows, arrays = [], {}
    for subject, conductivity in tasks:
        directory = _directory(output, subject, conductivity)
        record = json.loads((directory / f"{subject}_leadfield_record.json").read_text(encoding="utf-8"))
        arrays[(subject, conductivity)] = _arrays(directory / f"{subject}_oriented_leadfield.npz")
        solver = record["solver_validation"]
        rows.append({
            "subject": subject,
            "skull_conductivity_s_per_m": f"{conductivity:.4f}",
            "is_nominal": conductivity == nominal,
            "eeg_channel_count": record["eeg_channel_count"],
            "source_count": record["source_count"],
            "solve_count": solver["solve_count"],
            "maximum_unpreconditioned_relative_residual": solver["unpreconditioned_relative_residual"]["maximum"],
            "reciprocity_relative_frobenius_error": solver["reciprocity_relative_frobenius_error"],
            "reciprocity_max_absolute_error_relative_to_diagonal":
                solver["reciprocity_max_absolute_error_relative_to_diagonal"],
            "compact_average_reference_relative_residual":
                record["compact_validation"]["average_reference_max_column_sum_relative"],
        })
    comparisons = [
        conductivity_comparison(subject, float(value), arrays[(subject, nominal)], arrays[(subject, float(value))])
        for subject in sensitivity
        for value in config["head_model"]["skull_conductivities_s_per_m"]
        if float(value) != nominal
    ]
    with (output / "leadfield_validation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "nominal_subject_count": len({subject for subject, _ in tasks}),
        "conductivity_sensitivity_subjects": sensitivity,
        "validated_task_count": len(rows),
        "maximum_unpreconditioned_relative_residual": max(
            row["maximum_unpreconditioned_relative_residual"] for row in rows),
        "conductivity_sensitivity_comparisons": comparisons,
    }
    path = output / "leadfield_validation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    config, project_root = load_config(args.config)
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = project_root / output
    subjects = sorted(load_eligible_runs(output / "post_charm_eligibility.csv", "post_charm_eligible"))
    if not subjects:
        raise SystemExit("The post-CHARM eligibility table contains no eligible subjects")
    sensitivity = conductivity_sensitivity_subjects(
        subjects, int(config["seed"]),
        min(int(config["head_model"]["conductivity_sensitivity_subject_count"]), len(subjects)),
    )
    nominal = float(config["head_model"]["nominal_skull_conductivity_s_per_m"])
    tasks = [(subject, nominal) for subject in subjects] + [
        (subject, float(value))
        for subject in sensitivity
        for value in config["head_model"]["skull_conductivities_s_per_m"]
        if float(value) != nominal
    ]
    logs = Path(config["cache_root"]) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def solve(subject: str, conductivity: float) -> str:
        if (_directory(output, subject, conductivity) / f"{subject}_oriented_leadfield.npz").is_file():
            return "present"
        log = logs / f"leadfield_{subject}_skull_{conductivity:.4f}.log"
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "stimulation" / "run_head_model.py"), subject,
                 "--config", str(args.config.resolve()), "--leadfield",
                 "--skull-conductivity", f"{conductivity:.4f}", "--cpus", str(args.cpus_per_worker)],
                stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        return "computed" if completed.returncode == 0 else f"failed (see {log})"

    status = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(solve, s, c): (s, c) for s, c in tasks}
        for future in as_completed(futures):
            subject, conductivity = futures[future]
            status[f"{subject} skull {conductivity:.4f}"] = future.result()
    print(json.dumps(status, indent=2, sort_keys=True))
    failed = [key for key, value in status.items() if value.startswith("failed")]
    if failed:
        raise SystemExit(f"{len(failed)} lead-field solves failed")
    print(f"Wrote {write_validation(config, output, sensitivity, tasks)}")


if __name__ == "__main__":
    main()
