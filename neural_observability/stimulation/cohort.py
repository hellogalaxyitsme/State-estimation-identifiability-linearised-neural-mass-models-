"""Cohort selection, signal processing and inference for the finite-element analysis.

Eligibility is decided from metadata, MRI headers and head-model QC before any
signal is read. SimNIBS is imported only by the head-model layer, so eligibility
and the statistical functions do not depend on that installation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from neural_observability.stimulation.geometry import (
    infer_bipolar_orientation,
    parse_stimulation_description,
)
from neural_observability.head_models.referencing import average_reference_basis


_RUN_RE = re.compile(r"_run-(?P<run>\d+)_epochs\.npy$")
_PAIR_RE = re.compile(r"^(?P<shaft>.*?)(?P<a>\d+)-(?P<b>\d+)$")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("CSV rows do not have a common schema")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and check the finite-element cohort configuration."""
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "seed",
        "dataset",
        "software",
        "cache_root",
        "head_model",
        "head_model_qc",
        "fem_validation",
        "eligibility",
        "windows_seconds",
        "dynamic_proxy",
        "detectability",
        "inference",
        "output_dir",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    model = config["head_model"]
    conductivities = [float(value) for value in model["skull_conductivities_s_per_m"]]
    if any(value <= 0 for value in conductivities):
        raise ValueError("Skull conductivities must be positive")
    if float(model["nominal_skull_conductivity_s_per_m"]) not in conductivities:
        raise ValueError("The nominal skull conductivity must be one of the listed values")
    windows = config["windows_seconds"]
    if set(windows) != {"baseline", "static_early", "dynamic_propagated"}:
        raise ValueError("The baseline, static_early and dynamic_propagated windows are required")
    for name, window in windows.items():
        if len(window) != 2 or float(window[1]) <= float(window[0]):
            raise ValueError(f"Invalid {name} window")
    if float(windows["baseline"][1]) > 0:
        raise ValueError("The baseline window cannot extend beyond stimulation")
    if float(windows["static_early"][0]) < 0:
        raise ValueError("The early window must be nonnegative")
    if len(config["inference"].get("tests", [])) != 3:
        raise ValueError("Three named tests are required")
    return config, config_path.parent.parent


def bipolar_shaft(label: str) -> str:
    """Return the shaft prefix of an adjacent bipolar label."""
    match = _PAIR_RE.match(label)
    if match is None or int(match.group("b")) != int(match.group("a")) + 1:
        raise ValueError(f"Cannot parse adjacent bipolar label: {label!r}")
    return match.group("shaft")


def conductivity_sensitivity_subjects(
    subjects: Iterable[str], seed: int, count: int
) -> list[str]:
    """Select a deterministic, outcome-independent conductivity subset."""
    unique = sorted(set(subjects))
    if count < 1 or count > len(unique):
        raise ValueError("Sensitivity count must be between one and subject count")
    ranked = sorted(
        unique,
        key=lambda subject: hashlib.sha256(f"{seed}:{subject}".encode()).hexdigest(),
    )
    return ranked[:count]


def source_triangle_vertices(
    point_mm: Sequence[float], offset_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create a tiny deterministic triangle whose first vertex is the source."""
    point = np.asarray(point_mm, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_mm must be a finite three-vector")
    if not np.isfinite(offset_mm) or offset_mm <= 0:
        raise ValueError("offset_mm must be positive")
    vertices = np.vstack(
        [point, point + [offset_mm, 0.0, 0.0], point + [0.0, offset_mm, 0.0]]
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return vertices, faces


def _array_header(path: Path) -> tuple[int, int, int]:
    shape = np.load(path, mmap_mode="r").shape
    if len(shape) != 3:
        raise ValueError(f"{path}: expected trials x channels x samples")
    return tuple(int(value) for value in shape)


def _read_run_header(modality_dir: Path, subject: str, run: str) -> dict[str, Any]:
    prefix = f"{subject}_task-ccepcoreg_run-{run}"
    array_path = modality_dir / f"{prefix}_epochs.npy"
    channel_rows = _read_tsv(modality_dir / f"{prefix}_channels.tsv")
    epoch_rows = _read_tsv(modality_dir / f"{prefix}_epochs.tsv")
    metadata = json.loads(
        (modality_dir / f"{prefix}_epochs.json").read_text(encoding="utf-8")
    )
    shape = _array_header(array_path)
    if shape[0] != len(epoch_rows) or shape[1] != len(channel_rows):
        raise ValueError(f"{array_path}: array/sidecar dimension mismatch")
    zero_times = {float(row["zero_time"]) for row in epoch_rows}
    if len(zero_times) != 1:
        raise ValueError(f"{array_path}: non-constant zero_time")
    return {
        "array_path": array_path,
        "shape": shape,
        "channel_rows": channel_rows,
        "channel_names": [row["name"] for row in channel_rows],
        "good_channels": [
            row["name"]
            for row in channel_rows
            if row.get("status", "good").lower() == "good"
        ],
        "trial_count": shape[0],
        "channel_count": shape[1],
        "sample_count": shape[2],
        "zero_time_seconds": float(next(iter(zero_times))),
        "metadata": metadata,
    }


def _subject_geometry_headers(root: Path, subject: str) -> dict[str, Any]:
    prefix = f"{subject}_task-ccepcoreg"
    subject_root = root / "derivatives" / "epochs" / subject
    eeg_dir = subject_root / "eeg"
    ieeg_dir = subject_root / "ieeg"
    eeg_coordinates = json.loads(
        (eeg_dir / f"{prefix}_coordsystem.json").read_text(encoding="utf-8")
    )
    if eeg_coordinates.get("EEGCoordinateSystem") != "T1w":
        raise ValueError("EEG coordinates are not in T1w space")
    if eeg_coordinates.get("EEGCoordinateUnits") != "mm":
        raise ValueError("EEG coordinates are not in millimetres")
    eeg_rows = _read_tsv(eeg_dir / f"{prefix}_electrodes.tsv")
    eeg_positions_mm = {
        row["name"]: np.asarray([float(row[key]) for key in ("x", "y", "z")])
        for row in eeg_rows
    }
    if eeg_coordinates.get("AnatomicalLandmarkCoordinateSystem") != "T1w":
        raise ValueError("Anatomical landmarks are not in T1w space")
    if eeg_coordinates.get("AnatomicalLandmarkCoordinateUnits") != "mm":
        raise ValueError("Anatomical landmarks are not in millimetres")
    landmark_source = eeg_coordinates.get("AnatomicalLandmarkCoordinates", {})
    required_landmarks = {"LPA", "NAS", "RPA"}
    if set(landmark_source) != required_landmarks:
        raise ValueError("Exactly LPA, NAS, and RPA anatomical landmarks are required")
    anatomical_landmarks_mm = {
        name: np.asarray(landmark_source[name], dtype=float)
        for name in sorted(required_landmarks)
    }
    ieeg_coordinates = json.loads(
        (ieeg_dir / f"{prefix}_space-T1w_coordsystem.json").read_text(
            encoding="utf-8"
        )
    )
    if ieeg_coordinates.get("iEEGCoordinateUnits") != "m":
        raise ValueError("iEEG coordinates are not in metres")
    if "Native MRI space" not in ieeg_coordinates.get(
        "iEEGCoordinateSystemDescription", ""
    ):
        raise ValueError("iEEG coordinates are not identified as native MRI space")
    ieeg_rows = _read_tsv(ieeg_dir / f"{prefix}_space-T1w_electrodes.tsv")
    ieeg_positions_m = {
        row["name"]: np.asarray([float(row[key]) for key in ("x", "y", "z")])
        for row in ieeg_rows
    }
    if not eeg_positions_mm or not ieeg_positions_m:
        raise ValueError("Empty electrode coordinate table")
    if not all(np.all(np.isfinite(value)) for value in eeg_positions_mm.values()):
        raise ValueError("Non-finite EEG electrode coordinate")
    if not all(
        value.shape == (3,) and np.all(np.isfinite(value))
        for value in anatomical_landmarks_mm.values()
    ):
        raise ValueError("Invalid anatomical landmark coordinate")
    if not all(np.all(np.isfinite(value)) for value in ieeg_positions_m.values()):
        raise ValueError("Non-finite iEEG electrode coordinate")

    t1_path = root / subject / "anat" / f"{subject}_T1w.nii"
    image = nib.load(t1_path, mmap=False)
    header = image.header
    qform, qform_code = header.get_qform(coded=True)
    return {
        "eeg_dir": eeg_dir,
        "ieeg_dir": ieeg_dir,
        "eeg_positions_mm": eeg_positions_mm,
        "anatomical_landmarks_mm": anatomical_landmarks_mm,
        "ieeg_positions_m": ieeg_positions_m,
        "t1_path": t1_path,
        "t1_shape": tuple(int(value) for value in image.shape),
        "t1_zooms_mm": tuple(float(value) for value in header.get_zooms()[:3]),
        "qform_code": int(qform_code),
        "qform_finite": bool(qform is not None and np.all(np.isfinite(qform))),
        "qform_determinant": (
            float(np.linalg.det(qform[:3, :3])) if qform is not None else float("nan")
        ),
    }


def pre_charm_eligibility(
    config: dict[str, Any], max_subjects: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the criteria that precede segmentation, without reading signals."""
    root = Path(config["dataset"]["root"])
    subjects = sorted(
        path.name
        for path in root.glob("sub-*")
        if path.is_dir() and (path / "anat" / f"{path.name}_T1w.nii").is_file()
    )
    if max_subjects is not None:
        if max_subjects < 1:
            raise ValueError("max_subjects must be positive")
        subjects = subjects[:max_subjects]
    gates = config["eligibility"]
    required_shape = tuple(int(value) for value in gates["required_t1_shape"])
    required_zooms = tuple(float(value) for value in gates["required_t1_voxel_size_mm"])
    rows: list[dict[str, Any]] = []

    for subject in subjects:
        try:
            geometry = _subject_geometry_headers(root, subject)
            geometry_error = ""
        except Exception as error:
            geometry = None
            geometry_error = f"{type(error).__name__}: {error}"
        if geometry is None:
            run_names = sorted(
                match.group("run")
                for path in (root / "derivatives" / "epochs" / subject / "eeg").glob(
                    f"{subject}_task-ccepcoreg_run-*_epochs.npy"
                )
                if (match := _RUN_RE.search(path.name)) is not None
            )
        else:
            run_names = sorted(
                match.group("run")
                for path in geometry["eeg_dir"].glob(
                    f"{subject}_task-ccepcoreg_run-*_epochs.npy"
                )
                if (match := _RUN_RE.search(path.name)) is not None
            )
        subject_rows: list[dict[str, Any]] = []
        preliminarily_eligible_sites: set[str] = set()

        for run in run_names:
            reasons: list[str] = []
            run_error = ""
            source_label = ""
            eeg: dict[str, Any] | None = None
            ieeg: dict[str, Any] | None = None
            retained_sources: list[str] = []
            if geometry_error:
                reasons.append("geometry_or_t1_header_error")
            else:
                if geometry["t1_shape"] != required_shape:
                    reasons.append("t1_shape_mismatch")
                if not np.allclose(geometry["t1_zooms_mm"], required_zooms, atol=1e-6):
                    reasons.append("t1_voxel_size_mismatch")
                if (
                    geometry["qform_code"] <= 0
                    or not geometry["qform_finite"]
                    or not np.isfinite(geometry["qform_determinant"])
                    or abs(geometry["qform_determinant"]) <= np.finfo(float).eps
                ):
                    reasons.append("invalid_t1_qform")
                try:
                    eeg = _read_run_header(geometry["eeg_dir"], subject, run)
                    ieeg = _read_run_header(geometry["ieeg_dir"], subject, run)
                    description = eeg["metadata"].get("Description", "")
                    source_label = parse_stimulation_description(description)["source_label"]
                    ieeg_description = ieeg["metadata"].get("Description", description)
                    if ieeg_description and ieeg_description != description:
                        reasons.append("modality_description_mismatch")
                except Exception as error:
                    reasons.append("paired_run_header_error")
                    run_error = f"{type(error).__name__}: {error}"

            if eeg is not None and ieeg is not None and geometry is not None:
                if eeg["sample_count"] != ieeg["sample_count"]:
                    reasons.append("paired_sample_count_mismatch")
                if not np.isclose(eeg["zero_time_seconds"], ieeg["zero_time_seconds"]):
                    reasons.append("paired_zero_time_mismatch")
                if eeg["trial_count"] < int(gates["minimum_eeg_trials"]):
                    reasons.append("too_few_eeg_trials")
                if ieeg["trial_count"] < int(gates["minimum_ieeg_trials"]):
                    reasons.append("too_few_ieeg_trials")
                if eeg["sample_count"] < int(gates["minimum_samples"]):
                    reasons.append("too_few_samples")
                good_eeg = set(eeg["good_channels"])
                if len(good_eeg) < int(gates["minimum_good_eeg_channels"]):
                    reasons.append("too_few_good_eeg_channels")
                if not good_eeg.issubset(geometry["eeg_positions_mm"]):
                    reasons.append("eeg_coordinate_name_mismatch")
                try:
                    stimulation_shaft = bipolar_shaft(source_label)
                except ValueError:
                    stimulation_shaft = ""
                    reasons.append("invalid_stimulation_label")
                positions = geometry["ieeg_positions_m"]
                if source_label not in positions:
                    reasons.append("stimulation_source_coordinate_missing")
                else:
                    try:
                        infer_bipolar_orientation(source_label, positions)
                    except ValueError:
                        reasons.append("stimulation_orientation_unavailable")
                for name in ieeg["good_channels"]:
                    try:
                        shaft = bipolar_shaft(name)
                    except ValueError:
                        continue
                    if shaft == stimulation_shaft or name not in positions:
                        continue
                    try:
                        infer_bipolar_orientation(name, positions)
                    except ValueError:
                        continue
                    retained_sources.append(name)
                if len(retained_sources) < int(
                    gates["minimum_good_ieeg_sources_after_stimulated_shaft_exclusion"]
                ):
                    reasons.append("too_few_retained_ieeg_sources")

            unique_reasons = sorted(set(reasons))
            preliminary = not unique_reasons
            if preliminary:
                preliminarily_eligible_sites.add(source_label)
            subject_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "source_label": source_label,
                    "eeg_trial_count": "" if eeg is None else eeg["trial_count"],
                    "ieeg_trial_count": "" if ieeg is None else ieeg["trial_count"],
                    "cross_modal_trial_row_pairing": "unavailable_no_released_trial_identifier",
                    "eeg_channel_count": "" if eeg is None else eeg["channel_count"],
                    "good_eeg_channel_count": "" if eeg is None else len(eeg["good_channels"]),
                    "ieeg_channel_count": "" if ieeg is None else ieeg["channel_count"],
                    "good_ieeg_channel_count": "" if ieeg is None else len(ieeg["good_channels"]),
                    "retained_ieeg_source_count": len(retained_sources),
                    "sample_count": "" if eeg is None else eeg["sample_count"],
                    "zero_time_seconds": "" if eeg is None else eeg["zero_time_seconds"],
                    "t1_shape": "" if geometry is None else "x".join(map(str, geometry["t1_shape"])),
                    "t1_voxel_size_mm": "" if geometry is None else "x".join(
                        f"{value:g}" for value in geometry["t1_zooms_mm"]
                    ),
                    "t1_qform_code": "" if geometry is None else geometry["qform_code"],
                    "pre_charm_eligible": preliminary,
                    "exclusion_reasons": ";".join(unique_reasons),
                    "technical_error": geometry_error or run_error,
                }
            )

        if len(preliminarily_eligible_sites) < int(
            gates["minimum_unique_stimulation_sites_per_subject"]
        ):
            for row in subject_rows:
                reasons = set(filter(None, str(row["exclusion_reasons"]).split(";")))
                reasons.add("too_few_unique_stimulation_sites")
                row["exclusion_reasons"] = ";".join(sorted(reasons))
                row["pre_charm_eligible"] = False
        rows.extend(subject_rows)

    eligible_rows = [row for row in rows if bool(row["pre_charm_eligible"])]
    eligible_subjects = sorted({str(row["subject"]) for row in eligible_rows})
    sensitivity_count = min(
        int(config["head_model"]["conductivity_sensitivity_subject_count"]),
        len(eligible_subjects),
    )
    sensitivity = (
        conductivity_sensitivity_subjects(
            eligible_subjects, int(config["seed"]), sensitivity_count
        )
        if sensitivity_count
        else []
    )
    summary = {
        "discovered_subjects": len(subjects),
        "discovered_paired_candidate_runs": len(rows),
        "pre_charm_eligible_subjects": len(eligible_subjects),
        "pre_charm_eligible_runs": len(eligible_rows),
        "excluded_runs": len(rows) - len(eligible_rows),
        "conductivity_sensitivity_subjects": sensitivity,
    }
    return rows, summary


def write_pre_charm_eligibility(
    config: dict[str, Any], project_root: Path, max_subjects: int | None = None
) -> tuple[dict[str, Any], Path]:
    """Write the eligibility table that precedes head-model construction."""
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = (project_root / output).resolve()
    if max_subjects is not None:
        output = output.with_name(f"{output.name}_test_{max_subjects}")
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = pre_charm_eligibility(config, max_subjects=max_subjects)
    _write_csv(output / "pre_charm_eligibility.csv", rows)
    result_path = output / "pre_charm_summary.json"
    result_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary, result_path


def load_eligible_runs(
    ledger_path: str | Path,
    eligibility_field: str = "pre_charm_eligible",
) -> dict[str, list[str]]:
    """Load the eligible runs of each subject from an eligibility table."""
    path = Path(ledger_path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row[eligibility_field].lower() == "true":
            selected[row["subject"]].append(row["run"])
    return {subject: sorted(runs) for subject, runs in sorted(selected.items())}


def load_visual_qc_passes(path: str | Path) -> set[str]:
    """Return the subjects whose head model passed visual review."""
    path = Path(path)
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["subject"]
        for row in rows
        if row.get("visual_qc_status", "").strip().lower() == "pass"
    }


def post_charm_eligibility(
    config: dict[str, Any], project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the head-model QC and source-distance criteria, without reading signals."""
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = (project_root / output).resolve()
    pre_path = output / "pre_charm_eligibility.csv"
    with pre_path.open(encoding="utf-8", newline="") as stream:
        pre_rows = list(csv.DictReader(stream))
    root = Path(config["dataset"]["root"])
    maximum_distance = float(
        config["eligibility"]["maximum_source_to_brain_distance_mm"]
    )
    rows: list[dict[str, Any]] = []
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    visual_path = output / "headmodel_qc" / "visual_qc.csv"
    visual_decisions: dict[str, str] = {}
    visual_passes = load_visual_qc_passes(visual_path)
    if visual_path.is_file():
        with visual_path.open(encoding="utf-8", newline="") as stream:
            for decision in csv.DictReader(stream):
                visual_decisions[decision["subject"]] = decision.get(
                    "visual_qc_status", ""
                ).strip().lower()

    for pre in pre_rows:
        subject = pre["subject"]
        run = pre["run"]
        reasons = set(filter(None, pre["exclusion_reasons"].split(";")))
        source_distance: float | str = ""
        retained_count: int | str = ""
        qc_pass = False
        recorded_visual_status = visual_decisions.get(subject, "pending")
        visual_status = (
            "pass"
            if subject in visual_passes
            else ("fail" if recorded_visual_status == "fail" else "pending")
        )
        qc_path = output / "headmodel_qc" / f"{subject}_automated_qc.json"
        if pre["pre_charm_eligible"].lower() == "true":
            if not qc_path.is_file():
                reasons.add("automated_headmodel_qc_missing")
                distance_map: dict[str, float] = {}
            else:
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
                qc_pass = qc.get("automated_qc_pass") is True
                distance_map = {
                    str(name): float(value)
                    for name, value in qc.get("source_distance_mm_by_label", {}).items()
                }
                if not qc_pass:
                    reasons.add("automated_headmodel_qc_failed")
            if visual_status == "fail":
                reasons.add("visual_headmodel_qc_failed")
            elif visual_status != "pass":
                reasons.add("visual_headmodel_qc_missing")
            source_label = pre["source_label"]
            if source_label not in distance_map:
                reasons.add("stimulation_source_distance_missing")
            else:
                source_distance = distance_map[source_label]
                if float(source_distance) > maximum_distance:
                    reasons.add("stimulation_source_outside_brain_distance_gate")

            try:
                geometry = _subject_geometry_headers(root, subject)
                header = _read_run_header(geometry["ieeg_dir"], subject, run)
                stimulated_shaft = bipolar_shaft(source_label)
                retained = []
                for name in header["good_channels"]:
                    try:
                        shaft = bipolar_shaft(name)
                    except ValueError:
                        continue
                    if shaft == stimulated_shaft or name not in distance_map:
                        continue
                    if distance_map[name] > maximum_distance:
                        continue
                    try:
                        infer_bipolar_orientation(name, geometry["ieeg_positions_m"])
                    except ValueError:
                        continue
                    retained.append(name)
                retained_count = len(retained)
                if retained_count < int(
                    config["eligibility"][
                        "minimum_good_ieeg_sources_after_stimulated_shaft_exclusion"
                    ]
                ):
                    reasons.add("too_few_post_charm_ieeg_sources")
            except Exception as error:
                reasons.add("post_charm_source_error")
                technical_error = f"{type(error).__name__}: {error}"
            else:
                technical_error = ""
        else:
            technical_error = pre["technical_error"]

        row = {
            **pre,
            "automated_headmodel_qc_pass": qc_pass,
            "visual_headmodel_qc_status": visual_status,
            "stimulation_source_to_brain_distance_mm": source_distance,
            "post_charm_retained_ieeg_source_count": retained_count,
            "post_charm_eligible": not reasons,
            "post_charm_exclusion_reasons": ";".join(sorted(reasons)),
            "post_charm_technical_error": technical_error,
        }
        rows.append(row)
        by_subject[subject].append(row)

    minimum_sites = int(
        config["eligibility"]["minimum_unique_stimulation_sites_per_subject"]
    )
    for subject_rows in by_subject.values():
        eligible_sites = {
            row["source_label"]
            for row in subject_rows
            if bool(row["post_charm_eligible"])
        }
        if len(eligible_sites) < minimum_sites:
            for row in subject_rows:
                reasons = set(
                    filter(None, str(row["post_charm_exclusion_reasons"]).split(";"))
                )
                reasons.add("too_few_post_charm_stimulation_sites")
                row["post_charm_exclusion_reasons"] = ";".join(sorted(reasons))
                row["post_charm_eligible"] = False

    eligible_rows = [row for row in rows if bool(row["post_charm_eligible"])]
    eligible_subjects = sorted({row["subject"] for row in eligible_rows})
    preeligible_subjects = {
        row["subject"]
        for row in pre_rows
        if row["pre_charm_eligible"].lower() == "true"
    }
    sensitivity_count = min(
        int(config["head_model"]["conductivity_sensitivity_subject_count"]),
        len(eligible_subjects),
    )
    summary = {
        "pre_charm_run_count": len(pre_rows),
        "post_charm_eligible_runs": len(eligible_rows),
        "post_charm_eligible_subjects": len(eligible_subjects),
        "post_charm_excluded_runs": len(pre_rows) - len(eligible_rows),
        "conductivity_sensitivity_subjects": (
            conductivity_sensitivity_subjects(
                eligible_subjects, int(config["seed"]), sensitivity_count
            )
            if sensitivity_count
            else []
        ),
        "visual_qc_subject_counts": {
            status: sum(
                visual_decisions.get(subject, "pending") == status
                for subject in preeligible_subjects
            )
            for status in ["pass", "fail", "pending"]
        },
    }
    return rows, summary


def write_post_charm_eligibility(
    config: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], Path]:
    """Write the eligibility table that follows head-model construction and QC."""
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = (project_root / output).resolve()
    rows, summary = post_charm_eligibility(config, project_root)
    _write_csv(output / "post_charm_eligibility.csv", rows)
    path = output / "post_charm_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary, path


def half_open_time_mask(times: np.ndarray, window: Sequence[float]) -> np.ndarray:
    """Return the half-open ``[start, stop)`` sample mask."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a finite vector")
    if len(window) != 2 or float(window[1]) <= float(window[0]):
        raise ValueError("window must contain increasing endpoints")
    mask = (times >= float(window[0])) & (times < float(window[1]))
    if not np.any(mask):
        raise ValueError("The requested time window contains no samples")
    return mask


def baseline_channel_statistics(
    trials: np.ndarray, times: np.ndarray, baseline_window: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return channel means and pooled trial/time baseline sample SDs."""
    values = np.asarray(trials, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("trials must have shape trials x channels x samples")
    if values.shape[2] != len(times) or not np.all(np.isfinite(values)):
        raise ValueError("Signal dimensions are invalid or samples are non-finite")
    mask = half_open_time_mask(times, baseline_window)
    pooled = values[:, :, mask].transpose(1, 0, 2).reshape(values.shape[1], -1)
    if pooled.shape[1] < 2:
        raise ValueError("At least two pooled baseline observations are required")
    return pooled.mean(axis=1), pooled.std(axis=1, ddof=1)


def baseline_standardized_trial_mean(
    trials: np.ndarray,
    times: np.ndarray,
    baseline_window: Sequence[float],
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale a trial-mean iEEG trajectory using baseline only."""
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    means, scales = baseline_channel_statistics(trials, times, baseline_window)
    effective_scales = np.maximum(scales, float(scale_floor))
    trajectory = (np.asarray(trials, dtype=float).mean(axis=0) - means[:, None])
    trajectory /= effective_scales[:, None]
    return trajectory, means, effective_scales


def mean_spatial_squared_cosine(
    prediction: np.ndarray, observation: np.ndarray
) -> float:
    """Mean average-reference squared cosine over nondegenerate time samples."""
    predicted = np.asarray(prediction, dtype=float)
    observed = np.asarray(observation, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("Prediction and observation must be sensor x time matrices")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ValueError("Prediction and observation must be finite")
    predicted = predicted - predicted.mean(axis=0, keepdims=True)
    observed = observed - observed.mean(axis=0, keepdims=True)
    numerator = np.sum(predicted * observed, axis=0) ** 2
    denominator = np.sum(predicted**2, axis=0) * np.sum(observed**2, axis=0)
    valid = denominator > np.finfo(float).tiny
    if not np.any(valid):
        raise ValueError("No nondegenerate spatial time samples")
    return float(np.mean(numerator[valid] / denominator[valid]))


def dynamic_proxy_trajectory(lead_field: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """Project the partial standardized iEEG trajectory to scalp sensors."""
    lead = np.asarray(lead_field, dtype=float)
    state = np.asarray(trajectory, dtype=float)
    if lead.ndim != 2 or state.ndim != 2 or lead.shape[1] != state.shape[0]:
        raise ValueError("lead_field and trajectory dimensions are incompatible")
    if not np.all(np.isfinite(lead)) or not np.all(np.isfinite(state)):
        raise ValueError("lead_field and trajectory must be finite")
    return lead @ state


def loaded_reference_covariance(
    trials: np.ndarray,
    times: np.ndarray,
    baseline_window: Sequence[float],
    diagonal_loading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a loaded baseline covariance in independent reference coordinates."""
    values = np.asarray(trials, dtype=float)
    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] != len(times):
        raise ValueError("Invalid scalp trial dimensions")
    if not np.all(np.isfinite(values)):
        raise ValueError("Scalp trials contain non-finite samples")
    if not 0 < diagonal_loading < 1:
        raise ValueError("diagonal_loading must lie strictly between zero and one")
    mask = half_open_time_mask(times, baseline_window)
    basis = average_reference_basis(values.shape[1])
    referenced = np.einsum("sc,tcu->tsu", basis.T, values[:, :, mask])
    pooled = referenced.transpose(1, 0, 2).reshape(values.shape[1] - 1, -1)
    covariance = np.cov(pooled, bias=False)
    covariance = np.atleast_2d(covariance)
    dimension = covariance.shape[0]
    scale = float(np.trace(covariance) / dimension)
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("Baseline covariance has zero or invalid scale")
    loaded = covariance + diagonal_loading * scale * np.eye(dimension)
    np.linalg.cholesky(loaded)
    return loaded, basis


def mahalanobis_time_energy(
    signal: np.ndarray, covariance: np.ndarray, basis: np.ndarray
) -> float:
    """Return mean time-resolved squared energy in the loaded noise metric."""
    values = np.asarray(signal, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if values.ndim != 2 or basis.shape[0] != values.shape[0]:
        raise ValueError("signal and reference basis dimensions are incompatible")
    referenced = basis.T @ values
    if covariance.shape != (referenced.shape[0], referenced.shape[0]):
        raise ValueError("covariance and referenced signal dimensions differ")
    factor = cho_factor(covariance, lower=True, check_finite=True)
    solved = cho_solve(factor, referenced, check_finite=True)
    energies = np.sum(referenced * solved, axis=0)
    return float(np.mean(np.maximum(energies, 0.0)))


def whitened_scalp_detectability_target(
    trials: np.ndarray,
    times: np.ndarray,
    baseline_window: Sequence[float],
    response_window: Sequence[float],
    diagonal_loading: float,
) -> tuple[float, dict[str, Any]]:
    """Compute the log1p trial-count-scaled whitened response energy."""
    values = np.asarray(trials, dtype=float)
    covariance, basis = loaded_reference_covariance(
        values, times, baseline_window, diagonal_loading
    )
    mask = half_open_time_mask(times, response_window)
    evoked = values.mean(axis=0)[:, mask]
    raw_energy = mahalanobis_time_energy(evoked, covariance, basis)
    scaled_energy = values.shape[0] * raw_energy
    return float(np.log1p(scaled_energy)), {
        "trial_count": int(values.shape[0]),
        "raw_whitened_energy": raw_energy,
        "trial_count_scaled_whitened_energy": float(scaled_energy),
        "independent_channel_count": int(values.shape[1] - 1),
    }


def fixed_ridge_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float
) -> np.ndarray:
    """Fit standardized ridge regression and predict without test-fold leakage."""
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training and test feature matrices are incompatible")
    if train_y.shape != (train_x.shape[0],) or train_x.shape[0] < 2:
        raise ValueError("Training target dimensions are invalid")
    if not np.all(np.isfinite(train_x)) or not np.all(np.isfinite(train_y)) or not np.all(np.isfinite(test_x)):
        raise ValueError("Ridge inputs must be finite")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    x = (train_x - mean) / scale
    test = (test_x - mean) / scale
    intercept = float(train_y.mean())
    centered_y = train_y - intercept
    gram = x.T @ x / len(x)
    rhs = x.T @ centered_y / len(x)
    coefficients = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), rhs)
    return intercept + test @ coefficients


def leave_one_subject_out_ridge(
    features: np.ndarray,
    outcomes: np.ndarray,
    subjects: Sequence[str],
    alpha: float,
) -> np.ndarray:
    """Return one prediction per run from strictly subject-held-out folds."""
    features = np.asarray(features, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    labels = np.asarray(subjects, dtype=str)
    if features.ndim != 2 or outcomes.shape != (len(features),) or labels.shape != (len(features),):
        raise ValueError("LOSO inputs have incompatible dimensions")
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("LOSO requires at least two subjects")
    predictions = np.full(len(outcomes), np.nan)
    for subject in unique:
        test = labels == subject
        train = ~test
        predictions[test] = fixed_ridge_fit_predict(
            features[train], outcomes[train], features[test], alpha
        )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("LOSO prediction did not cover every observation")
    return predictions


def subject_mean_losses(
    outcomes: np.ndarray, predictions: np.ndarray, subjects: Sequence[str]
) -> dict[str, float]:
    """Compute equal-subject MSEs from run-level held-out predictions."""
    outcomes = np.asarray(outcomes, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(subjects, dtype=str)
    if outcomes.shape != predictions.shape or outcomes.shape != labels.shape:
        raise ValueError("Loss inputs have incompatible dimensions")
    return {
        subject: float(np.mean((outcomes[labels == subject] - predictions[labels == subject]) ** 2))
        for subject in sorted(set(labels))
    }


def one_sided_sign_permutation_test(
    subject_differences: Sequence[float], permutations: int, seed: int
) -> dict[str, float]:
    """Test whether an equal-subject mean contrast is greater than zero."""
    differences = np.asarray(subject_differences, dtype=float)
    if differences.ndim != 1 or len(differences) < 2 or not np.all(np.isfinite(differences)):
        raise ValueError("At least two finite subject differences are required")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    exceedances = 0
    null_sum = 0.0
    null_sum_squares = 0.0
    completed = 0
    batch_size = 4096
    while completed < permutations:
        count = min(batch_size, permutations - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(differences)))
        values = np.mean(signs * differences, axis=1)
        exceedances += int(np.sum(values >= observed))
        null_sum += float(np.sum(values))
        null_sum_squares += float(values @ values)
        completed += count
    null_mean = null_sum / permutations
    null_variance = max(0.0, null_sum_squares / permutations - null_mean**2)
    return {
        "subject_count": float(len(differences)),
        "observed_mean_subject_difference": observed,
        "median_subject_difference": float(np.median(differences)),
        "positive_subject_fraction": float(np.mean(differences > 0)),
        "permutation_count": float(permutations),
        "null_mean": float(null_mean),
        "null_sd": float(np.sqrt(null_variance)),
        "one_sided_p_value": float((exceedances + 1) / (permutations + 1)),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return multiplicity-adjusted p-values using Holm's step-down method."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) < 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a nonempty vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * sorted_values
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted
