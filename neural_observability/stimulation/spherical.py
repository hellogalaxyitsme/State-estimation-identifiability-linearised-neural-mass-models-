"""Known-site topography in the stimulation cohort with spherical head models.

Eligibility is decided from metadata and electrode geometry before any signal
is read. The head model is a four-layer sphere fitted to each patient's
digitised scalp electrodes in native T1 coordinates.
"""

from __future__ import annotations

import csv
import json
import platform
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import scipy
from mne.bem import fit_sphere_to_headshape

from neural_observability.stimulation.geometry import (
    infer_bipolar_orientation,
    parse_stimulation_description,
)


_RUN_RE = re.compile(r"_run-(?P<run>\d+)_epochs\.npy$")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and check the spherical-model configuration."""
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"seed", "dataset", "head_model", "eligibility", "windows_seconds", "inference", "output_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    model = config["head_model"]
    radii = model.get("relative_radii", [])
    if len(radii) != 4 or any(b <= a for a, b in zip(radii, radii[1:])):
        raise ValueError("head_model.relative_radii must be four increasing values")
    conductivities = model.get("skull_conductivities_s_per_m", [])
    if model.get("nominal_skull_conductivity_s_per_m") not in conductivities:
        raise ValueError("The nominal skull conductivity must be one of the listed values")
    windows = config["windows_seconds"]
    if set(windows) != {"primary_early", "secondary_propagated"}:
        raise ValueError("The primary_early and secondary_propagated windows are required")
    if any(len(window) != 2 or window[0] < 0 or window[1] <= window[0]
           for window in windows.values()):
        raise ValueError("Analysis windows must be increasing nonnegative pairs")
    return config, config_path.parent.parent


def _subject_paths(root: Path, subject: str) -> tuple[Path, Path]:
    derivative = root / "derivatives" / "epochs" / subject
    return derivative / "eeg", derivative / "ieeg"


def load_subject_geometry(
    root: str | Path, subject: str, sfreq: float = 1000.0
) -> dict[str, Any]:
    """Load coordinates and fit the sphere using EEG digitization only."""
    root = Path(root)
    eeg_dir, ieeg_dir = _subject_paths(root, subject)
    prefix = f"{subject}_task-ccepcoreg"
    eeg_rows = _read_tsv(eeg_dir / f"{prefix}_electrodes.tsv")
    channel_positions = {
        row["name"]: np.array([float(row[key]) for key in ("x", "y", "z")])
        / 1000.0
        for row in eeg_rows
    }
    coordinate_json = json.loads(
        (eeg_dir / f"{prefix}_coordsystem.json").read_text(encoding="utf-8")
    )
    if coordinate_json.get("EEGCoordinateSystem") != "T1w":
        raise ValueError(f"{subject}: EEG coordinates are not T1w")
    if coordinate_json.get("EEGCoordinateUnits") != "mm":
        raise ValueError(f"{subject}: EEG coordinates are not millimetres")
    landmarks = coordinate_json["AnatomicalLandmarkCoordinates"]
    montage = mne.channels.make_dig_montage(
        ch_pos=channel_positions,
        nasion=np.asarray(landmarks["NAS"], dtype=float) / 1000.0,
        lpa=np.asarray(landmarks["LPA"], dtype=float) / 1000.0,
        rpa=np.asarray(landmarks["RPA"], dtype=float) / 1000.0,
        coord_frame="head",
    )
    info = mne.create_info(list(channel_positions), sfreq, ch_types="eeg")
    info.set_montage(montage)
    radius, origin_head, _ = fit_sphere_to_headshape(
        info, dig_kinds="eeg", units="m", verbose=False
    )
    residuals = np.array(
        [np.linalg.norm(position - origin_head) - radius
         for position in channel_positions.values()]
    )

    seeg_path = ieeg_dir / f"{prefix}_space-T1w_electrodes.tsv"
    seeg_rows = _read_tsv(seeg_path)
    source_positions = {
        row["name"]: np.array([float(row[key]) for key in ("x", "y", "z")])
        for row in seeg_rows
    }
    return {
        "info": info,
        "channel_positions": channel_positions,
        "source_positions": source_positions,
        "sphere_radius_m": float(radius),
        "sphere_origin_m": np.asarray(origin_head, dtype=float),
        "sphere_rmse_m": float(np.sqrt(np.mean(residuals**2))),
        "sphere_max_abs_residual_m": float(np.max(np.abs(residuals))),
    }


def discover_subject_runs(root: str | Path, subject: str) -> list[dict[str, Any]]:
    """Discover runs without reading any EEG sample values."""
    root = Path(root)
    eeg_dir, _ = _subject_paths(root, subject)
    prefix = f"{subject}_task-ccepcoreg"
    records: list[dict[str, Any]] = []
    for array_path in sorted(eeg_dir.glob(f"{prefix}_run-*_epochs.npy")):
        match = _RUN_RE.search(array_path.name)
        if match is None:
            continue
        run = match.group("run")
        stem = eeg_dir / f"{prefix}_run-{run}"
        metadata = json.loads(
            stem.with_name(stem.name + "_epochs.json").read_text(encoding="utf-8")
        )
        stimulus = parse_stimulation_description(metadata["Description"])
        channel_rows = _read_tsv(stem.with_name(stem.name + "_channels.tsv"))
        good_channels = [
            row["name"] for row in channel_rows
            if row.get("status", "good").lower() == "good"
        ]
        epoch_rows = _read_tsv(stem.with_name(stem.name + "_epochs.tsv"))
        shape = np.load(array_path, mmap_mode="r").shape
        if len(shape) != 3:
            raise ValueError(f"{array_path}: expected trials x channels x samples")
        if shape[0] != len(epoch_rows) or shape[1] != len(channel_rows):
            raise ValueError(f"{array_path}: array/sidecar dimension mismatch")
        zero_times = {float(row["zero_time"]) for row in epoch_rows}
        if len(zero_times) != 1:
            raise ValueError(f"{array_path}: non-constant zero_time")
        records.append(
            {
                "subject": subject,
                "run": run,
                "array_path": array_path,
                "channel_names": [row["name"] for row in channel_rows],
                "good_channels": good_channels,
                "trial_count": int(shape[0]),
                "channel_count": int(shape[1]),
                "sample_count": int(shape[2]),
                "zero_time_seconds": float(next(iter(zero_times))),
                **stimulus,
            }
        )
    return records


def _in_range(value: float, bounds: Iterable[float]) -> bool:
    lower, upper = [float(item) for item in bounds]
    return lower <= value <= upper


def eligibility(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the anatomical and header-level criteria without reading signals."""
    root = Path(config["dataset"]["root"])
    subjects = sorted(
        path.name for path in (root / "derivatives" / "epochs").glob("sub-*")
        if path.is_dir()
    )
    gates = config["eligibility"]
    rows: list[dict[str, Any]] = []
    eligible_subjects = 0
    eligible_runs = 0
    for subject in subjects:
        try:
            geometry = load_subject_geometry(root, subject)
            subject_error = ""
        except Exception as error:
            geometry = None
            subject_error = f"{type(error).__name__}: {error}"
        runs = discover_subject_runs(root, subject)
        source_labels = set()
        subject_rows = []
        for run in runs:
            reasons: list[str] = []
            orientation_method = ""
            source_depth = float("nan")
            if subject_error:
                reasons.append("geometry_error")
            else:
                positions = geometry["source_positions"]
                if run["source_label"] not in positions:
                    reasons.append("source_not_found")
                else:
                    source_depth = float(
                        np.linalg.norm(
                            positions[run["source_label"]]
                            - geometry["sphere_origin_m"]
                        ) / geometry["sphere_radius_m"]
                    )
                    if source_depth > float(gates["maximum_source_radius_fraction"]):
                        reasons.append("source_outside_depth_gate")
                    try:
                        _, orientation_method = infer_bipolar_orientation(
                            run["source_label"], positions
                        )
                    except ValueError:
                        reasons.append("orientation_unavailable")
            if run["trial_count"] < int(gates["minimum_trials"]):
                reasons.append("too_few_trials")
            if len(run["good_channels"]) < int(gates["minimum_good_channels"]):
                reasons.append("too_few_good_channels")
            if run["sample_count"] < int(gates["minimum_samples"]):
                reasons.append("too_few_samples")
            if not reasons:
                source_labels.add(run["source_label"])
            subject_rows.append((run, reasons, source_depth, orientation_method))

        sphere_ok = geometry is not None
        if sphere_ok:
            sphere_ok = (
                _in_range(geometry["sphere_radius_m"], gates["sphere_radius_m"])
                and geometry["sphere_rmse_m"] <= float(gates["maximum_sphere_rmse_m"])
            )
        if not sphere_ok:
            for _, reasons, _, _ in subject_rows:
                reasons.append("sphere_qc_failed")
        if len(source_labels) < int(gates["minimum_unique_sources_per_subject"]):
            for _, reasons, _, _ in subject_rows:
                reasons.append("too_few_unique_sources")

        subject_any = False
        for run, reasons, source_depth, orientation_method in subject_rows:
            unique_reasons = sorted(set(reasons))
            eligible = not unique_reasons
            subject_any = subject_any or eligible
            eligible_runs += int(eligible)
            rows.append(
                {
                    "subject": subject,
                    "run": run["run"],
                    "source_label": run["source_label"],
                    "trial_count": run["trial_count"],
                    "good_channel_count": len(run["good_channels"]),
                    "sample_count": run["sample_count"],
                    "sphere_radius_m": "" if geometry is None else geometry["sphere_radius_m"],
                    "sphere_rmse_m": "" if geometry is None else geometry["sphere_rmse_m"],
                    "source_radius_fraction": source_depth,
                    "orientation_method": orientation_method,
                    "eligible": eligible,
                    "exclusion_reasons": ";".join(unique_reasons),
                    "technical_error": subject_error,
                }
            )
        eligible_subjects += int(subject_any)
    summary = {
        "discovered_subjects": len(subjects),
        "discovered_runs": len(rows),
        "eligible_subjects": eligible_subjects,
        "eligible_runs": eligible_runs,
        "excluded_runs": len(rows) - eligible_runs,
    }
    return rows, summary


def spatial_squared_cosine(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Average-reference both vectors and return their squared cosine."""
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    if prediction.shape != observation.shape or prediction.ndim != 1:
        raise ValueError("Topographies must be same-length vectors")
    prediction = prediction - prediction.mean()
    observation = observation - observation.mean()
    denominator = float((prediction @ prediction) * (observation @ observation))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float((prediction @ observation) ** 2 / denominator)


def subject_source_effect(
    score_matrix: np.ndarray, source_indices: np.ndarray
) -> tuple[float, np.ndarray]:
    """Matched-minus-other score, weighting responses equally within subject."""
    scores = np.asarray(score_matrix, dtype=float)
    labels = np.asarray(source_indices, dtype=int)
    if scores.ndim != 2 or labels.shape != (scores.shape[0],):
        raise ValueError("Invalid score matrix or source labels")
    if scores.shape[1] < 2:
        raise ValueError("At least two unique source sites are required")
    effects = []
    for row, label in zip(scores, labels):
        matched = row[label]
        others = np.delete(row, label)
        effects.append(matched - np.mean(others))
    values = np.asarray(effects, dtype=float)
    return float(np.mean(values)), values


def hierarchical_label_permutation_test(
    score_matrices: list[np.ndarray],
    source_indices: list[np.ndarray],
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """Subject-balanced, within-subject random-label test with plus-one p-value."""
    if len(score_matrices) != len(source_indices) or not score_matrices:
        raise ValueError("Score matrices and source labels must be nonempty and paired")
    observed_effects = np.array(
        [subject_source_effect(scores, labels)[0]
         for scores, labels in zip(score_matrices, source_indices)]
    )
    observed = float(np.mean(observed_effects))
    rng = np.random.default_rng(seed)
    exceedances = 0
    null_sum = 0.0
    null_sum_squares = 0.0
    for _ in range(int(permutations)):
        permuted_effects = [
            subject_source_effect(scores, rng.permutation(labels))[0]
            for scores, labels in zip(score_matrices, source_indices)
        ]
        statistic = float(np.mean(permuted_effects))
        exceedances += int(statistic >= observed)
        null_sum += statistic
        null_sum_squares += statistic * statistic
    count = float(permutations)
    null_mean = null_sum / count
    null_variance = max(0.0, null_sum_squares / count - null_mean**2)
    return {
        "subject_count": float(len(observed_effects)),
        "observed_mean_subject_effect": observed,
        "median_subject_effect": float(np.median(observed_effects)),
        "positive_subject_fraction": float(np.mean(observed_effects > 0)),
        "null_mean": null_mean,
        "null_sd": float(np.sqrt(null_variance)),
        "permutation_count": count,
        "one_sided_p_value": float((exceedances + 1) / (permutations + 1)),
    }


def make_subject_topographies(
    geometry: dict[str, Any], config: dict[str, Any], source_labels: list[str]
) -> dict[float, dict[str, np.ndarray]]:
    """Compute fixed-orientation scalp topographies for every conductivity/site."""
    model = config["head_model"]
    radius = geometry["sphere_radius_m"]
    origin = geometry["sphere_origin_m"]
    output: dict[float, dict[str, np.ndarray]] = {}
    for skull_conductivity in model["skull_conductivities_s_per_m"]:
        skull_conductivity = float(skull_conductivity)
        sphere = mne.make_sphere_model(
            r0=origin,
            head_radius=radius,
            relative_radii=tuple(float(value) for value in model["relative_radii"]),
            sigmas=(
                float(model["brain_conductivity_s_per_m"]),
                float(model["csf_conductivity_s_per_m"]),
                skull_conductivity,
                float(model["scalp_conductivity_s_per_m"]),
            ),
            verbose=False,
        )
        output[skull_conductivity] = {}
        for source_label in source_labels:
            position = geometry["source_positions"][source_label]
            orientation, _ = infer_bipolar_orientation(
                source_label, geometry["source_positions"]
            )
            dipole = mne.Dipole(
                times=np.array([0.0]),
                pos=position[np.newaxis, :],
                amplitude=np.array([1e-9]),
                ori=orientation[np.newaxis, :],
                gof=np.array([1.0]),
                name=source_label,
            )
            forward, _ = mne.make_forward_dipole(
                dipole, sphere, geometry["info"], trans=None, verbose=False
            )
            topography = np.asarray(forward["sol"]["data"][:, 0], dtype=float)
            if topography.shape != (len(geometry["info"]["ch_names"]),):
                raise RuntimeError(f"Unexpected forward shape for {source_label}")
            if not np.all(np.isfinite(topography)) or np.linalg.norm(topography) == 0:
                raise RuntimeError(f"Invalid forward topography for {source_label}")
            output[skull_conductivity][source_label] = topography
    return output


def score_run_topographies(
    run: dict[str, Any],
    predictions: dict[str, np.ndarray],
    geometry_channel_names: list[str],
    window_seconds: list[float],
    sfreq: float,
) -> np.ndarray:
    """Score all candidate sites against one run's trial-mean scalp response."""
    data = np.asarray(np.load(run["array_path"], mmap_mode="r"), dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{run['subject']} run {run['run']}: non-finite EEG samples")
    channel_lookup = {name: index for index, name in enumerate(run["channel_names"])}
    geometry_lookup = {name: index for index, name in enumerate(geometry_channel_names)}
    good_names = run["good_channels"]
    if not set(good_names).issubset(channel_lookup) or not set(good_names).issubset(geometry_lookup):
        raise ValueError(f"{run['subject']} run {run['run']}: channel name mismatch")
    data_indices = [channel_lookup[name] for name in good_names]
    geometry_indices = [geometry_lookup[name] for name in good_names]
    evoked = data.mean(axis=0)[data_indices]
    times = np.arange(run["sample_count"], dtype=float) / sfreq - run["zero_time_seconds"]
    start, stop = [float(value) for value in window_seconds]
    time_mask = (times >= start) & (times < stop)
    if not np.any(time_mask):
        raise ValueError(f"{run['subject']} run {run['run']}: empty time window")
    observations = evoked[:, time_mask]
    observations = observations - observations.mean(axis=0, keepdims=True)
    observation_norm_squared = np.sum(observations**2, axis=0)
    if np.any(observation_norm_squared <= np.finfo(float).tiny):
        raise ValueError(f"{run['subject']} run {run['run']}: zero-energy topography")

    scores = []
    for source_label in predictions:
        prediction = predictions[source_label][geometry_indices]
        prediction = prediction - prediction.mean()
        prediction_norm_squared = float(prediction @ prediction)
        numerator = prediction @ observations
        squared_cosines = numerator**2 / (
            prediction_norm_squared * observation_norm_squared
        )
        scores.append(float(np.mean(squared_cosines)))
    return np.asarray(scores, dtype=float)


def _conductivity_label(value: float) -> str:
    return f"{float(value):.4f}"


def run_spherical_analysis(
    config: dict[str, Any], project_root: Path, max_subjects: int | None = None
) -> tuple[dict[str, Any], Path]:
    """Score every run against every stimulated site and test the known-site advantage."""
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = (project_root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    eligibility_rows, eligibility_summary = eligibility(config)
    _write_csv(output / "eligibility.csv", eligibility_rows)
    eligible_pairs = {
        (row["subject"], str(row["run"]))
        for row in eligibility_rows if bool(row["eligible"])
    }
    subjects = sorted({subject for subject, _ in eligible_pairs})
    if max_subjects is not None:
        if max_subjects < 1:
            raise ValueError("max_subjects must be positive")
        subjects = subjects[:max_subjects]
    root = Path(config["dataset"]["root"])
    sfreq = float(config["dataset"]["sampling_frequency_hz"])
    conductivities = [float(value) for value in config["head_model"]["skull_conductivities_s_per_m"]]
    nominal = float(config["head_model"]["nominal_skull_conductivity_s_per_m"])
    windows = config["windows_seconds"]

    score_collections: dict[tuple[float, str], list[np.ndarray]] = defaultdict(list)
    label_collections: dict[tuple[float, str], list[np.ndarray]] = defaultdict(list)
    run_score_rows: list[dict[str, Any]] = []
    candidate_score_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    analyzed_subjects: list[str] = []
    analyzed_runs = 0

    for subject in subjects:
        geometry = load_subject_geometry(root, subject, sfreq=sfreq)
        runs = [
            run for run in discover_subject_runs(root, subject)
            if (subject, str(run["run"])) in eligible_pairs
        ]
        source_labels = sorted({run["source_label"] for run in runs})
        if len(source_labels) < 2:
            continue
        source_to_index = {source: index for index, source in enumerate(source_labels)}
        topographies = make_subject_topographies(geometry, config, source_labels)
        subject_scores: dict[tuple[float, str], list[np.ndarray]] = defaultdict(list)
        labels = np.array([source_to_index[run["source_label"]] for run in runs], dtype=int)

        for run in runs:
            for conductivity in conductivities:
                for window_name, window in windows.items():
                    scores = score_run_topographies(
                        run,
                        topographies[conductivity],
                        geometry["info"]["ch_names"],
                        window,
                        sfreq,
                    )
                    subject_scores[(conductivity, window_name)].append(scores)
                    for candidate_label, candidate_score in zip(source_labels, scores):
                        candidate_score_rows.append(
                            {
                                "subject": subject,
                                "run": run["run"],
                                "response_source_label": run["source_label"],
                                "candidate_source_label": candidate_label,
                                "candidate_is_matched": candidate_label == run["source_label"],
                                "skull_conductivity_s_per_m": conductivity,
                                "window": window_name,
                                "spatial_squared_cosine": float(candidate_score),
                            }
                        )
                    matched_index = source_to_index[run["source_label"]]
                    matched = float(scores[matched_index])
                    other = np.delete(scores, matched_index)
                    rank_percentile = float(
                        (np.sum(other < matched) + 0.5 * np.sum(other == matched))
                        / len(other)
                    )
                    run_score_rows.append(
                        {
                            "subject": subject,
                            "run": run["run"],
                            "source_label": run["source_label"],
                            "source_count": len(source_labels),
                            "skull_conductivity_s_per_m": conductivity,
                            "window": window_name,
                            "matched_score": matched,
                            "mean_wrong_source_score": float(np.mean(other)),
                            "matched_minus_wrong": float(matched - np.mean(other)),
                            "matched_rank_percentile": rank_percentile,
                        }
                    )

        for conductivity in conductivities:
            for window_name in windows:
                matrix = np.vstack(subject_scores[(conductivity, window_name)])
                key = (conductivity, window_name)
                score_collections[key].append(matrix)
                label_collections[key].append(labels.copy())
                effect, run_effects = subject_source_effect(matrix, labels)
                matched = matrix[np.arange(len(labels)), labels]
                ranks = []
                for row, label in zip(matrix, labels):
                    other = np.delete(row, label)
                    ranks.append(
                        (np.sum(other < row[label]) + 0.5 * np.sum(other == row[label]))
                        / len(other)
                    )
                subject_rows.append(
                    {
                        "subject": subject,
                        "skull_conductivity_s_per_m": conductivity,
                        "window": window_name,
                        "run_count": len(runs),
                        "unique_source_count": len(source_labels),
                        "mean_matched_score": float(np.mean(matched)),
                        "mean_matched_minus_wrong": effect,
                        "median_run_effect": float(np.median(run_effects)),
                        "mean_matched_rank_percentile": float(np.mean(ranks)),
                    }
                )

        nominal_topographies = topographies[nominal]
        for conductivity in conductivities:
            if conductivity == nominal:
                continue
            for source_label in source_labels:
                robustness_rows.append(
                    {
                        "subject": subject,
                        "source_label": source_label,
                        "comparison_conductivity_s_per_m": conductivity,
                        "nominal_conductivity_s_per_m": nominal,
                        "spatial_squared_cosine_to_nominal": spatial_squared_cosine(
                            nominal_topographies[source_label],
                            topographies[conductivity][source_label],
                        ),
                    }
                )
        analyzed_subjects.append(subject)
        analyzed_runs += len(runs)

    if not analyzed_subjects:
        raise RuntimeError("No eligible subjects were analysed")
    primary_results: dict[str, dict[str, float]] = {}
    sensitivity_results: dict[str, dict[str, float]] = {}
    for conductivity in conductivities:
        for window_name in windows:
            key = (conductivity, window_name)
            label = f"sigma_{_conductivity_label(conductivity)}__{window_name}"
            test = hierarchical_label_permutation_test(
                score_collections[key],
                label_collections[key],
                permutations=int(config["inference"]["permutations"]),
                seed=int(config["seed"]),
            )
            if conductivity == nominal and window_name == "primary_early":
                primary_results[label] = test
            else:
                sensitivity_results[label] = test

    results = {
        "cohort": {
            **eligibility_summary,
            "analysed_subjects": len(analyzed_subjects),
            "analysed_runs": analyzed_runs,
            "subject_limit": max_subjects,
        },
        "primary": primary_results,
        "sensitivity": sensitivity_results,
        "conductivity_robustness": {
            "comparison_count": len(robustness_rows),
            "minimum_spatial_squared_cosine_to_nominal": float(
                min(row["spatial_squared_cosine_to_nominal"] for row in robustness_rows)
            ),
            "median_spatial_squared_cosine_to_nominal": float(
                np.median([row["spatial_squared_cosine_to_nominal"] for row in robustness_rows])
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mne": mne.__version__,
            "platform": platform.platform(),
        },
    }
    _write_csv(output / "run_scores.csv", run_score_rows)
    _write_csv(output / "candidate_scores.csv", candidate_score_rows)
    _write_csv(output / "subject_metrics.csv", subject_rows)
    _write_csv(output / "conductivity_robustness.csv", robustness_rows)
    path = output / "results.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results, path


def write_eligibility(
    config: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], Path]:
    """Write the eligibility table and its summary."""
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = (project_root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = eligibility(config)
    _write_csv(output / "eligibility.csv", rows)
    path = output / "eligibility_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary, path
