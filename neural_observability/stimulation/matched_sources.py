"""Known-site topography against matched wrong sources in the finite-element cohort.

Each stimulated site is compared with ten wrong sources matched within patient
on standardised depth, log lead-field norm and absolute radiality, and with the
nearest other stimulated site. Source characteristics and matches use the head
models only; the endpoint is the 5--20 ms squared spatial cosine of the
finite-element analysis.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from neural_observability.stimulation.cohort import (
    _read_run_header,
    _write_csv,
    half_open_time_mask,
    holm_adjust,
    load_eligible_runs,
)
from neural_observability.stimulation.fem_analysis import (
    _brain_distance_eligible_sources,
    _load_leadfield,
    _times,
)
from neural_observability.stimulation.geometry import parse_stimulation_description
from neural_observability.stimulation.source_matching import (
    FEATURE_NAMES,
    characterize_sources,
    hierarchical_lookup_permutation_tests,
    score_constant_topographies,
    select_feature_matches,
    select_spatial_neighbours,
    subject_bootstrap_mean_interval,
)

TESTS = (
    "true_minus_feature_matched_wrong_source_topography",
    "true_minus_nearest_spatial_wrong_stimulation_site_topography",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and check the matched-source configuration."""
    config_path = Path(path).expanduser().resolve()
    config = _read_json(config_path)
    required = {"seed", "fem_config", "cohort", "fem_candidate_scores", "feature_matched_control",
                "endpoint", "inference", "consistency_tolerance", "output_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if tuple(config["inference"].get("tests", ())) != TESTS:
        raise ValueError(f"The tests must be {TESTS}")
    return config, config_path.parent.parent


def _fem_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    return _read_json(project_root / config["fem_config"])


def _eligible_ledger_rows(config: dict[str, Any], project_root: Path) -> list[dict[str, str]]:
    rows = _read_csv(project_root / config["cohort"]["eligibility_ledger"])
    field = config["cohort"]["eligibility_field"]
    return [row for row in rows if row[field].strip().lower() == "true"]


def _scalp_center(fem_config: dict[str, Any], project_root: Path, subject: str) -> np.ndarray:
    qc = _read_json(project_root / fem_config["output_dir"] / "headmodel_qc" / f"{subject}_automated_qc.json")
    if qc.get("automated_qc_pass") is not True:
        raise RuntimeError(f"{subject}: automated head-model QC did not pass")
    center = qc.get("electrode_projection_qc", {}).get("estimated_scalp_center_mm")
    if center is None:
        raise RuntimeError(f"{subject}: estimated scalp centre is missing")
    return np.asarray(center, dtype=float)


def select_matched_sources(config: dict[str, Any], project_root: Path, output: Path) -> None:
    """Characterise every eligible source and select the matched controls."""
    fem_config = _fem_config(config, project_root)
    eligible_runs = load_eligible_runs(
        project_root / config["cohort"]["eligibility_ledger"],
        config["cohort"]["eligibility_field"],
    )
    stimulation_by_subject: dict[str, set[str]] = defaultdict(set)
    for row in _eligible_ledger_rows(config, project_root):
        stimulation_by_subject[row["subject"]].add(row["source_label"])
    match_count = int(config["feature_matched_control"]["match_count"])

    source_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for subject in eligible_runs:
        lead_data = _load_leadfield(fem_config, subject)
        features = characterize_sources(
            lead_data["oriented_lead_field"],
            lead_data["source_labels"],
            lead_data["source_positions_mm"],
            lead_data["source_orientations"],
            lead_data["source_depth_mm"],
            _scalp_center(fem_config, project_root, subject),
            _brain_distance_eligible_sources(fem_config, subject),
        )
        labels = list(features["labels"])
        raw = np.asarray(features["raw_features"], dtype=float)
        standardized = np.asarray(features["standardized_features"], dtype=float)
        positions = np.asarray(features["positions_mm"], dtype=float)
        column_norms = np.asarray(features["column_norms"], dtype=float)
        indices = np.asarray(features["leadfield_indices"], dtype=int)
        stimulation = sorted(stimulation_by_subject[subject])
        feature_matches = select_feature_matches(labels, standardized, stimulation, match_count)
        spatial_matches = select_spatial_neighbours(labels, positions, stimulation)
        for row_index, label in enumerate(labels):
            source_rows.append(
                {
                    "subject": subject,
                    "source_label": label,
                    "leadfield_column_index": int(indices[row_index]),
                    "source_x_mm": float(positions[row_index, 0]),
                    "source_y_mm": float(positions[row_index, 1]),
                    "source_z_mm": float(positions[row_index, 2]),
                    "source_depth_mm": float(raw[row_index, 0]),
                    "lead_field_column_l2_norm": float(column_norms[row_index]),
                    "log_lead_field_column_norm": float(raw[row_index, 1]),
                    "absolute_dipole_radiality": float(raw[row_index, 2]),
                    "z_source_depth": float(standardized[row_index, 0]),
                    "z_log_lead_field_column_norm": float(standardized[row_index, 1]),
                    "z_absolute_dipole_radiality": float(standardized[row_index, 2]),
                    "is_observed_stimulation_site": label in stimulation,
                }
            )
        for target in stimulation:
            for rank, (matched, distance) in enumerate(feature_matches[target], start=1):
                match_rows.append(
                    {
                        "subject": subject,
                        "target_source_label": target,
                        "control_type": "feature_matched_wrong_source",
                        "match_rank": rank,
                        "matched_source_label": matched,
                        "match_distance": distance,
                    }
                )
            spatial_label, spatial_distance = spatial_matches[target]
            match_rows.append(
                {
                    "subject": subject,
                    "target_source_label": target,
                    "control_type": "nearest_spatial_wrong_stimulation_site",
                    "match_rank": 1,
                    "matched_source_label": spatial_label,
                    "match_distance": spatial_distance,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "source_features.csv", source_rows)
    _write_csv(output / "match_manifest.csv", match_rows)


def _matching_balance(
    feature_rows: list[dict[str, str]], match_rows: list[dict[str, str]]
) -> dict[str, Any]:
    feature_lookup = {
        (row["subject"], row["source_label"]): np.asarray(
            [
                float(row["z_source_depth"]),
                float(row["z_log_lead_field_column_norm"]),
                float(row["z_absolute_dipole_radiality"]),
            ]
        )
        for row in feature_rows
    }
    differences = []
    feature_distances = []
    spatial_distances = []
    for row in match_rows:
        if row["control_type"] == "feature_matched_wrong_source":
            differences.append(
                np.abs(
                    feature_lookup[(row["subject"], row["target_source_label"])]
                    - feature_lookup[(row["subject"], row["matched_source_label"])]
                )
            )
            feature_distances.append(float(row["match_distance"]))
        else:
            spatial_distances.append(float(row["match_distance"]))
    difference = np.vstack(differences)
    return {
        "mean_absolute_standardized_difference_by_feature": {
            name: float(value) for name, value in zip(FEATURE_NAMES, difference.mean(axis=0))
        },
        "feature_match_distance": {
            "median": float(np.median(feature_distances)),
            "p95": float(np.quantile(feature_distances, 0.95)),
            "maximum": float(np.max(feature_distances)),
        },
        "spatial_neighbour_distance_mm": {
            "median": float(np.median(spatial_distances)),
            "p95": float(np.quantile(spatial_distances, 0.95)),
            "maximum": float(np.max(spatial_distances)),
        },
    }


def run_matched_sources(
    config: dict[str, Any],
    project_root: Path,
    *,
    max_subjects: int | None = None,
    permutation_count: int | None = None,
    bootstrap_count: int | None = None,
) -> tuple[dict[str, Any], Path]:
    """Score every run against its true, matched and nearest-site sources and test the contrasts."""
    fem_config = _fem_config(config, project_root)
    eligible = load_eligible_runs(
        project_root / config["cohort"]["eligibility_ledger"],
        config["cohort"]["eligibility_field"],
    )
    if max_subjects is not None:
        eligible = dict(list(eligible.items())[: int(max_subjects)])
    permutations = int(permutation_count or config["inference"]["permutations"])
    bootstraps = int(bootstrap_count or config["inference"]["bootstrap_resamples"])
    match_count = int(config["feature_matched_control"]["match_count"])

    output = project_root / config["output_dir"]
    if not (output / "source_features.csv").is_file() or not (output / "match_manifest.csv").is_file():
        select_matched_sources(config, project_root, output)
    selected_subjects = set(eligible)
    feature_rows = [row for row in _read_csv(output / "source_features.csv") if row["subject"] in selected_subjects]
    match_rows = [row for row in _read_csv(output / "match_manifest.csv") if row["subject"] in selected_subjects]
    feature_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    match_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        feature_by_subject[row["subject"]].append(row)
    for row in match_rows:
        match_by_subject[row["subject"]].append(row)

    fem_scores = {
        (row["subject"], row["run"], row["candidate_source_label"]): (
            row["response_source_label"],
            float(row["known_site_early_spatial_squared_cosine"]),
        )
        for row in _read_csv(project_root / config["fem_candidate_scores"])
        if row["subject"] in selected_subjects
    }

    all_candidate_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    feature_lookup_matrices: list[np.ndarray] = []
    spatial_lookup_matrices: list[np.ndarray] = []
    observed_label_vectors: list[np.ndarray] = []
    reproduction_differences: list[float] = []

    root = Path(fem_config["dataset"]["root"])
    sfreq = float(fem_config["dataset"]["sampling_frequency_hz"])
    window = config["endpoint"]["window_seconds_half_open"]
    for subject, runs in eligible.items():
        lead_data = _load_leadfield(fem_config, subject)
        lead = np.asarray(lead_data["oriented_lead_field"], dtype=float)
        eeg_names = [str(value) for value in lead_data["eeg_channel_names"]]
        lead_source_labels = [str(value) for value in lead_data["source_labels"]]
        sources = sorted(feature_by_subject[subject], key=lambda row: int(row["leadfield_column_index"]))
        eligible_labels = [row["source_label"] for row in sources]
        eligible_indices = np.asarray([int(row["leadfield_column_index"]) for row in sources], dtype=int)
        if any(lead_source_labels[index] != label for index, label in zip(eligible_indices, eligible_labels)):
            raise RuntimeError(f"{subject}: source table does not match the lead-field columns")
        eligible_lookup = {label: index for index, label in enumerate(eligible_labels)}

        feature_matches: dict[str, list[str]] = defaultdict(list)
        spatial_matches: dict[str, str] = {}
        for row in sorted(
            match_by_subject[subject],
            key=lambda value: (
                value["target_source_label"], value["control_type"], int(value["match_rank"])
            ),
        ):
            if row["control_type"] == "feature_matched_wrong_source":
                feature_matches[row["target_source_label"]].append(row["matched_source_label"])
            else:
                spatial_matches[row["target_source_label"]] = row["matched_source_label"]
        stimulation_labels = sorted(feature_matches)
        if set(stimulation_labels) != set(spatial_matches):
            raise RuntimeError(f"{subject}: matched controls have inconsistent target sites")
        if any(len(set(value)) != match_count for value in feature_matches.values()):
            raise RuntimeError(f"{subject}: a feature-matched set does not contain {match_count} sources")
        stimulation_lookup = {label: index for index, label in enumerate(stimulation_labels)}
        subject_feature_lookup = np.empty((len(runs), len(stimulation_labels)), dtype=float)
        subject_spatial_lookup = np.empty_like(subject_feature_lookup)
        subject_observed_labels = np.empty(len(runs), dtype=int)
        subject_run_feature_effects = []
        subject_run_spatial_effects = []
        eeg_dir = root / "derivatives" / "epochs" / subject / "eeg"

        for run_index, run in enumerate(runs):
            header = _read_run_header(eeg_dir, subject, run)
            stimulus = parse_stimulation_description(header["metadata"]["Description"])
            observed_label = stimulus["source_label"]
            if observed_label not in stimulation_lookup:
                raise RuntimeError(f"{subject} run {run}: stimulated site is not among the matched targets")
            eeg_lookup = {name: index for index, name in enumerate(header["channel_names"])}
            lead_eeg_lookup = {name: index for index, name in enumerate(eeg_names)}
            good_eeg = [
                name for name in header["good_channels"] if name in eeg_lookup and name in lead_eeg_lookup
            ]
            eeg_indices = [eeg_lookup[name] for name in good_eeg]
            lead_eeg_indices = [lead_eeg_lookup[name] for name in good_eeg]
            trials = np.asarray(np.load(header["array_path"], mmap_mode="r"), dtype=float)[
                :, eeg_indices, :
            ]
            if fem_config["eligibility"]["require_all_finite_signal_samples"] and not np.all(
                np.isfinite(trials)
            ):
                raise ValueError(f"{subject} run {run}: non-finite EEG samples")
            times = _times(header, sfreq)
            early_mask = half_open_time_mask(times, window)
            evoked = trials.mean(axis=0)[:, early_mask]
            scores = score_constant_topographies(
                lead[np.ix_(lead_eeg_indices, eligible_indices)], evoked
            )
            for source_label, score in zip(eligible_labels, scores):
                all_candidate_rows.append(
                    {
                        "subject": subject,
                        "run": run,
                        "response_source_label": observed_label,
                        "candidate_source_label": source_label,
                        "candidate_is_true": source_label == observed_label,
                        "candidate_is_feature_match_for_observed": source_label
                        in feature_matches[observed_label],
                        "candidate_is_spatial_match_for_observed": source_label
                        == spatial_matches[observed_label],
                        "early_spatial_squared_cosine": float(score),
                    }
                )
            for candidate_index, candidate in enumerate(stimulation_labels):
                true_score = scores[eligible_lookup[candidate]]
                feature_indices = [eligible_lookup[label] for label in feature_matches[candidate]]
                spatial_index = eligible_lookup[spatial_matches[candidate]]
                subject_feature_lookup[run_index, candidate_index] = float(
                    true_score - np.mean(scores[feature_indices])
                )
                subject_spatial_lookup[run_index, candidate_index] = float(
                    true_score - scores[spatial_index]
                )
                key = (subject, run, candidate)
                if key not in fem_scores:
                    raise RuntimeError(f"Missing finite-element candidate score: {key}")
                response, fem_score = fem_scores[key]
                if response != observed_label:
                    raise RuntimeError(f"Stimulated site differs from the finite-element analysis: {key}")
                reproduction_differences.append(abs(float(true_score) - fem_score))

            observed_index = stimulation_lookup[observed_label]
            subject_observed_labels[run_index] = observed_index
            true_score = float(scores[eligible_lookup[observed_label]])
            feature_control_score = float(
                np.mean([scores[eligible_lookup[label]] for label in feature_matches[observed_label]])
            )
            spatial_control_score = float(scores[eligible_lookup[spatial_matches[observed_label]]])
            feature_effect = true_score - feature_control_score
            spatial_effect = true_score - spatial_control_score
            subject_run_feature_effects.append(feature_effect)
            subject_run_spatial_effects.append(spatial_effect)
            run_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "true_source_label": observed_label,
                    "good_eeg_channel_count": len(good_eeg),
                    "true_source_score": true_score,
                    "feature_matched_control_source_labels": ";".join(feature_matches[observed_label]),
                    "feature_matched_control_mean_score": feature_control_score,
                    "true_minus_feature_matched_score": feature_effect,
                    "spatial_neighbour_source_label": spatial_matches[observed_label],
                    "spatial_neighbour_score": spatial_control_score,
                    "true_minus_spatial_neighbour_score": spatial_effect,
                }
            )
        feature_lookup_matrices.append(subject_feature_lookup)
        spatial_lookup_matrices.append(subject_spatial_lookup)
        observed_label_vectors.append(subject_observed_labels)
        subject_rows.append(
            {
                "subject": subject,
                "run_count": len(runs),
                "unique_stimulation_site_count": len(stimulation_labels),
                "eligible_source_count": len(eligible_labels),
                "mean_true_minus_feature_matched_score": float(np.mean(subject_run_feature_effects)),
                "mean_true_minus_spatial_neighbour_score": float(np.mean(subject_run_spatial_effects)),
            }
        )

    maximum_difference = float(max(reproduction_differences, default=np.inf))
    if maximum_difference > float(config["consistency_tolerance"]):
        raise RuntimeError(
            "True-site scores differ from the finite-element analysis by "
            f"{maximum_difference:.3g}"
        )

    permutation = hierarchical_lookup_permutation_tests(
        {
            TESTS[0]: feature_lookup_matrices,
            TESTS[1]: spatial_lookup_matrices,
        },
        observed_label_vectors,
        permutations,
        int(config["seed"]),
    )
    adjusted = holm_adjust([permutation[name]["one_sided_p_value"] for name in TESTS])
    tests: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(TESTS):
        subject_effects = np.asarray(permutation[name].pop("subject_effects"), dtype=float)
        tests[name] = {
            **permutation[name],
            "subject_bootstrap_mean_95_ci": subject_bootstrap_mean_interval(
                subject_effects,
                bootstraps,
                int(config["seed"]) + 10 + index,
                tuple(config["inference"]["confidence_interval_quantiles"]),
            ),
            "holm_adjusted_p_value": float(adjusted[index]),
        }

    _write_csv(output / "candidate_scores.csv", all_candidate_rows)
    _write_csv(output / "run_effects.csv", run_rows)
    _write_csv(output / "subject_effects.csv", subject_rows)
    result = {
        "subject_count": len(eligible),
        "run_count": sum(map(len, eligible.values())),
        "eligible_source_count": len(feature_rows),
        "unique_stimulation_site_count": sum(
            row["unique_stimulation_site_count"] for row in subject_rows
        ),
        "tests": tests,
        "matching_balance": _matching_balance(feature_rows, match_rows),
        "maximum_difference_from_finite_element_scores": maximum_difference,
        "permutation_count": permutations,
        "bootstrap_resample_count": bootstraps,
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, result_path
