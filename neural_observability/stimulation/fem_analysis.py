"""Finite-element analysis of the stimulation cohort.

Known-site topography in 5--20 ms, projected intracranial trajectories in
20--100 ms and held-out prediction of the scalp response amplitude, all with
individual finite-element head models.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neural_observability.statistics import subject_bootstrap_mean_interval
from neural_observability.stimulation.spherical import (
    hierarchical_label_permutation_test,
    subject_source_effect,
)
from neural_observability.stimulation.geometry import parse_stimulation_description
from neural_observability.stimulation.cohort import (
    _read_run_header,
    _write_csv,
    baseline_channel_statistics,
    baseline_standardized_trial_mean,
    bipolar_shaft,
    half_open_time_mask,
    holm_adjust,
    leave_one_subject_out_ridge,
    load_eligible_runs,
    load_visual_qc_passes,
    loaded_reference_covariance,
    mahalanobis_time_energy,
    mean_spatial_squared_cosine,
    one_sided_sign_permutation_test,
    subject_mean_losses,
    whitened_scalp_detectability_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def verify_headmodel_qc(
    config: dict[str, Any],
    project_root: Path,
    eligible_subjects: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Require automated and visual head-model QC for every eligible subject."""
    output = _resolve(config["output_dir"], project_root)
    qc_dir = output / "headmodel_qc"
    visual_passes = load_visual_qc_passes(qc_dir / "visual_qc.csv")
    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for subject in eligible_subjects:
        path = qc_dir / f"{subject}_automated_qc.json"
        if not path.is_file():
            failures.append(f"{subject}:automated_qc_missing")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("automated_qc_pass") is not True:
            failures.append(f"{subject}:automated_qc_failed")
        if subject not in visual_passes:
            failures.append(f"{subject}:visual_qc_not_passed")
        results[subject] = result
    if failures:
        raise RuntimeError("Head-model QC is incomplete: " + ", ".join(failures))
    return results


def _resolve(value: str | Path, project_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root or PROJECT_ROOT) / path


def _leadfield_root(config: dict[str, Any]) -> Path:
    if "leadfield_root" in config:
        return _resolve(config["leadfield_root"])
    return _resolve(config["output_dir"]) / "leadfields"


def _load_leadfield(config: dict[str, Any], subject: str) -> dict[str, Any]:
    conductivity = float(config["head_model"]["nominal_skull_conductivity_s_per_m"])
    path = (
        _leadfield_root(config)
        / subject
        / f"skull_{conductivity:.4f}"
        / f"{subject}_oriented_leadfield.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing nominal FEM lead field: {path}")
    with np.load(path) as archive:
        result = {name: archive[name] for name in archive.files}
    lead = np.asarray(result["oriented_lead_field"], dtype=float)
    if lead.shape != (
        len(result["eeg_channel_names"]),
        len(result["source_labels"]),
    ):
        raise ValueError(f"{subject}: inconsistent compact lead-field dimensions")
    result["path"] = path
    return result


def _brain_distance_eligible_sources(
    config: dict[str, Any], subject: str
) -> set[str]:
    """Return the sources within the MRI brain-distance criterion."""
    qc_path = _resolve(config["output_dir"]) / "headmodel_qc" / f"{subject}_automated_qc.json"
    if not qc_path.is_file():
        raise FileNotFoundError(f"Missing automated head-model QC: {qc_path}")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc.get("automated_qc_pass") is not True:
        raise RuntimeError(f"{subject}: automated head-model QC did not pass")
    distances = qc.get("source_distance_mm_by_label", {})
    maximum = float(config["eligibility"]["maximum_source_to_brain_distance_mm"])
    eligible = {
        str(label)
        for label, distance in distances.items()
        if np.isfinite(float(distance)) and float(distance) <= maximum
    }
    if not eligible:
        raise RuntimeError(f"{subject}: no sources pass the MRI brain-distance criterion")
    return eligible


def _times(header: dict[str, Any], sfreq: float) -> np.ndarray:
    return (
        np.arange(header["sample_count"], dtype=float) / sfreq
        - float(header["zero_time_seconds"])
    )


def _retained_ieeg_names(
    header: dict[str, Any], source_label: str, available_sources: set[str]
) -> list[str]:
    stimulated_shaft = bipolar_shaft(source_label)
    retained = []
    for name in header["good_channels"]:
        try:
            shaft = bipolar_shaft(name)
        except ValueError:
            continue
        if shaft != stimulated_shaft and name in available_sources:
            retained.append(name)
    return retained


def geometry_permutation_scores(
    lead_field: np.ndarray,
    trajectory: np.ndarray,
    observation: np.ndarray,
    permutations: int,
    seed: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Shuffle trajectory-to-source geometry and score each negative control."""
    lead = np.asarray(lead_field, dtype=float)
    state = np.asarray(trajectory, dtype=float)
    observed = np.asarray(observation, dtype=float)
    if lead.ndim != 2 or state.ndim != 2 or observed.ndim != 2:
        raise ValueError("Geometry-permutation inputs must be matrices")
    if lead.shape[1] != state.shape[0] or lead.shape[0] != observed.shape[0]:
        raise ValueError("Geometry-permutation dimensions are incompatible")
    if state.shape[1] != observed.shape[1] or permutations < 1 or batch_size < 1:
        raise ValueError("Invalid time or permutation dimensions")
    rng = np.random.default_rng(seed)
    index = np.vstack([rng.permutation(state.shape[0]) for _ in range(permutations)])
    output = np.empty(permutations, dtype=float)
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_lead = torch.as_tensor(lead, dtype=torch.float64, device=device)
        torch_state = torch.as_tensor(state, dtype=torch.float64, device=device)
        torch_observed = torch.as_tensor(observed, dtype=torch.float64, device=device)
        torch_observed = torch_observed - torch_observed.mean(dim=0, keepdim=True)
        observed_norm = torch.sum(torch_observed**2, dim=0)
        for start in range(0, permutations, batch_size):
            stop = min(permutations, start + batch_size)
            permutation_index = torch.as_tensor(index[start:stop], device=device)
            permuted_state = torch_state[permutation_index]
            prediction = torch.einsum("cs,bst->bct", torch_lead, permuted_state)
            prediction -= prediction.mean(dim=1, keepdim=True)
            numerator = torch.sum(prediction * torch_observed[None, :, :], dim=1) ** 2
            denominator = torch.sum(prediction**2, dim=1) * observed_norm[None, :]
            scores = torch.where(
                denominator > torch.finfo(torch.float64).tiny,
                numerator / denominator,
                torch.nan,
            ).nanmean(dim=1)
            output[start:stop] = scores.detach().cpu().numpy()
    except ImportError:
        for number, permutation in enumerate(index):
            output[number] = mean_spatial_squared_cosine(
                lead[:, permutation] @ state, observed
            )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("A geometry-permutation score is non-finite")
    return output


def _subject_baseline_floor(
    config: dict[str, Any], subject: str, runs: Sequence[str], source_names: set[str]
) -> float:
    root = Path(config["dataset"]["root"])
    sfreq = float(config["dataset"]["sampling_frequency_hz"])
    ieeg_dir = root / "derivatives" / "epochs" / subject / "ieeg"
    scales: list[np.ndarray] = []
    for run in runs:
        header = _read_run_header(ieeg_dir, subject, run)
        description = header["metadata"]["Description"]
        source_label = parse_stimulation_description(description)["source_label"]
        retained = _retained_ieeg_names(header, source_label, source_names)
        lookup = {name: index for index, name in enumerate(header["channel_names"])}
        data = np.asarray(np.load(header["array_path"], mmap_mode="r"), dtype=float)
        selected = data[:, [lookup[name] for name in retained], :]
        _, channel_scales = baseline_channel_statistics(
            selected, _times(header, sfreq), config["windows_seconds"]["baseline"]
        )
        scales.append(channel_scales)
    median = float(np.median(np.concatenate(scales)))
    relative = float(config["dynamic_proxy"]["baseline_scale_floor_relative_to_subject_median"])
    floor = relative * median
    if not np.isfinite(floor) or floor <= 0:
        raise RuntimeError(f"{subject}: invalid subject baseline scale floor")
    return floor


def _seed_for_run(seed: int, subject: str, run: str) -> int:
    digest = hashlib.sha256(f"{seed}:{subject}:{run}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def compute_subject_run_metrics(
    config: dict[str, Any], subject: str, runs: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Compute the run-level endpoints for one subject."""
    root = Path(config["dataset"]["root"])
    sfreq = float(config["dataset"]["sampling_frequency_hz"])
    lead_data = _load_leadfield(config, subject)
    lead = np.asarray(lead_data["oriented_lead_field"], dtype=float)
    eeg_names = [str(value) for value in lead_data["eeg_channel_names"]]
    source_names = [str(value) for value in lead_data["source_labels"]]
    source_lookup = {name: index for index, name in enumerate(source_names)}
    brain_distance_eligible = _brain_distance_eligible_sources(config, subject)
    available_sources = set(source_names) & brain_distance_eligible
    depth = np.asarray(lead_data["source_depth_mm"], dtype=float)
    if depth.shape != (len(source_names),):
        raise ValueError(f"{subject}: missing or invalid MRI-derived source depth")
    floor = _subject_baseline_floor(config, subject, runs, available_sources)
    eeg_dir = root / "derivatives" / "epochs" / subject / "eeg"
    ieeg_dir = root / "derivatives" / "epochs" / subject / "ieeg"

    headers = []
    stimulation_sites = []
    for run in runs:
        eeg_header = _read_run_header(eeg_dir, subject, run)
        ieeg_header = _read_run_header(ieeg_dir, subject, run)
        stimulus = parse_stimulation_description(eeg_header["metadata"]["Description"])
        if stimulus["source_label"] not in source_lookup:
            raise ValueError(f"{subject} run {run}: stimulation source absent from lead field")
        if stimulus["source_label"] not in available_sources:
            raise ValueError(
                f"{subject} run {run}: stimulation source fails MRI brain-distance gate"
            )
        headers.append((run, eeg_header, ieeg_header, stimulus))
        stimulation_sites.append(stimulus["source_label"])
    candidate_sites = sorted(set(stimulation_sites))
    if len(candidate_sites) < 2:
        raise ValueError(f"{subject}: fewer than two stimulation sites")
    candidate_lookup = {name: index for index, name in enumerate(candidate_sites)}
    candidate_matrix_rows: list[np.ndarray] = []
    source_indices: list[int] = []
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for run, eeg_header, ieeg_header, stimulus in headers:
        source_label = stimulus["source_label"]
        eeg_lookup = {name: index for index, name in enumerate(eeg_header["channel_names"])}
        good_eeg = [
            name
            for name in eeg_header["good_channels"]
            if name in eeg_lookup and name in eeg_names
        ]
        lead_eeg_lookup = {name: index for index, name in enumerate(eeg_names)}
        eeg_indices = [eeg_lookup[name] for name in good_eeg]
        lead_eeg_indices = [lead_eeg_lookup[name] for name in good_eeg]
        eeg_trials = np.asarray(
            np.load(eeg_header["array_path"], mmap_mode="r"), dtype=float
        )[:, eeg_indices, :]
        if config["eligibility"]["require_all_finite_signal_samples"] and not np.all(
            np.isfinite(eeg_trials)
        ):
            raise ValueError(f"{subject} run {run}: non-finite EEG samples")
        eeg_times = _times(eeg_header, sfreq)
        evoked = eeg_trials.mean(axis=0)
        early_mask = half_open_time_mask(
            eeg_times, config["windows_seconds"]["static_early"]
        )
        dynamic_mask = half_open_time_mask(
            eeg_times, config["windows_seconds"]["dynamic_propagated"]
        )

        candidate_scores = []
        for candidate in candidate_sites:
            topography = lead[lead_eeg_indices, source_lookup[candidate]][:, None]
            prediction = np.repeat(topography, int(np.sum(early_mask)), axis=1)
            candidate_scores.append(
                mean_spatial_squared_cosine(prediction, evoked[:, early_mask])
            )
        candidate_matrix_rows.append(np.asarray(candidate_scores, dtype=float))
        source_indices.append(candidate_lookup[source_label])
        for candidate, score in zip(candidate_sites, candidate_scores):
            candidate_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "response_source_label": source_label,
                    "candidate_source_label": candidate,
                    "candidate_is_matched": candidate == source_label,
                    "known_site_early_spatial_squared_cosine": float(score),
                }
            )

        retained = _retained_ieeg_names(ieeg_header, source_label, available_sources)
        ieeg_lookup = {name: index for index, name in enumerate(ieeg_header["channel_names"])}
        ieeg_trials = np.asarray(
            np.load(ieeg_header["array_path"], mmap_mode="r"), dtype=float
        )[:, [ieeg_lookup[name] for name in retained], :]
        if config["eligibility"]["require_all_finite_signal_samples"] and not np.all(
            np.isfinite(ieeg_trials)
        ):
            raise ValueError(f"{subject} run {run}: non-finite iEEG samples")
        ieeg_times = _times(ieeg_header, sfreq)
        trajectory, _, _ = baseline_standardized_trial_mean(
            ieeg_trials,
            ieeg_times,
            config["windows_seconds"]["baseline"],
            floor,
        )
        ieeg_dynamic_mask = half_open_time_mask(
            ieeg_times, config["windows_seconds"]["dynamic_propagated"]
        )
        if int(np.sum(ieeg_dynamic_mask)) != int(np.sum(dynamic_mask)):
            raise ValueError(f"{subject} run {run}: modality time grids differ")
        retained_lead = lead[
            np.ix_(lead_eeg_indices, [source_lookup[name] for name in retained])
        ]
        dynamic_prediction = retained_lead @ trajectory[:, ieeg_dynamic_mask]
        observation = evoked[:, dynamic_mask]
        static_topography = lead[
            lead_eeg_indices, source_lookup[source_label]
        ][:, None]
        static_prediction = np.repeat(
            static_topography, observation.shape[1], axis=1
        )
        dynamic_score = mean_spatial_squared_cosine(dynamic_prediction, observation)
        static_score = mean_spatial_squared_cosine(static_prediction, observation)

        covariance, basis = loaded_reference_covariance(
            eeg_trials,
            eeg_times,
            config["windows_seconds"]["baseline"],
            float(config["detectability"]["covariance_diagonal_loading"]),
        )
        target, target_detail = whitened_scalp_detectability_target(
            eeg_trials,
            eeg_times,
            config["windows_seconds"]["baseline"],
            config["windows_seconds"]["dynamic_propagated"],
            float(config["detectability"]["covariance_diagonal_loading"]),
        )
        static_sensitivity = mahalanobis_time_energy(
            static_topography, covariance, basis
        )
        dynamic_information_energy = mahalanobis_time_energy(
            dynamic_prediction, covariance, basis
        )
        null_scores = geometry_permutation_scores(
            retained_lead,
            trajectory[:, ieeg_dynamic_mask],
            observation,
            int(config["dynamic_proxy"]["source_geometry_permutations"]),
            _seed_for_run(int(config["seed"]), subject, run),
        )
        null_p = float((np.sum(null_scores >= dynamic_score) + 1) / (len(null_scores) + 1))
        source_index = source_lookup[source_label]
        rows.append(
            {
                "subject": subject,
                "run": run,
                "source_label": source_label,
                "eeg_trial_count": eeg_header["trial_count"],
                "ieeg_trial_count": ieeg_header["trial_count"],
                "good_eeg_channel_count": len(good_eeg),
                "retained_ieeg_source_count": len(retained),
                "intensity_ma": stimulus["intensity_ma"],
                "pulse_width_ms": stimulus["pulse_width_ms"],
                "source_depth_mm": float(depth[source_index]),
                "known_site_early_score": float(candidate_scores[candidate_lookup[source_label]]),
                "mean_wrong_site_early_score": float(
                    np.mean(np.delete(candidate_scores, candidate_lookup[source_label]))
                ),
                "static_propagated_score": static_score,
                "dynamic_propagated_score": dynamic_score,
                "dynamic_minus_static_score": dynamic_score - static_score,
                "geometry_permutation_null_mean": float(np.mean(null_scores)),
                "geometry_permutation_null_sd": float(np.std(null_scores, ddof=1)),
                "geometry_permutation_one_sided_p": null_p,
                "detectability_target": target,
                "raw_whitened_scalp_energy": target_detail["raw_whitened_energy"],
                "static_fem_sensitivity": static_sensitivity,
                "dynamic_fem_information_energy": dynamic_information_energy,
                "subject_baseline_scale_floor": floor,
            }
        )
    return (
        rows,
        candidate_rows,
        np.vstack(candidate_matrix_rows),
        np.asarray(source_indices, dtype=int),
    )


def _intercept_only_loso(outcomes: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    predictions = np.empty_like(outcomes)
    for subject in np.unique(subjects):
        test = subjects == subject
        predictions[test] = float(np.mean(outcomes[~test]))
    return predictions


def _prediction_summary(
    outcomes: np.ndarray,
    predictions: np.ndarray,
    subjects: np.ndarray,
    intercept_predictions: np.ndarray,
) -> dict[str, float]:
    losses = subject_mean_losses(outcomes, predictions, subjects)
    null_losses = subject_mean_losses(outcomes, intercept_predictions, subjects)
    mse = float(np.mean(list(losses.values())))
    null_mse = float(np.mean(list(null_losses.values())))
    pooled_sse = float(np.sum((outcomes - predictions) ** 2))
    pooled_sst = float(np.sum((outcomes - outcomes.mean()) ** 2))
    design = np.column_stack([np.ones(len(predictions)), predictions])
    calibration = np.linalg.lstsq(design, outcomes, rcond=None)[0]
    return {
        "subject_balanced_mse": mse,
        "subject_balanced_mae": float(
            np.mean(
                [
                    np.mean(np.abs(outcomes[subjects == subject] - predictions[subjects == subject]))
                    for subject in sorted(set(subjects))
                ]
            )
        ),
        "subject_balanced_r2_vs_loso_intercept": float(1.0 - mse / null_mse),
        "pooled_r2": float(1.0 - pooled_sse / pooled_sst),
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
    }


def run_fem_analysis(
    config: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], Path]:
    """Run the three tests of the finite-element analysis."""
    output = _resolve(config["output_dir"], project_root)
    ledger = output / "post_charm_eligibility.csv"
    if not ledger.is_file():
        raise RuntimeError("The post-CHARM eligibility table is missing")
    eligible = load_eligible_runs(ledger, "post_charm_eligible")
    if not eligible:
        raise RuntimeError("The post-CHARM eligibility table contains no eligible runs")
    verify_headmodel_qc(config, project_root, list(eligible))
    result_path = output / "results.json"

    all_rows: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []
    score_matrices: list[np.ndarray] = []
    label_vectors: list[np.ndarray] = []
    for subject, runs in eligible.items():
        rows, candidate_rows, scores, labels = compute_subject_run_metrics(
            config, subject, runs
        )
        all_rows.extend(rows)
        all_candidate_rows.extend(candidate_rows)
        score_matrices.append(scores)
        label_vectors.append(labels)

    permutations = int(config["inference"]["permutations"])
    seed = int(config["seed"])
    test1 = hierarchical_label_permutation_test(
        score_matrices, label_vectors, permutations, seed
    )
    test1_subject_effects = np.asarray(
        [
            subject_source_effect(scores, labels)[0]
            for scores, labels in zip(score_matrices, label_vectors)
        ]
    )
    test1_effect_by_subject = {
        subject: float(effect)
        for subject, effect in zip(eligible, test1_subject_effects)
    }
    dynamic_by_subject: dict[str, list[float]] = defaultdict(list)
    geometry_control_by_subject: dict[str, list[float]] = defaultdict(list)
    for row in all_rows:
        dynamic_by_subject[row["subject"]].append(row["dynamic_minus_static_score"])
        geometry_control_by_subject[row["subject"]].append(
            row["dynamic_propagated_score"]
            - row["geometry_permutation_null_mean"]
        )
    dynamic_subjects = sorted(dynamic_by_subject)
    dynamic_differences = np.array(
        [np.mean(dynamic_by_subject[subject]) for subject in dynamic_subjects]
    )
    test2 = one_sided_sign_permutation_test(
        dynamic_differences, permutations, seed + 1
    )

    subjects = np.asarray([row["subject"] for row in all_rows], dtype=str)
    outcomes = np.asarray([row["detectability_target"] for row in all_rows])
    base = np.asarray(
        [
            [
                np.log(row["intensity_ma"]),
                np.log(row["pulse_width_ms"]),
                row["source_depth_mm"],
            ]
            for row in all_rows
        ]
    )
    static = np.column_stack(
        [base, np.log(np.maximum([row["static_fem_sensitivity"] for row in all_rows], np.finfo(float).tiny))]
    )
    dynamic = np.column_stack(
        [
            static,
            np.log(
                np.maximum(
                    [row["dynamic_fem_information_energy"] for row in all_rows],
                    np.finfo(float).tiny,
                )
            ),
        ]
    )
    alpha = float(config["detectability"]["ridge_alpha"])
    predictions = {
        "amplitude_depth": leave_one_subject_out_ridge(base, outcomes, subjects, alpha),
        "static": leave_one_subject_out_ridge(static, outcomes, subjects, alpha),
        "dynamic": leave_one_subject_out_ridge(dynamic, outcomes, subjects, alpha),
    }
    intercept_predictions = _intercept_only_loso(outcomes, subjects)
    prediction_summaries = {
        name: _prediction_summary(
            outcomes, values, subjects, intercept_predictions
        )
        for name, values in predictions.items()
    }
    static_losses = subject_mean_losses(outcomes, predictions["static"], subjects)
    dynamic_losses = subject_mean_losses(outcomes, predictions["dynamic"], subjects)
    heldout_subjects = sorted(static_losses)
    heldout_differences = np.array(
        [static_losses[subject] - dynamic_losses[subject] for subject in heldout_subjects]
    )
    test3 = one_sided_sign_permutation_test(
        heldout_differences, permutations, seed + 2
    )
    bootstrap_resamples = int(config["inference"]["bootstrap_resamples"])
    test1["subject_bootstrap_mean_95_ci"] = subject_bootstrap_mean_interval(
        test1_subject_effects, bootstrap_resamples, seed + 10
    )
    test2["subject_bootstrap_mean_95_ci"] = subject_bootstrap_mean_interval(
        dynamic_differences, bootstrap_resamples, seed + 11
    )
    test3["subject_bootstrap_mean_95_ci"] = subject_bootstrap_mean_interval(
        heldout_differences, bootstrap_resamples, seed + 12
    )
    tests = [test1, test2, test3]
    raw_p = [test["one_sided_p_value"] for test in tests]
    adjusted = holm_adjust(raw_p)
    test_names = config["inference"]["tests"]
    test_results = {
        name: {**test, "holm_adjusted_p_value": float(adjusted[index])}
        for index, (name, test) in enumerate(zip(test_names, tests))
    }

    for index, row in enumerate(all_rows):
        row["loso_prediction_amplitude_depth"] = float(predictions["amplitude_depth"][index])
        row["loso_prediction_static"] = float(predictions["static"][index])
        row["loso_prediction_dynamic"] = float(predictions["dynamic"][index])
    _write_csv(output / "run_metrics.csv", all_rows)
    _write_csv(output / "candidate_scores.csv", all_candidate_rows)
    subject_rows = []
    for index, subject in enumerate(dynamic_subjects):
        subject_rows.append(
            {
                "subject": subject,
                "run_count": len(dynamic_by_subject[subject]),
                "mean_known_minus_wrong_early_score": test1_effect_by_subject[subject],
                "mean_dynamic_minus_static_score": float(dynamic_differences[index]),
                "mean_dynamic_minus_geometry_null_score": float(
                    np.mean(geometry_control_by_subject[subject])
                ),
                "static_loso_mse": static_losses[subject],
                "dynamic_loso_mse": dynamic_losses[subject],
                "static_minus_dynamic_loso_mse": static_losses[subject]
                - dynamic_losses[subject],
            }
        )
    _write_csv(output / "subject_metrics.csv", subject_rows)
    results = {
        "subject_count": len(eligible),
        "run_count": len(all_rows),
        "tests": test_results,
        "prediction_summaries": prediction_summaries,
        "dynamic_vs_static_prediction": {
            "mean_subject_static_minus_dynamic_mse": float(
                np.mean(heldout_differences)
            ),
            "adverse_subject_fraction": float(np.mean(heldout_differences < 0)),
        },
        "secondary_geometry_negative_control": {
            "mean_subject_dynamic_minus_permuted_geometry_score": float(
                np.mean(
                    [
                        np.mean(geometry_control_by_subject[subject])
                        for subject in dynamic_subjects
                    ]
                )
            ),
            "positive_subject_fraction": float(
                np.mean(
                    [
                        np.mean(geometry_control_by_subject[subject]) > 0
                        for subject in dynamic_subjects
                    ]
                )
            ),
        },
    }
    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results, result_path
