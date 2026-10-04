"""Generative state-space mapping from intracranial to scalp recordings during working memory.

For every session, latent current dipoles at the retained bipolar intracranial
sites are fitted on the encoding period and observed intracranially through a
homogeneous-medium dipole operator and at the scalp through the template-head
lead field. The model is scored on the maintenance period by held-out Gaussian
log-likelihood, and compared with controls that permute the source geometry,
shift the predictions in time, permute trial identity or replace the
intracranial operator by the identity.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neural_observability.head_models.referencing import average_reference_basis
from neural_observability.statistics import (
    exact_subject_sign_flip_test,
    subject_bootstrap_mean_interval,
)
from neural_observability.stimulation.cohort import _write_csv
from neural_observability.working_memory.data import (
    _extract_edf_trials,
    _paired_sessions,
    _read_json,
    _read_tsv,
    _sidecar,
    _subject_seed,
    _validate_paired_events,
    load_config as load_data_config,
    output_dir as data_output_dir,
    prepare_edf_header_repairs,
    verify_downloaded_dataset,
)
from neural_observability.working_memory.generative_model import (
    ScalpObservationFit,
    analytic_bipolar_operator,
    causal_filter_resample_trials,
    fit_latent_dipole_model,
    fit_scalp_observation_model,
    held_out_log_likelihood_gain,
    interval_mask,
    predict_scalp_trials,
    refit_gains_for_lead_field,
)
from neural_observability.working_memory.sources import (
    adjacent_bipolar_sources,
    bipolarize_trials,
)

CONTROL_SEED_LABEL = "phase_c7_controls"


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and check the generative-model configuration."""
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"seed", "data_config", "preprocessing", "candidate", "controls", "output_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    return config, config_path.parent.parent


def _subject_means(
    values: Sequence[float], subjects: Sequence[str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, subject in zip(values, subjects):
        grouped[str(subject)].append(float(value))
    return {subject: float(np.mean(grouped[subject])) for subject in sorted(grouped)}


def _summarize_effect(
    session_rows: list[dict[str, Any]], field: str, *, seed: int
) -> tuple[dict[str, Any], dict[str, float]]:
    by_subject = _subject_means(
        [float(row[field]) for row in session_rows],
        [str(row["subject"]) for row in session_rows],
    )
    ordered = np.asarray([by_subject[key] for key in sorted(by_subject)])
    sign_flip = exact_subject_sign_flip_test(ordered)
    interval = subject_bootstrap_mean_interval(ordered, 50000, seed)
    return (
        {
            **sign_flip,
            "median_subject_effect": float(np.median(ordered)),
            "positive_subject_fraction": float(np.mean(ordered > 0)),
            "subject_bootstrap_mean_95_ci_descriptive": list(interval),
        },
        by_subject,
    )


def _score_arrays(
    scalp_trials: np.ndarray,
    predicted_trials: np.ndarray,
    times: np.ndarray,
    scoring_interval: Sequence[float],
    fit: ScalpObservationFit,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    mask = interval_mask(times, scoring_interval, closed_right=True)
    observed = scalp_trials - fit.scalp_mean[None, :, None]
    observed = observed[:, :, mask]
    predicted = predicted_trials[:, :, mask]
    observed_matrix = observed.transpose(1, 0, 2).reshape(observed.shape[1], -1)
    predicted_matrix = predicted.transpose(1, 0, 2).reshape(predicted.shape[1], -1)
    precision_observed = np.linalg.solve(fit.nuisance_covariance, observed_matrix)
    precision_prediction = np.linalg.solve(
        fit.nuisance_covariance, predicted_matrix
    )
    cross = float(np.mean(np.sum(predicted_matrix * precision_observed, axis=0)))
    prediction_quadratic = float(
        np.mean(np.sum(predicted_matrix * precision_prediction, axis=0))
    )
    score = 0.5 * (2.0 * cross - prediction_quadratic) / observed.shape[1]
    return score, observed, precision_observed, prediction_quadratic


def _permuted_prediction_score(
    prediction: np.ndarray,
    precision_observed: np.ndarray,
    prediction_quadratic: float,
    n_scalp_dimensions: int,
) -> float:
    matrix = prediction.transpose(1, 0, 2).reshape(n_scalp_dimensions, -1)
    cross = float(np.mean(np.sum(matrix * precision_observed, axis=0)))
    return 0.5 * (2.0 * cross - prediction_quadratic) / n_scalp_dimensions


def _geometry_score(
    lead_field: np.ndarray,
    gains: np.ndarray,
    source_covariance: np.ndarray,
    source_precision_scalp_cross: np.ndarray,
    nuisance_covariance: np.ndarray,
) -> float:
    mean_operator = lead_field * gains[None, :]
    cross = float(np.sum(mean_operator.T * source_precision_scalp_cross))
    precision_operator = np.linalg.solve(nuisance_covariance, mean_operator)
    quadratic_kernel = mean_operator.T @ precision_operator
    quadratic = float(np.sum(quadratic_kernel * source_covariance))
    return 0.5 * (2.0 * cross - quadratic) / lead_field.shape[0]



def load_session_inputs(data_config: dict[str, Any], data_root: Path) -> tuple[dict[str, Path], dict[str, Any], set[str]]:
    """Verify the dataset, repair the two EDF headers and return the eligible subjects."""
    verify_downloaded_dataset(data_config)
    repaired_signals, repairs = prepare_edf_header_repairs(data_config)
    eligibility = _read_json(data_root / "eligibility.json")
    eligible = {row["subject"] for row in eligibility["subject_metadata"] if row["eligible"]}
    return repaired_signals, repairs, eligible


def run_generative_analysis(
    config: dict[str, Any], project_root: Path, max_subjects: int | None = None
) -> tuple[dict[str, Any], Path]:
    """Fit and score the generative model in every session, with its controls."""
    data_config, data_project_root = load_data_config(project_root / config["data_config"])
    data_root = data_output_dir(data_config, data_project_root)
    dataset_root = Path(data_config["dataset"]["root"])
    sessions = _paired_sessions(dataset_root)
    all_subjects = sorted({item[0] for item in sessions})
    if max_subjects is not None:
        selected = set(all_subjects[: int(max_subjects)])
        sessions = [item for item in sessions if item[0] in selected]

    output = project_root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    repaired_signals, repairs, eligible = load_session_inputs(data_config, data_root)
    (output / "edf_header_repairs.json").write_text(json.dumps(repairs, indent=2, sort_keys=True) + "\n")
    forward_root = data_root / "template_forward"
    sessions = [item for item in sessions if item[0] in eligible]

    preprocessing = config["preprocessing"]
    candidate = config["candidate"]
    controls = config["controls"]
    target_frequency = float(preprocessing["target_sampling_frequency_hz"])
    encoding_interval = preprocessing["encoding_reference_window_seconds"]
    scoring_interval = preprocessing["maintenance_response_window_seconds"]
    session_rows: list[dict[str, Any]] = []

    for number, (subject, session, eeg_dir, ieeg_dir) in enumerate(sessions, start=1):
        print(f"Session {number}/{len(sessions)}: {subject} {session}", flush=True)
        forward_data = np.load(
            forward_root / f"{subject}_template_forward.npz", allow_pickle=False
        )
        lead = np.asarray(forward_data["lead_field"], dtype=float)
        eeg_names = [row["name"] for row in _read_tsv(_sidecar(eeg_dir, "channels.tsv"))]
        ieeg_names = [row["name"] for row in _read_tsv(_sidecar(ieeg_dir, "channels.tsv"))]
        if eeg_names != [str(value) for value in forward_data["eeg_names"]]:
            raise RuntimeError("EEG channel order differs from the template forward model")
        electrode_rows = _read_tsv(_sidecar(ieeg_dir, "electrodes.tsv"))
        source_lookup = {
            source.name: source
            for source in adjacent_bipolar_sources(ieeg_names, electrode_rows)
        }
        retained_names = [str(value) for value in forward_data["source_names"]]
        if any(name not in source_lookup for name in retained_names):
            raise RuntimeError("A retained bipolar source is absent")
        retained_sources = [source_lookup[name] for name in retained_names]
        coordinates = {
            row["name"]: np.asarray(
                [float(row[axis]) for axis in ("x", "y", "z")], dtype=float
            )
            for row in electrode_rows
            if row["name"] in {source.lower_name for source in retained_sources}
            | {source.upper_name for source in retained_sources}
        }
        raw_intracranial, intracranial_operator = analytic_bipolar_operator(
            retained_sources,
            coordinates,
            softening_mm=float(candidate["intracranial_operator_softening_mm"]),
        )

        eeg_events = _read_tsv(_sidecar(eeg_dir, "events.tsv"))
        ieeg_events = _read_tsv(_sidecar(ieeg_dir, "events.tsv"))
        keep = _validate_paired_events(eeg_events, ieeg_events)
        eeg_frequency = float(_read_json(_sidecar(eeg_dir, "eeg.json"))["SamplingFrequency"])
        ieeg_frequency = float(_read_json(_sidecar(ieeg_dir, "ieeg.json"))["SamplingFrequency"])
        eeg_signal = _sidecar(eeg_dir, "eeg.edf")
        ieeg_signal = _sidecar(ieeg_dir, "ieeg.edf")
        eeg_trials = _extract_edf_trials(
            repaired_signals.get(str(eeg_signal.relative_to(dataset_root)), eeg_signal),
            eeg_names,
            eeg_events,
            keep,
            eeg_frequency,
        )
        contact_trials = _extract_edf_trials(
            repaired_signals.get(str(ieeg_signal.relative_to(dataset_root)), ieeg_signal),
            ieeg_names,
            ieeg_events,
            keep,
            ieeg_frequency,
        )
        bipolar_trials = bipolarize_trials(contact_trials, retained_sources)
        del contact_trials
        eeg_trials = causal_filter_resample_trials(
            eeg_trials, eeg_frequency, target_frequency, preprocessing["bandpass_hz"]
        )
        bipolar_trials = causal_filter_resample_trials(
            bipolar_trials,
            ieeg_frequency,
            target_frequency,
            preprocessing["bandpass_hz"],
        )
        if eeg_trials.shape[2] != bipolar_trials.shape[2]:
            raise RuntimeError("Scalp and intracranial trial lengths differ")
        times = np.arange(eeg_trials.shape[2], dtype=float) / target_frequency
        basis = average_reference_basis(len(eeg_names))
        scalp_trials = np.einsum("dc,tcs->tds", basis.T, eeg_trials, optimize=True)
        lead_basis = basis.T @ lead

        latent_arguments = {
            "inverse_ridge": float(candidate["latent_inverse_ridge"]),
            "transition_ridge": float(candidate["transition_ridge"]),
            "maximum_transition_radius": float(
                candidate["maximum_discrete_spectral_radius"]
            ),
            "process_loading": float(
                candidate["process_covariance_diagonal_loading"]
            ),
            "observation_loading": float(
                candidate["intracranial_noise_covariance_diagonal_loading"]
            ),
        }
        latent_fit, source_states = fit_latent_dipole_model(
            bipolar_trials,
            times,
            encoding_interval,
            intracranial_operator,
            **latent_arguments,
        )
        scalp_fit = fit_scalp_observation_model(
            source_states,
            scalp_trials,
            times,
            encoding_interval,
            lead_basis,
            gain_ridge=float(candidate["source_to_scalp_diagonal_gain_ridge"]),
            nuisance_loading=float(
                candidate["unobserved_source_and_sensor_covariance_diagonal_loading"]
            ),
        )
        prediction = predict_scalp_trials(source_states, lead_basis, scalp_fit)
        actual_score, observed_response, precision_observed, prediction_quadratic = _score_arrays(
            scalp_trials, prediction, times, scoring_interval, scalp_fit
        )
        direct_score = held_out_log_likelihood_gain(
            scalp_trials, prediction, times, scoring_interval, scalp_fit
        )
        score_implementation_error = abs(actual_score - direct_score)
        if score_implementation_error > 1e-12:
            raise RuntimeError("Optimized and direct likelihood scores disagree")

        response_mask = interval_mask(times, scoring_interval, closed_right=True)
        centered_sources = source_states - scalp_fit.source_mean[None, :, None]
        q = centered_sources[:, :, response_mask].transpose(1, 0, 2).reshape(
            len(retained_sources), -1
        )
        source_covariance = q @ q.T / q.shape[1]
        source_precision_scalp_cross = q @ precision_observed.T / q.shape[1]

        rng = np.random.default_rng(
            _subject_seed(config["seed"], subject, session, CONTROL_SEED_LABEL)
        )
        geometry_scores = []
        for _ in range(int(controls["geometry_column_permutations_per_session"])):
            permutation = rng.permutation(len(retained_sources))
            while np.array_equal(permutation, np.arange(len(retained_sources))):
                permutation = rng.permutation(len(retained_sources))
            control_lead = lead_basis[:, permutation]
            control_gains = refit_gains_for_lead_field(
                scalp_fit,
                control_lead,
                gain_ridge=float(candidate["source_to_scalp_diagonal_gain_ridge"]),
            )
            geometry_scores.append(
                _geometry_score(
                    control_lead,
                    control_gains,
                    source_covariance,
                    source_precision_scalp_cross,
                    scalp_fit.nuisance_covariance,
                )
            )
        geometry_scores = np.asarray(geometry_scores)

        prediction_response = prediction[:, :, response_mask]
        response_samples = prediction_response.shape[2]
        minimum_shift = int(
            round(float(controls["minimum_circular_time_shift_seconds"]) * target_frequency)
        )
        if 2 * minimum_shift >= response_samples:
            raise RuntimeError("The fixed minimum circular shift leaves no valid offsets")
        time_scores = []
        for _ in range(int(controls["circular_time_permutations_per_session"])):
            offset = int(rng.integers(minimum_shift, response_samples - minimum_shift + 1))
            shifted = np.roll(prediction_response, offset, axis=2)
            time_scores.append(
                _permuted_prediction_score(
                    shifted,
                    precision_observed,
                    prediction_quadratic,
                    observed_response.shape[1],
                )
            )
        time_scores = np.asarray(time_scores)
        fixed_offset = int(
            round(float(controls["fixed_circular_time_shift_seconds"]) * target_frequency)
        )
        fixed_shift_score = _permuted_prediction_score(
            np.roll(prediction_response, fixed_offset, axis=2),
            precision_observed,
            prediction_quadratic,
            observed_response.shape[1],
        )

        trial_scores = []
        trial_indices = np.arange(len(keep))
        for _ in range(int(controls["trial_identity_permutations_per_session"])):
            permutation = rng.permutation(len(keep))
            while np.array_equal(permutation, trial_indices):
                permutation = rng.permutation(len(keep))
            trial_scores.append(
                _permuted_prediction_score(
                    prediction_response[permutation],
                    precision_observed,
                    prediction_quadratic,
                    observed_response.shape[1],
                )
            )
        trial_scores = np.asarray(trial_scores)

        _, identity_states = fit_latent_dipole_model(
            bipolar_trials,
            times,
            encoding_interval,
            np.eye(len(retained_sources)),
            **latent_arguments,
        )
        identity_scalp_fit = fit_scalp_observation_model(
            identity_states,
            scalp_trials,
            times,
            encoding_interval,
            lead_basis,
            gain_ridge=float(candidate["source_to_scalp_diagonal_gain_ridge"]),
            nuisance_loading=float(
                candidate["unobserved_source_and_sensor_covariance_diagonal_loading"]
            ),
        )
        identity_prediction = predict_scalp_trials(
            identity_states, lead_basis, identity_scalp_fit
        )
        identity_score = held_out_log_likelihood_gain(
            scalp_trials,
            identity_prediction,
            times,
            scoring_interval,
            scalp_fit,
        )
        diagonal_fit = replace(
            scalp_fit,
            nuisance_covariance=np.diag(np.diag(scalp_fit.nuisance_covariance)),
        )
        diagonal_nuisance_score = held_out_log_likelihood_gain(
            scalp_trials, prediction, times, scoring_interval, diagonal_fit
        )

        off_diagonal = raw_intracranial - np.diag(np.diag(raw_intracranial))
        raw_energy = float(np.sum(raw_intracranial**2))
        session_rows.append(
            {
                "subject": subject,
                "session": session,
                "clean_trial_count": len(keep),
                "scalp_channel_count": len(eeg_names),
                "source_count": len(retained_sources),
                "actual_held_out_log_likelihood_gain": actual_score,
                "geometry_null_mean": float(np.mean(geometry_scores)),
                "geometry_null_p95": float(np.quantile(geometry_scores, 0.95)),
                "actual_minus_geometry_null_mean": actual_score
                - float(np.mean(geometry_scores)),
                "time_null_mean": float(np.mean(time_scores)),
                "time_null_p95": float(np.quantile(time_scores, 0.95)),
                "actual_minus_time_null_mean": actual_score - float(np.mean(time_scores)),
                "fixed_one_second_shift_score": fixed_shift_score,
                "actual_minus_fixed_one_second_shift": actual_score - fixed_shift_score,
                "trial_identity_null_mean": float(np.mean(trial_scores)),
                "trial_identity_null_p95": float(np.quantile(trial_scores, 0.95)),
                "actual_minus_trial_identity_null_mean": actual_score
                - float(np.mean(trial_scores)),
                "identity_intracranial_operator_score": identity_score,
                "actual_minus_identity_intracranial_operator": actual_score
                - identity_score,
                "diagonal_nuisance_covariance_score": diagonal_nuisance_score,
                "full_minus_diagonal_nuisance_score": actual_score
                - diagonal_nuisance_score,
                "encoding_scalp_variance_explained": scalp_fit.encoding_variance_explained,
                "nuisance_covariance_effective_rank": scalp_fit.nuisance_effective_rank,
                "latent_transition_spectral_radius": latent_fit.transition_spectral_radius,
                "kalman_iterations": latent_fit.kalman_iterations,
                "intracranial_operator_condition_number": latent_fit.operator_condition_number,
                "intracranial_operator_offdiagonal_energy_fraction": float(
                    np.sum(off_diagonal**2) / raw_energy
                ),
                "score_implementation_absolute_error": score_implementation_error,
            }
        )
        del eeg_trials, bipolar_trials, scalp_trials, source_states, identity_states

    summary_fields = [
        ("predictive_value", "actual_held_out_log_likelihood_gain"),
        ("geometry", "actual_minus_geometry_null_mean"),
        ("time", "actual_minus_time_null_mean"),
        ("trial_identity", "actual_minus_trial_identity_null_mean"),
        ("intracranial_operator", "actual_minus_identity_intracranial_operator"),
        ("fixed_one_second", "actual_minus_fixed_one_second_shift"),
    ]
    summaries: dict[str, Any] = {}
    subject_effects: dict[str, dict[str, Any]] = {}
    for index, (label, field) in enumerate(summary_fields):
        summaries[label], effects = _summarize_effect(
            session_rows, field, seed=int(config["seed"]) + index
        )
        for subject, effect in effects.items():
            subject_effects.setdefault(subject, {"subject": subject})[
                f"{label}_effect"
            ] = effect

    result = {
        "subject_count": len({row["subject"] for row in session_rows}),
        "session_count": len(session_rows),
        "clean_trial_count": int(sum(row["clean_trial_count"] for row in session_rows)),
        "control_summaries": summaries,
    }
    _write_csv(output / "session_metrics.csv", session_rows)
    _write_csv(
        output / "subject_control_effects.csv",
        [subject_effects[key] for key in sorted(subject_effects)],
    )
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result, result_path
