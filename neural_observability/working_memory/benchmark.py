"""Comparison of scalp mappings from intracranial recordings during working memory.

Four models are scored on the maintenance period by the held-out Gaussian
log-likelihood gain of the generative analysis: a nuisance-only model, a static
lead-field mapping of the intracranial signals, a lagged linear ridge mapping
whose lag set and regularisation are selected on the other participants, and
the generative state-space model.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from neural_observability.head_models.referencing import average_reference_basis
from neural_observability.statistics import (
    exact_subject_sign_flip_test,
    subject_bootstrap_mean_interval,
)
from neural_observability.stimulation.cohort import _write_csv
from neural_observability.working_memory.baselines import (
    encoding_standardize_channels,
    lagged_ridge_candidate_scores,
    nested_loso_hyperparameter_selection,
)
from neural_observability.working_memory.data import (
    _extract_edf_trials,
    _read_json,
    _read_tsv,
    _sidecar,
    _validate_paired_events,
    _paired_sessions,
    load_config as load_data_config,
    output_dir as data_output_dir,
)
from neural_observability.working_memory.generative_analysis import (
    load_config as load_generative_config,
    load_session_inputs,
)
from neural_observability.working_memory.generative_model import (
    analytic_bipolar_operator,
    causal_filter_resample_trials,
    fit_latent_dipole_model,
    fit_scalp_observation_model,
    held_out_log_likelihood_gain,
    predict_scalp_trials,
)
from neural_observability.working_memory.sources import (
    adjacent_bipolar_sources,
    bipolarize_trials,
)

MODELS = (
    "nuisance_only",
    "static_lead_field",
    "lagged_linear_ridge",
    "generative_state_space",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and check the mapping-benchmark configuration."""
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"seed", "generative_config", "generative_session_metrics", "models", "inference",
                "consistency_tolerance", "output_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    return config, config_path.parent.parent


def _summarize_values(values: np.ndarray, bootstrap: int, seed: int) -> dict[str, Any]:
    vector = np.asarray(values, dtype=float)
    sign_flip = exact_subject_sign_flip_test(vector)
    return {
        **sign_flip,
        "median_subject_value": float(np.median(vector)),
        "positive_subject_fraction": float(np.mean(vector > 0)),
        "subject_bootstrap_mean_95_ci_descriptive": subject_bootstrap_mean_interval(
            vector, bootstrap, seed
        ),
    }



def run_benchmark(
    config: dict[str, Any],
    project_root: Path,
    *,
    max_subjects: int | None = None,
    bootstrap_resamples: int | None = None,
) -> tuple[dict[str, Any], Path]:
    """Score the four mappings in every session and compare them across participants."""
    generative_config, _ = load_generative_config(project_root / config["generative_config"])
    data_config, data_project_root = load_data_config(project_root / generative_config["data_config"])
    data_root = data_output_dir(data_config, data_project_root)
    dataset_root = Path(data_config["dataset"]["root"])
    sessions = _paired_sessions(dataset_root)
    all_subjects = sorted({item[0] for item in sessions})
    if max_subjects is not None:
        selected = set(all_subjects[: int(max_subjects)])
        sessions = [item for item in sessions if item[0] in selected]
    bootstraps = int(bootstrap_resamples or config["inference"]["subject_bootstrap_resamples"])

    output = project_root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    repaired_signals, _, eligible = load_session_inputs(data_config, data_root)
    sessions = [item for item in sessions if item[0] in eligible]
    generative_reference = {
        (row["subject"], row["session"]): row
        for row in _read_csv(project_root / config["generative_session_metrics"])
    }
    preprocessing = generative_config["preprocessing"]
    candidate = generative_config["candidate"]
    target_frequency = float(preprocessing["target_sampling_frequency_hz"])
    encoding_interval = preprocessing["encoding_reference_window_seconds"]
    scoring_interval = preprocessing["maintenance_response_window_seconds"]
    lag_sets = config["models"]["lagged_linear_ridge"]["lag_tap_sets_samples_at_200_hz"]
    alphas = config["models"]["lagged_linear_ridge"]["ridge_alpha_multipliers"]
    forward_root = data_root / "template_forward"

    session_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    score_tensor = np.empty((len(sessions), len(lag_sets), len(alphas)), dtype=float)
    reproduction_differences: list[float] = []
    for session_index, (subject, session, eeg_dir, ieeg_dir) in enumerate(sessions):
        print(f"Session {session_index + 1}/{len(sessions)}: {subject} {session}", flush=True)
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
        coordinate_names = {
            source.lower_name for source in retained_sources
        } | {source.upper_name for source in retained_sources}
        coordinates = {
            row["name"]: np.asarray(
                [float(row[axis]) for axis in ("x", "y", "z")], dtype=float
            )
            for row in electrode_rows
            if row["name"] in coordinate_names
        }
        _, intracranial_operator = analytic_bipolar_operator(
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
            eeg_trials,
            eeg_frequency,
            target_frequency,
            preprocessing["bandpass_hz"],
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

        latent_fit, source_states = fit_latent_dipole_model(
            bipolar_trials,
            times,
            encoding_interval,
            intracranial_operator,
            inverse_ridge=float(candidate["latent_inverse_ridge"]),
            transition_ridge=float(candidate["transition_ridge"]),
            maximum_transition_radius=float(candidate["maximum_discrete_spectral_radius"]),
            process_loading=float(candidate["process_covariance_diagonal_loading"]),
            observation_loading=float(candidate["intracranial_noise_covariance_diagonal_loading"]),
        )
        generative_fit = fit_scalp_observation_model(
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
        generative_prediction = predict_scalp_trials(source_states, lead_basis, generative_fit)
        generative_score = held_out_log_likelihood_gain(
            scalp_trials, generative_prediction, times, scoring_interval, generative_fit
        )

        reference = generative_reference[(subject, session)]
        if (
            int(reference["clean_trial_count"]) != len(keep)
            or int(reference["scalp_channel_count"]) != len(eeg_names)
            or int(reference["source_count"]) != len(retained_sources)
        ):
            raise RuntimeError(f"{subject} {session}: session dimensions differ from the generative analysis")
        reproduction_difference = abs(
            generative_score - float(reference["actual_held_out_log_likelihood_gain"])
        )
        reproduction_differences.append(reproduction_difference)

        static_sources, _, _ = encoding_standardize_channels(
            bipolar_trials, times, encoding_interval
        )
        static_fit = fit_scalp_observation_model(
            static_sources,
            scalp_trials,
            times,
            encoding_interval,
            lead_basis,
            gain_ridge=float(
                config["models"]["static_lead_field"]["source_gain_ridge"]
            ),
            nuisance_loading=float(
                candidate["unobserved_source_and_sensor_covariance_diagonal_loading"]
            ),
        )
        static_prediction = predict_scalp_trials(
            static_sources, lead_basis, static_fit
        )
        static_score = held_out_log_likelihood_gain(
            scalp_trials,
            static_prediction,
            times,
            scoring_interval,
            generative_fit,
        )

        ridge_candidates = lagged_ridge_candidate_scores(
            bipolar_trials,
            scalp_trials,
            times,
            encoding_interval,
            scoring_interval,
            generative_fit.scalp_mean,
            generative_fit.nuisance_covariance,
            lag_sets,
            alphas,
        )
        if len(ridge_candidates) != len(lag_sets) * len(alphas):
            raise RuntimeError("Ridge candidate grid is incomplete")
        for candidate_index, fitted in enumerate(ridge_candidates):
            lag_index = candidate_index // len(alphas)
            alpha_index = candidate_index % len(alphas)
            if fitted.lag_taps != tuple(lag_sets[lag_index]) or fitted.alpha != float(
                alphas[alpha_index]
            ):
                raise RuntimeError("Ridge candidate ordering changed")
            score_tensor[session_index, lag_index, alpha_index] = fitted.score
            candidate_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "lag_index": lag_index,
                    "lag_taps_samples": ";".join(map(str, fitted.lag_taps)),
                    "maximum_lag_samples": max(fitted.lag_taps),
                    "maximum_lag_milliseconds": 1000.0 * max(fitted.lag_taps) / target_frequency,
                    "alpha_index": alpha_index,
                    "ridge_alpha_multiplier": fitted.alpha,
                    "ridge_alpha_absolute": fitted.alpha_absolute,
                    "held_out_log_likelihood_gain": fitted.score,
                    "coefficient_frobenius_norm": fitted.coefficient_frobenius_norm,
                    "training_observation_count": fitted.training_observation_count,
                    "scoring_observation_count": fitted.scoring_observation_count,
                }
            )
        session_rows.append(
            {
                "subject": subject,
                "session": session,
                "clean_trial_count": len(keep),
                "scalp_channel_count": len(eeg_names),
                "source_count": len(retained_sources),
                "nuisance_only_score": 0.0,
                "static_lead_field_score": static_score,
                "generative_state_space_score": generative_score,
                "difference_from_generative_analysis": reproduction_difference,
                "nuisance_covariance_effective_rank": generative_fit.nuisance_effective_rank,
                "transition_spectral_radius": latent_fit.transition_spectral_radius,
            }
        )
        del (
            eeg_trials,
            bipolar_trials,
            scalp_trials,
            source_states,
            static_sources,
            generative_prediction,
            static_prediction,
        )

    maximum_difference = float(max(reproduction_differences, default=np.inf))
    if maximum_difference > float(config["consistency_tolerance"]):
        raise RuntimeError(
            f"Generative-model scores differ from the generative analysis by {maximum_difference:.3g}"
        )

    session_subjects = [row["subject"] for row in session_rows]
    selections = nested_loso_hyperparameter_selection(
        score_tensor, session_subjects, lag_sets, alphas
    )
    selected_scores = np.full(len(session_rows), np.nan, dtype=float)
    selected_lag_indices = np.full(len(session_rows), -1, dtype=int)
    selected_alpha_indices = np.full(len(session_rows), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    leakage_pass = True
    for selection in selections:
        held_out = selection["held_out_subject"]
        training_subjects = selection["training_subjects"]
        leakage_pass &= held_out not in training_subjects
        indices = np.asarray(selection["test_session_indices"], dtype=int)
        selected_scores[indices] = np.asarray(selection["test_session_scores"], dtype=float)
        selected_lag_indices[indices] = int(selection["selected_lag_index"])
        selected_alpha_indices[indices] = int(selection["selected_alpha_index"])
        fold_rows.append(
            {
                "held_out_subject": held_out,
                "training_subject_count": len(training_subjects),
                "training_subjects": ";".join(training_subjects),
                "test_subject_excluded_from_selection": held_out not in training_subjects,
                "selected_lag_index": int(selection["selected_lag_index"]),
                "selected_lag_taps_samples": ";".join(
                    map(str, selection["selected_lag_taps"])
                ),
                "selected_maximum_lag_samples": max(selection["selected_lag_taps"]),
                "selected_maximum_lag_milliseconds": 1000.0
                * max(selection["selected_lag_taps"])
                / target_frequency,
                "selected_alpha_index": int(selection["selected_alpha_index"]),
                "selected_alpha": float(selection["selected_alpha"]),
                "training_equal_subject_mean_score": float(
                    selection["training_equal_subject_mean_score"]
                ),
                "held_out_session_count": len(indices),
            }
        )
    if not leakage_pass or not np.all(np.isfinite(selected_scores)):
        raise RuntimeError("Hyperparameter selection used the test subject or left sessions unscored")
    for index, row in enumerate(session_rows):
        row["lagged_linear_ridge_score"] = float(selected_scores[index])
        row["selected_lag_index_from_training_subjects"] = int(
            selected_lag_indices[index]
        )
        row["selected_lag_taps_samples_from_training_subjects"] = ";".join(
            map(str, lag_sets[selected_lag_indices[index]])
        )
        row["selected_ridge_alpha_from_training_subjects"] = float(
            alphas[selected_alpha_indices[index]]
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[row["subject"]].append(row)
    subject_rows: list[dict[str, Any]] = []
    for subject in sorted(grouped):
        rows = grouped[subject]
        scores = {
            name: float(np.mean([float(row[f"{name}_score"]) for row in rows]))
            for name in MODELS
        }
        subject_rows.append(
            {
                "subject": subject,
                "session_count": len(rows),
                **{f"{name}_score": value for name, value in scores.items()},
                "generative_minus_ridge": scores["generative_state_space"]
                - scores["lagged_linear_ridge"],
                "ridge_minus_static": scores["lagged_linear_ridge"]
                - scores["static_lead_field"],
                "generative_minus_static": scores["generative_state_space"]
                - scores["static_lead_field"],
            }
        )

    model_values = {
        name: np.asarray([row[f"{name}_score"] for row in subject_rows], dtype=float)
        for name in MODELS
    }
    model_summaries = {
        name: _summarize_values(
            values, bootstraps, int(config["seed"]) + index
        )
        for index, (name, values) in enumerate(model_values.items())
    }
    primary_values = (
        model_values["generative_state_space"]
        - model_values["lagged_linear_ridge"]
    )
    primary = _summarize_values(
        primary_values, bootstraps, int(config["seed"]) + 10
    )
    secondary = {
        "lagged_linear_ridge_minus_static_lead_field": _summarize_values(
            model_values["lagged_linear_ridge"] - model_values["static_lead_field"],
            bootstraps,
            int(config["seed"]) + 11,
        ),
        "generative_state_space_minus_static_lead_field": _summarize_values(
            model_values["generative_state_space"]
            - model_values["static_lead_field"],
            bootstraps,
            int(config["seed"]) + 12,
        ),
    }
    ranking = sorted(
        (
            {
                "model": name,
                "equal_subject_mean_score": float(summary["observed_mean"]),
            }
            for name, summary in model_summaries.items()
        ),
        key=lambda row: (-row["equal_subject_mean_score"], row["model"]),
    )

    total_trials = int(sum(row["clean_trial_count"] for row in session_rows))
    result = {
        "subject_count": len(subject_rows),
        "session_count": len(session_rows),
        "clean_trial_count": total_trials,
        "models": list(MODELS),
        "model_summaries": model_summaries,
        "model_ranking": ranking,
        "generative_minus_ridge": primary,
        "secondary_contrasts": secondary,
        "ridge_outer_fold_selection_counts": {
            key: int(value)
            for key, value in sorted(
                Counter(
                    f"lags={row['selected_lag_taps_samples']};alpha={float(row['selected_alpha']):g}"
                    for row in fold_rows
                ).items()
            )
        },
        "maximum_difference_from_generative_analysis": maximum_difference,
        "bootstrap_resample_count": bootstraps,
    }
    _write_csv(output / "ridge_candidate_scores.csv", candidate_rows)
    _write_csv(output / "outer_fold_hyperparameters.csv", fold_rows)
    _write_csv(output / "session_model_scores.csv", session_rows)
    _write_csv(output / "subject_model_scores.csv", subject_rows)
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, result_path
