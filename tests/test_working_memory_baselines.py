from __future__ import annotations

import numpy as np

from neural_observability.working_memory.baselines import (
    encoding_standardize_channels,
    gaussian_mean_log_likelihood_gain,
    lagged_ridge_candidate_scores,
    nested_loso_hyperparameter_selection,
)


def test_encoding_standardization_uses_only_encoding_samples() -> None:
    times = np.arange(12, dtype=float)
    trials = np.arange(2 * 2 * 12, dtype=float).reshape(2, 2, 12)
    standardized, mean, scale = encoding_standardize_channels(
        trials, times, (2.0, 7.0)
    )
    encoding = (times >= 2.0) & (times < 7.0)
    np.testing.assert_allclose(
        standardized[:, :, encoding].mean(axis=(0, 2)), 0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        standardized[:, :, encoding].std(axis=(0, 2)), 1.0, atol=1e-15
    )
    changed = trials.copy()
    changed[:, :, times >= 7.0] += 10000
    _, changed_mean, changed_scale = encoding_standardize_channels(
        changed, times, (2.0, 7.0)
    )
    np.testing.assert_array_equal(mean, changed_mean)
    np.testing.assert_array_equal(scale, changed_scale)


def test_gaussian_gain_matches_expanded_quadratic_definition() -> None:
    rng = np.random.default_rng(2)
    observed = rng.normal(size=(4, 30))
    predicted = rng.normal(size=(4, 30))
    matrix = rng.normal(size=(4, 4))
    covariance = matrix @ matrix.T + 0.5 * np.eye(4)
    actual = gaussian_mean_log_likelihood_gain(observed, predicted, covariance)
    residual = observed - predicted
    expected = 0.5 * np.mean(
        np.sum(observed * np.linalg.solve(covariance, observed), axis=0)
        - np.sum(residual * np.linalg.solve(covariance, residual), axis=0)
    ) / observed.shape[0]
    assert np.isclose(actual, expected, atol=1e-14)
    assert gaussian_mean_log_likelihood_gain(
        observed, np.zeros_like(observed), covariance
    ) == 0.0


def test_lagged_ridge_detects_a_held_out_one_sample_relationship() -> None:
    rng = np.random.default_rng(8)
    n_trials, n_samples = 20, 100
    times = np.arange(n_samples, dtype=float) / 20.0
    predictor = rng.normal(size=(n_trials, 1, n_samples))
    scalp = np.zeros((n_trials, 1, n_samples))
    scalp[:, 0, 1:] = 1.5 * predictor[:, 0, :-1]
    scalp += rng.normal(scale=0.05, size=scalp.shape)
    candidates = lagged_ridge_candidate_scores(
        predictor,
        scalp,
        times,
        (0.5, 2.0),
        (2.5, 4.5),
        np.asarray([0.0]),
        np.asarray([[1.0]]),
        [[0], [0, 1]],
        [0.001],
    )
    assert len(candidates) == 2
    assert candidates[1].lag_taps == (0, 1)
    assert candidates[1].score > candidates[0].score + 0.5
    assert candidates[1].training_observation_count > 0
    assert candidates[1].scoring_observation_count > 0


def test_outer_loso_selection_is_invariant_to_held_out_subject_scores() -> None:
    subjects = np.asarray(["A", "A", "B", "C"])
    scores = np.asarray(
        [
            [[10.0], [-10.0]],
            [[10.0], [-10.0]],
            [[0.0], [2.0]],
            [[0.0], [3.0]],
        ]
    )
    baseline = nested_loso_hyperparameter_selection(
        scores, subjects, [[0], [0, 1]], [0.1]
    )
    changed = scores.copy()
    changed[subjects == "A"] *= -1000
    altered = nested_loso_hyperparameter_selection(
        changed, subjects, [[0], [0, 1]], [0.1]
    )
    baseline_a = next(row for row in baseline if row["held_out_subject"] == "A")
    altered_a = next(row for row in altered if row["held_out_subject"] == "A")
    assert baseline_a["selected_lag_index"] == 1
    assert altered_a["selected_lag_index"] == 1
    assert baseline_a["training_equal_subject_mean_score"] == altered_a[
        "training_equal_subject_mean_score"
    ]
    assert "A" not in baseline_a["training_subjects"]


def test_training_objective_weights_subjects_not_sessions() -> None:
    subjects = np.asarray(["A", "B", "B", "B", "B", "C"])
    scores = np.zeros((len(subjects), 2, 1))
    # For held-out C: subject A strongly prefers candidate 1; B's four sessions
    # weakly prefer candidate 0. Equal-subject weighting must select candidate 1.
    scores[subjects == "A", 1, 0] = 10.0
    scores[subjects == "B", 0, 0] = 1.0
    selected = nested_loso_hyperparameter_selection(
        scores, subjects, [[0], [0, 1]], [1.0]
    )
    held_out_c = next(row for row in selected if row["held_out_subject"] == "C")
    assert held_out_c["selected_lag_index"] == 1
    assert held_out_c["training_subject_count"] == 2
