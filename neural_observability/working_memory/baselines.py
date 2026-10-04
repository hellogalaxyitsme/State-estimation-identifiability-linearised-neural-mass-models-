"""Lagged ridge regression and nested hyperparameter selection for the mapping benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from neural_observability.working_memory.generative_model import interval_mask


@dataclass(frozen=True)
class LaggedRidgeCandidate:
    """One fixed multiscale lag/ridge candidate and its held-out score."""

    lag_taps: tuple[int, ...]
    alpha: float
    alpha_absolute: float
    score: float
    coefficient_frobenius_norm: float
    training_observation_count: int
    scoring_observation_count: int


def encoding_standardize_channels(
    trials: np.ndarray, times: np.ndarray, encoding_interval: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize channels using only trials and samples in the encoding window."""
    values = np.asarray(trials, dtype=np.float64)
    sample_times = np.asarray(times, dtype=np.float64)
    if values.ndim != 3 or sample_times.shape != (values.shape[2],):
        raise ValueError("trials must be trial x channel x sample and match times")
    if not np.all(np.isfinite(values)):
        raise ValueError("trials must be finite")
    encoding = interval_mask(sample_times, encoding_interval)
    if int(np.sum(encoding)) < 2:
        raise ValueError("encoding interval is too short")
    selected = values[:, :, encoding]
    mean = selected.mean(axis=(0, 2))
    scale = selected.std(axis=(0, 2), ddof=0)
    scale = np.maximum(scale, np.finfo(np.float64).eps)
    standardized = (values - mean[None, :, None]) / scale[None, :, None]
    return standardized, mean, scale


def gaussian_mean_log_likelihood_gain(
    observed: np.ndarray, predicted: np.ndarray, covariance: np.ndarray
) -> float:
    """Gaussian mean-model log-likelihood gain per dimension and observation."""
    y = np.asarray(observed, dtype=np.float64)
    mean = np.asarray(predicted, dtype=np.float64)
    noise = np.asarray(covariance, dtype=np.float64)
    if y.ndim != 2 or mean.shape != y.shape:
        raise ValueError("observed and predicted must be dimension x observation matrices")
    if noise.shape != (y.shape[0], y.shape[0]):
        raise ValueError("covariance has incompatible dimensions")
    if not all(np.all(np.isfinite(value)) for value in (y, mean, noise)):
        raise ValueError("likelihood inputs must be finite")
    np.linalg.cholesky(noise)
    precision_observed = np.linalg.solve(noise, y)
    precision_prediction = np.linalg.solve(noise, mean)
    cross = float(np.mean(np.sum(mean * precision_observed, axis=0)))
    quadratic = float(np.mean(np.sum(mean * precision_prediction, axis=0)))
    return 0.5 * (2.0 * cross - quadratic) / y.shape[0]


def _time_indices(
    times: np.ndarray,
    interval: Sequence[float],
    maximum_lag: int,
    *,
    training: bool,
) -> np.ndarray:
    mask = interval_mask(times, interval, closed_right=not training)
    indices = np.flatnonzero(mask)
    if training:
        # Every target and every lagged predictor must lie inside encoding.
        valid = indices >= maximum_lag
        valid[valid] &= mask[indices[valid] - maximum_lag]
        indices = indices[valid]
    else:
        indices = indices[indices >= maximum_lag]
    if len(indices) < 2:
        raise ValueError("The lagged interval contains too few valid samples")
    return indices


def _lagged_design(
    standardized_predictors: np.ndarray,
    time_indices: np.ndarray,
    lag_taps: tuple[int, ...],
) -> np.ndarray:
    blocks = [standardized_predictors[:, :, time_indices - lag] for lag in lag_taps]
    stacked = np.concatenate(blocks, axis=1)
    return stacked.transpose(0, 2, 1).reshape(-1, stacked.shape[1])


def lagged_ridge_candidate_scores(
    predictor_trials: np.ndarray,
    scalp_trials: np.ndarray,
    times: np.ndarray,
    encoding_interval: Sequence[float],
    scoring_interval: Sequence[float],
    evaluation_scalp_mean: np.ndarray,
    evaluation_covariance: np.ndarray,
    lag_tap_sets: Sequence[Sequence[int]],
    ridge_alphas: Sequence[float],
) -> list[LaggedRidgeCandidate]:
    """Fit the fixed ridge grid on encoding and score maintenance predictions."""
    predictors, _, _ = encoding_standardize_channels(
        predictor_trials, times, encoding_interval
    )
    scalp = np.asarray(scalp_trials, dtype=np.float64)
    sample_times = np.asarray(times, dtype=np.float64)
    scalp_mean = np.asarray(evaluation_scalp_mean, dtype=np.float64)
    covariance = np.asarray(evaluation_covariance, dtype=np.float64)
    if scalp.ndim != 3 or scalp.shape[0] != predictors.shape[0] or scalp.shape[2] != predictors.shape[2]:
        raise ValueError("predictor and scalp trials must share trial/sample axes")
    if scalp_mean.shape != (scalp.shape[1],):
        raise ValueError("evaluation scalp mean has the wrong dimension")
    if covariance.shape != (scalp.shape[1], scalp.shape[1]):
        raise ValueError("evaluation covariance has the wrong dimension")
    centered_scalp = scalp - scalp_mean[None, :, None]
    candidates: list[LaggedRidgeCandidate] = []
    seen_lags: set[tuple[int, ...]] = set()
    for lag_source in lag_tap_sets:
        lags = tuple(int(value) for value in lag_source)
        if (
            not lags
            or lags in seen_lags
            or tuple(sorted(set(lags))) != lags
            or lags[0] != 0
            or any(value < 0 for value in lags)
        ):
            raise ValueError("lag taps must be unique increasing tuples beginning at zero")
        seen_lags.add(lags)
        training_indices = _time_indices(
            sample_times, encoding_interval, max(lags), training=True
        )
        scoring_indices = _time_indices(
            sample_times, scoring_interval, max(lags), training=False
        )
        x_train = _lagged_design(predictors, training_indices, lags)
        x_score = _lagged_design(predictors, scoring_indices, lags)
        y_train = centered_scalp[:, :, training_indices].transpose(0, 2, 1).reshape(
            -1, scalp.shape[1]
        )
        y_score = centered_scalp[:, :, scoring_indices].transpose(0, 2, 1).reshape(
            -1, scalp.shape[1]
        )
        x_mean = x_train.mean(axis=0)
        y_mean = y_train.mean(axis=0)
        x_train = x_train - x_mean[None, :]
        y_train = y_train - y_mean[None, :]
        x_score = x_score - x_mean[None, :]
        gram = x_train.T @ x_train / len(x_train)
        cross = x_train.T @ y_train / len(x_train)
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2.0)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        rotated_cross = eigenvectors.T @ cross
        ridge_scale = max(
            float(np.trace(gram) / gram.shape[0]), np.finfo(np.float64).eps
        )
        observed_matrix = y_score.T
        for alpha_source in ridge_alphas:
            alpha = float(alpha_source)
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError("ridge alpha multipliers must be positive and finite")
            alpha_absolute = alpha * ridge_scale
            coefficient = eigenvectors @ (
                rotated_cross / (eigenvalues[:, None] + alpha_absolute)
            )
            prediction = y_mean[None, :] + x_score @ coefficient
            score = gaussian_mean_log_likelihood_gain(
                observed_matrix, prediction.T, covariance
            )
            candidates.append(
                LaggedRidgeCandidate(
                    lag_taps=lags,
                    alpha=alpha,
                    alpha_absolute=float(alpha_absolute),
                    score=float(score),
                    coefficient_frobenius_norm=float(np.linalg.norm(coefficient)),
                    training_observation_count=len(x_train),
                    scoring_observation_count=len(x_score),
                )
            )
    return candidates


def nested_loso_hyperparameter_selection(
    candidate_scores: np.ndarray,
    subjects: Sequence[str],
    lag_tap_sets: Sequence[Sequence[int]],
    ridge_alphas: Sequence[float],
) -> list[dict[str, Any]]:
    """Select one candidate per held-out subject using training subjects only.

    ``candidate_scores`` is session x lag-set x alpha. Sessions are averaged
    within each training subject before the equal-training-subject objective.
    """
    scores = np.asarray(candidate_scores, dtype=np.float64)
    subject_vector = np.asarray([str(value) for value in subjects], dtype=str)
    lag_sets = [tuple(int(value) for value in source) for source in lag_tap_sets]
    alphas = [float(value) for value in ridge_alphas]
    if scores.shape != (len(subject_vector), len(lag_sets), len(alphas)):
        raise ValueError("candidate_scores has incompatible dimensions")
    if not np.all(np.isfinite(scores)) or len(set(subject_vector)) < 2:
        raise ValueError("scores must be finite and contain at least two subjects")
    output: list[dict[str, Any]] = []
    for held_out in sorted(set(subject_vector)):
        training_subjects = sorted(set(subject_vector) - {held_out})
        objectives = np.empty((len(lag_sets), len(alphas)), dtype=float)
        for lag_index in range(len(lag_sets)):
            for alpha_index in range(len(alphas)):
                subject_means = [
                    float(
                        np.mean(
                            scores[
                                subject_vector == training_subject,
                                lag_index,
                                alpha_index,
                            ]
                        )
                    )
                    for training_subject in training_subjects
                ]
                objectives[lag_index, alpha_index] = float(np.mean(subject_means))
        ranked = sorted(
            (
                (
                    -float(objectives[lag_index, alpha_index]),
                    len(lag_sets[lag_index]),
                    -alphas[alpha_index],
                    lag_index,
                    alpha_index,
                )
                for lag_index in range(len(lag_sets))
                for alpha_index in range(len(alphas))
            )
        )
        _, _, _, lag_index, alpha_index = ranked[0]
        test_mask = subject_vector == held_out
        output.append(
            {
                "held_out_subject": held_out,
                "training_subjects": training_subjects,
                "training_subject_count": len(training_subjects),
                "selected_lag_index": lag_index,
                "selected_lag_taps": lag_sets[lag_index],
                "selected_alpha_index": alpha_index,
                "selected_alpha": alphas[alpha_index],
                "training_equal_subject_mean_score": float(
                    objectives[lag_index, alpha_index]
                ),
                "test_session_indices": np.flatnonzero(test_mask),
                "test_session_scores": scores[test_mask, lag_index, alpha_index],
            }
        )
    return output
