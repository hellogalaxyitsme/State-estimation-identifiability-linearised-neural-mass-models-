"""Finite-horizon information maps, invariant spectra, and benchmark metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import expm_multiply
from scipy.spatial.distance import cdist
from scipy.stats import norm, spearmanr


LOCAL_OUTPUT = np.array([[0.0, 1.0, -1.0, 0.0, 0.0, 0.0]])


def symmetric_psd_sqrt(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Return the symmetric square root of a numerically PSD matrix."""
    matrix = np.asarray(matrix, dtype=float)
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    scale = max(1.0, float(np.max(np.abs(values))))
    if values.min() < -tolerance * scale:
        raise ValueError(f"Matrix is not PSD; minimum eigenvalue={values.min():.3e}")
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def spatial_covariance(positions_m: np.ndarray, specification: dict[str, Any]) -> np.ndarray:
    """Construct a unit-diagonal spatial covariance from the specification."""
    count = len(positions_m)
    kind = specification["kind"]
    if kind == "iid":
        return np.eye(count)
    if kind == "rbf":
        length_scale = float(specification["length_scale_m"])
        nugget = float(specification.get("nugget", 0.0))
        distances = cdist(positions_m, positions_m)
        covariance = np.exp(-0.5 * (distances / length_scale) ** 2)
        covariance = (1.0 - nugget) * covariance + nugget * np.eye(count)
        diagonal = np.sqrt(np.diag(covariance))
        return covariance / np.outer(diagonal, diagonal)
    raise ValueError(f"Unsupported spatial prior kind: {kind}")


def prior_sqrt(
    positions_m: np.ndarray,
    spatial_specification: dict[str, Any],
    local_standard_deviations: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``P^(1/2)`` for a separable spatial/local physical state prior."""
    local_std = np.asarray(local_standard_deviations, dtype=float)
    if local_std.shape != (6,) or np.any(local_std <= 0):
        raise ValueError("local_standard_deviations must contain six positive values")
    spatial = spatial_covariance(positions_m, spatial_specification)
    spatial_sqrt = symmetric_psd_sqrt(spatial)
    result = np.kron(spatial_sqrt, np.diag(local_std))
    return result, {
        "name": spatial_specification["name"],
        "kind": spatial_specification["kind"],
        "spatial_condition_number": float(np.linalg.cond(spatial)),
        "local_standard_deviations": local_std.tolist(),
    }


def calibrated_observation_matrix(
    lead_field: np.ndarray,
    local_standard_deviations: np.ndarray,
    target_median_static_snr: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Globally calibrate the source-to-sensor scale at a declared static SNR."""
    lead_field = np.asarray(lead_field, dtype=float)
    local_std = np.asarray(local_standard_deviations, dtype=float)
    local_covariance = np.diag(local_std**2)
    output_sd = float(np.sqrt(LOCAL_OUTPUT @ local_covariance @ LOCAL_OUTPUT.T).item())
    column_norms = np.linalg.norm(lead_field, axis=0)
    median_unscaled = float(np.median(column_norms) * output_sd)
    if median_unscaled <= 0:
        raise ValueError("Lead field has zero median source sensitivity")
    scale = float(target_median_static_snr / median_unscaled)
    calibrated_lead = scale * lead_field
    observation = np.kron(calibrated_lead, LOCAL_OUTPUT)
    calibrated_static = column_norms * scale * output_sd
    return observation, scale, {
        "target_median_static_snr": float(target_median_static_snr),
        "unscaled_median_static_norm": median_unscaled,
        "source_to_sensor_global_scale": scale,
        "calibrated_static_norm_min": float(calibrated_static.min()),
        "calibrated_static_norm_median": float(np.median(calibrated_static)),
        "calibrated_static_norm_max": float(calibrated_static.max()),
    }


def sampled_dynamic_map(a: np.ndarray, c: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute stacked ``C exp(A t_k)`` efficiently for equally spaced samples."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    times = np.asarray(times, dtype=float)
    if a.shape[0] != a.shape[1] or c.shape[1] != a.shape[0]:
        raise ValueError("Incompatible A and C dimensions")
    if len(times) < 2 or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing and nonnegative")
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-10, atol=1e-14):
        raise ValueError("sampled_dynamic_map currently requires equally spaced times")
    propagated = expm_multiply(
        a.T,
        c.T,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
        traceA=float(np.trace(a)),
    )
    return np.vstack([item.T for item in propagated])


def repeated_static_map(c: np.ndarray, sample_count: int) -> np.ndarray:
    """Static baseline that assumes every time point repeats the instant map."""
    return np.tile(np.asarray(c, dtype=float), (sample_count, 1))


def information_eigenvalues(dynamic_map: np.ndarray, p_sqrt: np.ndarray) -> np.ndarray:
    """Return the full invariant eigenvalue spectrum using singular values."""
    standardized_map = dynamic_map @ p_sqrt
    singular_values = np.linalg.svd(standardized_map, compute_uv=False)
    eigenvalues = np.zeros(p_sqrt.shape[0], dtype=float)
    eigenvalues[: len(singular_values)] = singular_values**2
    return np.sort(eigenvalues)[::-1]


def information_summary(
    dynamic_map: np.ndarray,
    p_sqrt: np.ndarray,
    relative_rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Summarize the coordinate-invariant finite-horizon information spectrum."""
    eigenvalues = information_eigenvalues(dynamic_map, p_sqrt)
    maximum = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    threshold = relative_rank_tolerance * maximum
    nonzero = eigenvalues[eigenvalues > threshold]
    total = float(eigenvalues.sum())
    if total > 0:
        proportions = eigenvalues[eigenvalues > np.finfo(float).tiny] / total
        proportions = proportions[proportions > 0]
        participation_ratio = total**2 / float(np.sum(eigenvalues**2))
        entropy_dimension = float(np.exp(-np.sum(proportions * np.log(proportions))))
    else:
        participation_ratio = 0.0
        entropy_dimension = 0.0
    condition_number = (
        float(nonzero[0] / nonzero[-1]) if len(nonzero) > 1 else 1.0
    )
    return {
        "state_dimension": int(p_sqrt.shape[0]),
        "sampled_observation_dimension": int(dynamic_map.shape[0]),
        "numerical_information_rank": int(len(nonzero)),
        "relative_rank_tolerance": relative_rank_tolerance,
        "information_trace": total,
        "participation_ratio": float(participation_ratio),
        "entropy_effective_dimension": entropy_dimension,
        "gaussian_mutual_information_nats": float(0.5 * np.sum(np.log1p(eigenvalues))),
        "normalized_posterior_variance": float(np.mean(1.0 / (1.0 + eigenvalues))),
        "nonzero_condition_number": condition_number,
        "largest_information_eigenvalue": maximum,
        "smallest_retained_information_eigenvalue": (
            float(nonzero[-1]) if len(nonzero) else 0.0
        ),
        "information_eigenvalues": eigenvalues.tolist(),
    }


def regional_event_matrix(
    region_count: int, local_standard_deviations: np.ndarray
) -> np.ndarray:
    """Return one prespecified standardized pyramidal-difference event per region."""
    standardized_local = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)
    physical_local = np.asarray(local_standard_deviations) * standardized_local
    events = np.zeros((6 * region_count, region_count), dtype=float)
    for region in range(region_count):
        events[6 * region : 6 * region + 6, region] = physical_local
    return events


def regional_event_scores(
    dynamic_map: np.ndarray,
    region_count: int,
    local_standard_deviations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute regional Mahalanobis distances and exact binary Bayes accuracies."""
    events = regional_event_matrix(region_count, local_standard_deviations)
    projected = dynamic_map @ events
    squared_distance = np.sum(projected**2, axis=0)
    return {
        "squared_mahalanobis_distance": squared_distance,
        "bayes_accuracy": norm.cdf(np.sqrt(squared_distance) / 2.0),
    }


def expected_standardized_mse(
    assumed_map: np.ndarray,
    true_map: np.ndarray,
    p_sqrt: np.ndarray,
) -> dict[str, float]:
    """Expected prior-whitened state MSE for a possibly mismatched Bayes estimator."""
    assumed = assumed_map @ p_sqrt
    truth = true_map @ p_sqrt
    state_count = assumed.shape[1]
    precision = np.eye(state_count) + assumed.T @ assumed
    gain = solve(precision, assumed.T, assume_a="pos", check_finite=False)
    bias_operator = gain @ truth - np.eye(state_count)
    actual_mse = (
        np.linalg.norm(bias_operator, ord="fro") ** 2
        + np.linalg.norm(gain, ord="fro") ** 2
    ) / state_count
    assumed_posterior = float(np.trace(solve(precision, np.eye(state_count), assume_a="pos")) / state_count)
    return {
        "expected_standardized_mse": float(actual_mse),
        "assumed_normalized_posterior_variance": assumed_posterior,
        "mismatch_calibration_ratio": float(actual_mse / assumed_posterior),
    }


def regional_prediction_benchmark(
    predicted_dynamic_score: np.ndarray,
    static_score: np.ndarray,
    true_dynamic_score: np.ndarray,
) -> dict[str, float]:
    """Compare regional detection ordering for dynamic and static predictions."""
    dynamic_correlation = spearmanr(predicted_dynamic_score, true_dynamic_score).statistic
    static_correlation = spearmanr(static_score, true_dynamic_score).statistic
    log_dynamic = np.corrcoef(
        np.log(np.maximum(predicted_dynamic_score, np.finfo(float).tiny)),
        np.log(np.maximum(true_dynamic_score, np.finfo(float).tiny)),
    )[0, 1]
    log_static = np.corrcoef(
        np.log(np.maximum(static_score, np.finfo(float).tiny)),
        np.log(np.maximum(true_dynamic_score, np.finfo(float).tiny)),
    )[0, 1]
    return {
        "dynamic_spearman": float(dynamic_correlation),
        "static_spearman": float(static_correlation),
        "dynamic_log_pearson": float(log_dynamic),
        "static_log_pearson": float(log_static),
        "spearman_improvement": float(dynamic_correlation - static_correlation),
    }
