"""Generative state-space model of latent current dipoles observed intracranially and at the scalp.

    q[t + 1] = A q[t] + b + w[t]
    x[t]     = c_x + H q[t] + v_x[t]
    y[t]     = c_y + L diag(g) q[t] + u[t]

``q`` is a latent cortical current-dipole state, ``x`` is bipolar iEEG,
``y`` is average-referenced scalp EEG, ``H`` is an analytic bipolar
intracranial observation operator, and ``L`` is the independently constructed
scalp BEM lead field.  ``u`` includes unobserved cortical sources and sensor
noise and is marginalized with an encoding-only residual covariance.

All data-dependent parameters in this module are fit on the encoding interval.
The maintenance interval is reserved for held-out likelihood scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from neural_observability.working_memory.sources import BipolarSource


@dataclass(frozen=True)
class LatentDipoleFit:
    """Encoding-only parameters of the intracranial state-space model."""

    observation_mean: np.ndarray
    observation_scale: np.ndarray
    proxy_mean: np.ndarray
    proxy_scale: np.ndarray
    transition: np.ndarray
    intercept: np.ndarray
    observation_operator: np.ndarray
    observation_intercept: np.ndarray
    process_covariance: np.ndarray
    observation_covariance: np.ndarray
    kalman_gain: np.ndarray
    transition_spectral_radius: float
    kalman_iterations: int
    kalman_converged: bool
    operator_condition_number: float


@dataclass(frozen=True)
class ScalpObservationFit:
    """Encoding-only parameters of the scalp generative observation model."""

    source_mean: np.ndarray
    scalp_mean: np.ndarray
    gains: np.ndarray
    nuisance_covariance: np.ndarray
    source_covariance: np.ndarray
    source_scalp_cross_covariance: np.ndarray
    gain_ridge_absolute: float
    encoding_variance_explained: float
    nuisance_effective_rank: float


def causal_filter_resample_trials(
    trials: np.ndarray,
    original_frequency_hz: float,
    target_frequency_hz: float,
    bandpass_hz: Sequence[float],
    order: int = 4,
) -> np.ndarray:
    """Polyphase-resample, then causally band-pass each trial.

    The anti-alias FIR has finite local support and is applied before the causal
    IIR.  The 0.5 s guard interval between the encoding and scoring periods is
    longer than the resampling kernel support for the sampling ratios used.
    Subtracting the first sample before zero-state filtering reduces trial-onset
    transients without introducing any future sample into the IIR output.
    """

    values = _validate_trials(trials, "trials")
    original = float(original_frequency_hz)
    target = float(target_frequency_hz)
    low, high = (float(value) for value in bandpass_hz)
    if not 0 < low < high < min(original, target) / 2:
        raise ValueError("Sampling frequencies and bandpass are incompatible")
    ratio = target / original
    fraction = Fraction(ratio).limit_denominator(100000)
    if not math.isclose(float(fraction), ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Sampling-rate ratio is not representable exactly enough")
    if fraction.numerator == fraction.denominator:
        resampled = np.array(values, copy=True)
    else:
        resampled = resample_poly(
            values, fraction.numerator, fraction.denominator, axis=-1
        )
    expected = int(round(values.shape[-1] * ratio))
    if resampled.shape[-1] != expected:
        raise RuntimeError("Polyphase resampling produced an unexpected length")
    sos = butter(int(order), [low, high], btype="bandpass", fs=target, output="sos")
    centered = resampled - resampled[..., :1]
    filtered = sosfilt(sos, centered, axis=-1)
    if not np.isfinite(filtered).all():
        raise RuntimeError("Causal filtering produced non-finite samples")
    return filtered


def _validate_trials(trials: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(trials, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (trials, channels, samples)")
    if min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def interval_mask(
    times: np.ndarray,
    interval: Sequence[float],
    *,
    closed_right: bool = False,
) -> np.ndarray:
    """Return a half-open interval mask unless ``closed_right`` is requested."""

    vector = np.asarray(times, dtype=np.float64)
    if vector.ndim != 1 or len(interval) != 2:
        raise ValueError("times must be one-dimensional and interval must have two values")
    lower, upper = (float(interval[0]), float(interval[1]))
    if not lower < upper:
        raise ValueError("interval lower bound must be smaller than upper bound")
    if closed_right:
        return (vector >= lower) & (vector <= upper)
    return (vector >= lower) & (vector < upper)


def average_reference_basis(n_channels: int) -> np.ndarray:
    """Return an orthonormal basis for the average-reference subspace."""

    if n_channels < 2:
        raise ValueError("at least two scalp channels are required")
    centering = np.eye(n_channels) - np.ones((n_channels, n_channels)) / n_channels
    eigenvalues, eigenvectors = np.linalg.eigh(centering)
    basis = eigenvectors[:, eigenvalues > 0.5]
    if basis.shape != (n_channels, n_channels - 1):
        raise RuntimeError("failed to construct the average-reference basis")
    return basis


def analytic_bipolar_operator(
    sources: Sequence[BipolarSource],
    electrode_coordinates_mm: Mapping[str, np.ndarray],
    *,
    softening_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an analytic bipolar-iEEG current-dipole operator.

    For unit dipole ``d_j`` at ``r_j``, the infinite homogeneous conductor
    potential is proportional to

        d_j dot (r - r_j) / (||r - r_j||^2 + eps^2)^(3/2).

    The conductivity constant is common to all entries and is absorbed by the
    fitted observation units.  A bipolar row is the lower-contact potential
    minus the upper-contact potential.  Rows are normalized only after the raw
    physically signed operator has been constructed.
    """

    if not sources:
        raise ValueError("at least one source is required")
    if softening_mm <= 0:
        raise ValueError("softening_mm must be positive")

    n_sources = len(sources)
    raw = np.empty((n_sources, n_sources), dtype=np.float64)

    def potential(contact: str, source: BipolarSource) -> float:
        if contact not in electrode_coordinates_mm:
            raise KeyError(f"missing coordinate for contact {contact}")
        coordinate = np.asarray(electrode_coordinates_mm[contact], dtype=np.float64)
        midpoint = np.asarray(source.midpoint_mni_mm, dtype=np.float64)
        direction = np.asarray(source.direction_mni, dtype=np.float64)
        if coordinate.shape != (3,) or midpoint.shape != (3,) or direction.shape != (3,):
            raise ValueError("contact, midpoint, and direction coordinates must be 3-vectors")
        displacement = coordinate - midpoint
        radius_squared = float(displacement @ displacement + softening_mm**2)
        return float(direction @ displacement) / radius_squared**1.5

    for row, measured in enumerate(sources):
        for column, source in enumerate(sources):
            raw[row, column] = potential(measured.lower_name, source) - potential(
                measured.upper_name, source
            )

    row_norms = np.linalg.norm(raw, axis=1)
    if np.any(row_norms <= np.finfo(np.float64).eps):
        raise ValueError("analytic bipolar operator contains a zero row")
    normalized = raw / row_norms[:, None]
    if not np.isfinite(normalized).all():
        raise ValueError("analytic bipolar operator is not finite")
    return raw, normalized


def _loaded_covariance(samples: np.ndarray, loading: float) -> np.ndarray:
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("covariance samples must have shape (features, observations)")
    if loading <= 0:
        raise ValueError("covariance loading must be positive")
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / centered.shape[1]
    scale = float(np.trace(covariance) / covariance.shape[0])
    scale = max(scale, np.finfo(np.float64).eps)
    covariance = covariance + loading * scale * np.eye(covariance.shape[0])
    return (covariance + covariance.T) / 2.0


def _stable_affine_var1(
    source_trials: np.ndarray,
    transition_mask: np.ndarray,
    *,
    ridge: float,
    maximum_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ridge <= 0 or not 0 < maximum_radius < 1:
        raise ValueError("ridge must be positive and maximum_radius must lie in (0, 1)")
    mask = np.asarray(transition_mask, dtype=bool)
    if mask.ndim != 1 or mask.size != source_trials.shape[-1] - 1 or not mask.any():
        raise ValueError("transition_mask does not identify any valid transitions")

    predictors = source_trials[:, :, :-1][:, :, mask].transpose(0, 2, 1).reshape(-1, source_trials.shape[1])
    targets = source_trials[:, :, 1:][:, :, mask].transpose(0, 2, 1).reshape(-1, source_trials.shape[1])
    predictor_mean = predictors.mean(axis=0)
    target_mean = targets.mean(axis=0)
    x = predictors - predictor_mean
    y = targets - target_mean
    gram = x.T @ x / len(x)
    cross = y.T @ x / len(x)
    ridge_absolute = ridge * max(float(np.trace(gram) / gram.shape[0]), np.finfo(float).eps)
    transition = np.linalg.solve(gram + ridge_absolute * np.eye(gram.shape[0]), cross.T).T

    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius > maximum_radius:
        transition *= maximum_radius / radius
    intercept = target_mean - transition @ predictor_mean
    residuals = (targets - predictors @ transition.T - intercept).T
    return transition, intercept, residuals


def _steady_state_kalman_gain(
    transition: np.ndarray,
    observation: np.ndarray,
    process_covariance: np.ndarray,
    observation_covariance: np.ndarray,
    *,
    maximum_iterations: int = 2000,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, int, bool]:
    n_states = transition.shape[0]
    covariance = np.array(process_covariance, copy=True)
    identity = np.eye(n_states)
    gain = np.zeros((n_states, observation.shape[0]))

    for iteration in range(1, maximum_iterations + 1):
        predicted = transition @ covariance @ transition.T + process_covariance
        innovation_covariance = observation @ predicted @ observation.T + observation_covariance
        gain = np.linalg.solve(innovation_covariance, observation @ predicted).T
        correction = identity - gain @ observation
        updated = (
            correction @ predicted @ correction.T
            + gain @ observation_covariance @ gain.T
        )
        updated = (updated + updated.T) / 2.0
        denominator = max(float(np.linalg.norm(covariance, ord="fro")), 1.0)
        if np.linalg.norm(updated - covariance, ord="fro") / denominator < tolerance:
            return gain, iteration, True
        covariance = updated
    return gain, maximum_iterations, False


def fit_latent_dipole_model(
    bipolar_trials: np.ndarray,
    times: np.ndarray,
    encoding_interval: Sequence[float],
    observation_operator: np.ndarray,
    *,
    inverse_ridge: float = 0.1,
    transition_ridge: float = 0.1,
    maximum_transition_radius: float = 0.98,
    process_loading: float = 0.05,
    observation_loading: float = 0.05,
) -> tuple[LatentDipoleFit, np.ndarray]:
    """Fit the encoding-only latent model and causally filter all trials."""

    observations = _validate_trials(bipolar_trials, "bipolar_trials")
    sample_times = np.asarray(times, dtype=np.float64)
    operator = np.asarray(observation_operator, dtype=np.float64)
    n_trials, n_channels, n_samples = observations.shape
    if sample_times.shape != (n_samples,):
        raise ValueError("times must match the trial sample axis")
    if operator.shape[0] != n_channels or operator.shape[1] < 1:
        raise ValueError("observation_operator has incompatible dimensions")
    if inverse_ridge <= 0:
        raise ValueError("inverse_ridge must be positive")

    encoding = interval_mask(sample_times, encoding_interval)
    if encoding.sum() < 3:
        raise ValueError("encoding interval is too short")
    encoding_values = observations[:, :, encoding]
    observation_mean = encoding_values.mean(axis=(0, 2))
    observation_scale = encoding_values.std(axis=(0, 2), ddof=0)
    observation_scale = np.maximum(observation_scale, np.finfo(np.float64).eps)
    standardized = (observations - observation_mean[None, :, None]) / observation_scale[None, :, None]

    gram = operator.T @ operator
    ridge_absolute = inverse_ridge * max(float(np.trace(gram) / gram.shape[0]), np.finfo(float).eps)
    inverse_map = np.linalg.solve(
        gram + ridge_absolute * np.eye(gram.shape[0]), operator.T
    )
    proxy = np.einsum("sc,nct->nst", inverse_map, standardized, optimize=True)
    proxy_encoding = proxy[:, :, encoding]
    proxy_mean = proxy_encoding.mean(axis=(0, 2))
    proxy_scale = proxy_encoding.std(axis=(0, 2), ddof=0)
    proxy_scale = np.maximum(proxy_scale, np.finfo(np.float64).eps)
    standardized_proxy = (proxy - proxy_mean[None, :, None]) / proxy_scale[None, :, None]

    transition_encoding = encoding[:-1] & encoding[1:]
    transition, intercept, process_residuals = _stable_affine_var1(
        standardized_proxy,
        transition_encoding,
        ridge=transition_ridge,
        maximum_radius=maximum_transition_radius,
    )
    process_covariance = _loaded_covariance(process_residuals, process_loading)

    effective_operator = operator * proxy_scale[None, :]
    # q_raw = proxy_mean + diag(proxy_scale) q_standardized, hence the
    # observation intercept has the same (not opposite) sign as proxy_mean.
    observation_intercept = operator @ proxy_mean
    fitted_observations = (
        observation_intercept[None, :, None]
        + np.einsum("cs,nst->nct", effective_operator, standardized_proxy, optimize=True)
    )
    observation_residuals = (standardized - fitted_observations)[:, :, encoding]
    observation_residuals = observation_residuals.transpose(1, 0, 2).reshape(n_channels, -1)
    observation_covariance = _loaded_covariance(
        observation_residuals, observation_loading
    )

    gain, kalman_iterations, kalman_converged = _steady_state_kalman_gain(
        transition,
        effective_operator,
        process_covariance,
        observation_covariance,
    )
    if not kalman_converged:
        raise RuntimeError("steady-state Kalman covariance did not converge")

    filtered = np.empty((n_trials, operator.shape[1], n_samples), dtype=np.float64)
    state = np.zeros((n_trials, operator.shape[1]), dtype=np.float64)
    for sample in range(n_samples):
        prior = state @ transition.T + intercept[None, :]
        prediction = prior @ effective_operator.T + observation_intercept[None, :]
        innovation = standardized[:, :, sample] - prediction
        state = prior + innovation @ gain.T
        filtered[:, :, sample] = state

    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    condition = float(np.linalg.cond(operator))
    fit = LatentDipoleFit(
        observation_mean=observation_mean,
        observation_scale=observation_scale,
        proxy_mean=proxy_mean,
        proxy_scale=proxy_scale,
        transition=transition,
        intercept=intercept,
        observation_operator=effective_operator,
        observation_intercept=observation_intercept,
        process_covariance=process_covariance,
        observation_covariance=observation_covariance,
        kalman_gain=gain,
        transition_spectral_radius=radius,
        kalman_iterations=kalman_iterations,
        kalman_converged=kalman_converged,
        operator_condition_number=condition,
    )
    return fit, filtered


def _gain_solution_from_statistics(
    lead_field: np.ndarray,
    source_covariance: np.ndarray,
    source_scalp_cross_covariance: np.ndarray,
    gain_ridge: float,
) -> tuple[np.ndarray, float]:
    lead = np.asarray(lead_field, dtype=np.float64)
    if lead.ndim != 2 or lead.shape[1] != source_covariance.shape[0]:
        raise ValueError("lead_field and source covariance dimensions do not agree")
    if source_scalp_cross_covariance.shape != (lead.shape[1], lead.shape[0]):
        raise ValueError("source-scalp cross covariance has incompatible dimensions")
    if gain_ridge <= 0:
        raise ValueError("gain_ridge must be positive")

    gain_gram = (lead.T @ lead) * source_covariance
    right_hand_side = np.einsum(
        "ds,sd->s", lead, source_scalp_cross_covariance, optimize=True
    )
    diagonal = np.diag(gain_gram)
    positive = diagonal[diagonal > np.finfo(np.float64).eps]
    scale = float(np.median(positive)) if positive.size else 1.0
    ridge_absolute = gain_ridge * scale
    gains = np.linalg.solve(
        gain_gram + ridge_absolute * np.eye(gain_gram.shape[0]),
        right_hand_side,
    )
    return gains, ridge_absolute


def fit_scalp_observation_model(
    source_trials: np.ndarray,
    scalp_trials: np.ndarray,
    times: np.ndarray,
    encoding_interval: Sequence[float],
    lead_field: np.ndarray,
    *,
    gain_ridge: float = 0.1,
    nuisance_loading: float = 0.1,
) -> ScalpObservationFit:
    """Fit source gains and unobserved-source covariance on encoding data."""

    sources = _validate_trials(source_trials, "source_trials")
    scalp = _validate_trials(scalp_trials, "scalp_trials")
    lead = np.asarray(lead_field, dtype=np.float64)
    if sources.shape[0] != scalp.shape[0] or sources.shape[2] != scalp.shape[2]:
        raise ValueError("source and scalp trials must share trial and sample axes")
    if lead.shape != (scalp.shape[1], sources.shape[1]):
        raise ValueError("lead_field has incompatible dimensions")

    encoding = interval_mask(np.asarray(times, dtype=np.float64), encoding_interval)
    source_mean = sources[:, :, encoding].mean(axis=(0, 2))
    scalp_mean = scalp[:, :, encoding].mean(axis=(0, 2))
    centered_sources = sources - source_mean[None, :, None]
    centered_scalp = scalp - scalp_mean[None, :, None]

    q = centered_sources[:, :, encoding].transpose(1, 0, 2).reshape(sources.shape[1], -1)
    y = centered_scalp[:, :, encoding].transpose(1, 0, 2).reshape(scalp.shape[1], -1)
    source_covariance = q @ q.T / q.shape[1]
    source_scalp_cross = q @ y.T / q.shape[1]
    gains, ridge_absolute = _gain_solution_from_statistics(
        lead, source_covariance, source_scalp_cross, gain_ridge
    )

    predicted = (lead * gains[None, :]) @ q
    residuals = y - predicted
    nuisance_covariance = _loaded_covariance(residuals, nuisance_loading)
    baseline_variance = float(np.sum(y**2))
    variance_explained = (
        1.0 - float(np.sum(residuals**2)) / baseline_variance
        if baseline_variance > 0
        else 0.0
    )
    eigenvalues = np.linalg.eigvalsh(nuisance_covariance)
    effective_rank = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
    return ScalpObservationFit(
        source_mean=source_mean,
        scalp_mean=scalp_mean,
        gains=gains,
        nuisance_covariance=nuisance_covariance,
        source_covariance=source_covariance,
        source_scalp_cross_covariance=source_scalp_cross,
        gain_ridge_absolute=ridge_absolute,
        encoding_variance_explained=variance_explained,
        nuisance_effective_rank=effective_rank,
    )


def refit_gains_for_lead_field(
    fit: ScalpObservationFit,
    lead_field: np.ndarray,
    *,
    gain_ridge: float,
) -> np.ndarray:
    """Refit gains for a control geometry using only the encoding statistics."""

    gains, _ = _gain_solution_from_statistics(
        np.asarray(lead_field, dtype=np.float64),
        fit.source_covariance,
        fit.source_scalp_cross_covariance,
        gain_ridge,
    )
    return gains


def predict_scalp_trials(
    source_trials: np.ndarray,
    lead_field: np.ndarray,
    fit: ScalpObservationFit,
    *,
    gains: np.ndarray | None = None,
) -> np.ndarray:
    sources = _validate_trials(source_trials, "source_trials")
    lead = np.asarray(lead_field, dtype=np.float64)
    selected_gains = fit.gains if gains is None else np.asarray(gains, dtype=np.float64)
    if lead.shape != (fit.scalp_mean.size, fit.source_mean.size):
        raise ValueError("lead_field has incompatible dimensions")
    if selected_gains.shape != (fit.source_mean.size,):
        raise ValueError("gains have incompatible dimensions")
    centered = sources - fit.source_mean[None, :, None]
    return np.einsum(
        "ds,s,nst->ndt", lead, selected_gains, centered, optimize=True
    )


def held_out_log_likelihood_gain(
    scalp_trials: np.ndarray,
    predicted_mean: np.ndarray,
    times: np.ndarray,
    scoring_interval: Sequence[float],
    fit: ScalpObservationFit,
) -> float:
    """Return per-dimension/sample Gaussian LL gain over nuisance-only baseline."""

    scalp = _validate_trials(scalp_trials, "scalp_trials")
    prediction = _validate_trials(predicted_mean, "predicted_mean")
    if scalp.shape != prediction.shape:
        raise ValueError("scalp_trials and predicted_mean shapes must match")
    scoring = interval_mask(
        np.asarray(times, dtype=np.float64), scoring_interval, closed_right=True
    )
    observed = scalp - fit.scalp_mean[None, :, None]
    observed = observed[:, :, scoring].transpose(1, 0, 2).reshape(scalp.shape[1], -1)
    predicted = prediction[:, :, scoring].transpose(1, 0, 2).reshape(scalp.shape[1], -1)
    residual = observed - predicted
    solved_observed = np.linalg.solve(fit.nuisance_covariance, observed)
    solved_residual = np.linalg.solve(fit.nuisance_covariance, residual)
    baseline_quadratic = np.sum(observed * solved_observed, axis=0)
    residual_quadratic = np.sum(residual * solved_residual, axis=0)
    return float(0.5 * np.mean(baseline_quadratic - residual_quadratic) / scalp.shape[1])


def circularly_shift_prediction(
    prediction: np.ndarray,
    times: np.ndarray,
    scoring_interval: Sequence[float],
    offset_samples: int,
) -> np.ndarray:
    """Shift only the held-out scoring segment, independently within each trial."""

    shifted = np.array(prediction, dtype=np.float64, copy=True)
    mask = interval_mask(np.asarray(times, dtype=np.float64), scoring_interval, closed_right=True)
    length = int(mask.sum())
    if length < 2 or not 0 < offset_samples < length:
        raise ValueError("offset_samples must lie strictly inside the scoring segment")
    shifted[:, :, mask] = np.roll(shifted[:, :, mask], offset_samples, axis=2)
    return shifted


def permute_prediction_trials(prediction: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Apply a trial-identity control permutation to a predicted scalp array."""

    array = _validate_trials(prediction, "prediction")
    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (array.shape[0],) or not np.array_equal(
        np.sort(order), np.arange(array.shape[0])
    ):
        raise ValueError("permutation must contain every trial index exactly once")
    return array[order]
