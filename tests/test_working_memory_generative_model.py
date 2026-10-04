from __future__ import annotations

import numpy as np

from neural_observability.working_memory.sources import BipolarSource
from neural_observability.working_memory.generative_model import (
    analytic_bipolar_operator,
    average_reference_basis,
    causal_filter_resample_trials,
    circularly_shift_prediction,
    fit_latent_dipole_model,
    fit_scalp_observation_model,
    held_out_log_likelihood_gain,
    interval_mask,
    permute_prediction_trials,
    predict_scalp_trials,
    refit_gains_for_lead_field,
)
from neural_observability.working_memory.generative_analysis import _geometry_score, _score_arrays


def _sources() -> tuple[list[BipolarSource], dict[str, np.ndarray]]:
    coordinates = {
        "A1": np.array([-1.0, 0.0, 0.0]),
        "A2": np.array([1.0, 0.0, 0.0]),
        "B1": np.array([20.0, -1.0, 0.0]),
        "B2": np.array([20.0, 1.0, 0.0]),
    }
    sources = [
        BipolarSource(
            name="A1-A2",
            lower_name="A1",
            upper_name="A2",
            lower_index=0,
            upper_index=1,
            midpoint_mni_mm=(0.0, 0.0, 0.0),
            direction_mni=(1.0, 0.0, 0.0),
        ),
        BipolarSource(
            name="B1-B2",
            lower_name="B1",
            upper_name="B2",
            lower_index=2,
            upper_index=3,
            midpoint_mni_mm=(20.0, 0.0, 0.0),
            direction_mni=(0.0, 1.0, 0.0),
        ),
    ]
    return sources, coordinates


def test_analytic_bipolar_operator_is_signed_finite_and_normalized() -> None:
    sources, coordinates = _sources()
    raw, normalized = analytic_bipolar_operator(sources, coordinates, softening_mm=0.5)
    assert raw.shape == normalized.shape == (2, 2)
    assert np.isfinite(raw).all()
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert np.all(np.diag(raw) < 0)


def test_average_reference_basis_removes_common_mode() -> None:
    basis = average_reference_basis(5)
    assert basis.shape == (5, 4)
    assert np.allclose(basis.T @ basis, np.eye(4))
    assert np.allclose(basis.T @ np.ones(5), 0.0, atol=1e-12)


def test_causal_filter_does_not_move_late_signal_before_guard_interval() -> None:
    frequency = 1000.0
    target = 200.0
    baseline = np.zeros((1, 1, 4000))
    perturbed = baseline.copy()
    perturbed[:, :, 2500:] = 1.0
    filtered_baseline = causal_filter_resample_trials(
        baseline, frequency, target, (0.5, 40.0)
    )
    filtered_perturbed = causal_filter_resample_trials(
        perturbed, frequency, target, (0.5, 40.0)
    )
    # A perturbation at 2.5 s cannot affect samples through 2.0 s.
    assert np.array_equal(filtered_baseline[:, :, :400], filtered_perturbed[:, :, :400])


def test_joint_model_recovers_positive_held_out_likelihood_and_controls() -> None:
    rng = np.random.default_rng(41)
    n_trials, n_sources, n_scalp, n_samples = 24, 3, 4, 120
    times = np.linspace(-1.0, 2.0, n_samples)
    encoding_interval = (-1.0, 0.0)
    scoring_interval = (0.0, 2.0)
    h = np.array(
        [[1.0, 0.2, -0.1], [0.1, 1.0, 0.15], [-0.1, 0.2, 1.0]],
        dtype=float,
    )
    transition = np.array(
        [[0.86, 0.08, 0.0], [0.0, 0.78, 0.09], [0.04, 0.0, 0.82]],
        dtype=float,
    )
    q = np.zeros((n_trials, n_sources, n_samples))
    q[:, :, 0] = rng.normal(size=(n_trials, n_sources))
    for sample in range(1, n_samples):
        drive = rng.normal(scale=0.25, size=(n_trials, n_sources))
        q[:, :, sample] = q[:, :, sample - 1] @ transition.T + drive
    x = np.einsum("cs,nst->nct", h, q) + rng.normal(
        scale=0.08, size=(n_trials, n_sources, n_samples)
    )
    lead = rng.normal(size=(n_scalp, n_sources))
    true_gains = np.array([1.2, -0.8, 0.7])
    y = np.einsum("ds,s,nst->ndt", lead, true_gains, q) + rng.normal(
        scale=0.3, size=(n_trials, n_scalp, n_samples)
    )

    latent_fit, q_filtered = fit_latent_dipole_model(
        x,
        times,
        encoding_interval,
        h,
        inverse_ridge=0.02,
        transition_ridge=0.02,
    )
    assert latent_fit.kalman_converged
    assert latent_fit.transition_spectral_radius < 1.0
    assert np.linalg.eigvalsh(latent_fit.process_covariance).min() > 0
    scalp_fit = fit_scalp_observation_model(
        q_filtered,
        y,
        times,
        encoding_interval,
        lead,
        gain_ridge=0.02,
    )
    prediction = predict_scalp_trials(q_filtered, lead, scalp_fit)
    actual = held_out_log_likelihood_gain(
        y, prediction, times, scoring_interval, scalp_fit
    )
    fast_actual, _, precision_y, _ = _score_arrays(
        y, prediction, times, scoring_interval, scalp_fit
    )
    response = interval_mask(times, scoring_interval, closed_right=True)
    centered_q = q_filtered - scalp_fit.source_mean[None, :, None]
    q_response = centered_q[:, :, response].transpose(1, 0, 2).reshape(n_sources, -1)
    q_covariance = q_response @ q_response.T / q_response.shape[1]
    q_precision_y = q_response @ precision_y.T / q_response.shape[1]
    geometry_formula_actual = _geometry_score(
        lead,
        scalp_fit.gains,
        q_covariance,
        q_precision_y,
        scalp_fit.nuisance_covariance,
    )
    shifted = circularly_shift_prediction(
        prediction,
        times,
        scoring_interval,
        offset_samples=17,
    )
    shifted_score = held_out_log_likelihood_gain(
        y, shifted, times, scoring_interval, scalp_fit
    )
    control_lead = lead[:, [2, 0, 1]]
    control_gains = refit_gains_for_lead_field(
        scalp_fit, control_lead, gain_ridge=0.02
    )
    geometry_prediction = predict_scalp_trials(
        q_filtered, control_lead, scalp_fit, gains=control_gains
    )
    geometry_score = held_out_log_likelihood_gain(
        y, geometry_prediction, times, scoring_interval, scalp_fit
    )
    assert actual > 0
    assert np.isclose(actual, fast_actual, atol=1e-12)
    assert np.isclose(actual, geometry_formula_actual, atol=1e-12)
    assert actual > shifted_score
    assert actual > geometry_score

    _, scaled_q_filtered = fit_latent_dipole_model(
        11.3 * x,
        times,
        encoding_interval,
        h,
        inverse_ridge=0.02,
        transition_ridge=0.02,
    )
    assert np.allclose(q_filtered, scaled_q_filtered, rtol=0.0, atol=2e-12)


def test_masks_and_trial_permutations_are_exact() -> None:
    times = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert interval_mask(times, (-1.0, 0.0)).tolist() == [True, True, False, False, False]
    prediction = np.arange(3 * 2 * 5, dtype=float).reshape(3, 2, 5)
    permuted = permute_prediction_trials(prediction, np.array([2, 0, 1]))
    assert np.array_equal(permuted[0], prediction[2])
    shifted = circularly_shift_prediction(prediction, times, (0.0, 1.0), 1)
    assert np.array_equal(shifted[:, :, :2], prediction[:, :, :2])
