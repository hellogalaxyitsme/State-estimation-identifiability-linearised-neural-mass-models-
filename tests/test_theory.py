"""Tests of the visibility theory: each test checks one stated result."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from neural_observability.theory import visibility as V
from neural_observability.theory.families import JansenRit, NextGenerationEI, Wendling
from neural_observability.theory.linear_families import JansenRitLinear


def _jr(branch=0, drive=50.0):
    model = JansenRit()
    x = model.equilibria(drive)[branch]
    a, b, h = model.jacobian(x), model.input_vector(), model.output_vector()
    p = V.stationary_covariance(a, b)
    return a, b / np.sqrt(h @ p @ h), h


def _snr_to_r(snr_db):
    return 1.0 / (200.0 * 10 ** (snr_db / 10.0))


def test_square_root_hankel_matches_high_precision():
    a, b, h = _jr(branch=2)
    fast = V.hankel_singular_values(a, b, h)
    exact = V.hankel_singular_values_mp(a, b, h, digits=40)
    assert np.max(np.abs(fast - exact) / exact) < 1e-5


def test_stationary_smoother_matches_brute_force_conditioning():
    a, b, h = _jr()
    r = _snr_to_r(10.0)
    rec = V.stationary_recoverability(a, b, h, r)
    p = rec.prior
    dt, k = 2e-3, 600
    idx = np.arange(-k, k + 1)

    def cov_x0_xt(t):
        return p @ expm(a.T * t) if t >= 0 else expm(a * (-t)) @ p

    cross = np.column_stack([cov_x0_xt(i * dt) @ h for i in idx])
    lags = {}
    cyy = np.empty((len(idx), len(idx)))
    for i, ti in enumerate(idx):
        for j, tj in enumerate(idx):
            lag = tj - ti
            if lag not in lags:
                lags[lag] = h @ cov_x0_xt(lag * dt) @ h
            cyy[i, j] = lags[lag]
    cyy += np.eye(len(idx)) * r / dt
    post = p - cross @ np.linalg.solve(cyy, cross.T)
    brute = 1 - np.diag(post) / np.diag(p)
    assert np.max(np.abs(brute - rec.physical_r2("smoothed"))) < 0.01


def test_gramian_bounds_dominate_exact_spectra():
    a, b, h = _jr()
    r = _snr_to_r(10.0)
    rec = V.stationary_recoverability(a, b, h, r)
    bounds = V.gramian_recoverability_bound(a, b, h, r)
    assert np.all(rec.spectrum("future_only") <= bounds["future_only"] + 1e-9)


def test_separation_theorem_is_exact_for_identical_columns():
    rng = np.random.default_rng(0)
    a, b, h = _jr()
    lead = rng.standard_normal((5, 7))
    source_cov = np.cov(rng.standard_normal((7, 30)))
    sensor = np.cov(rng.standard_normal((5, 40))) * 0.05
    sep = V.separate_network(a, b, h, lead, sensor, source_cov)
    big_a, big_b, big_c = V.network_matrices(a, b, h, lead, source_cov)
    full = V.stationary_recoverability(big_a, big_b, big_c, sensor)
    assert np.max(np.abs(full.physical_r2("filtered").reshape(7, 6) - sep.source_r2())) < 1e-6


def test_exact_model_tracking_approaches_one():
    a, b, h = _jr()
    r2 = V.stationary_recoverability(a, b, h, _snr_to_r(60.0)).physical_r2("filtered")
    # every potential (non-derivative) state is tracked essentially perfectly
    assert np.all(r2[:3] > 0.999)


def test_jansen_rit_and_wendling_are_minimum_phase():
    for model, drive in ((JansenRit(), 90.0), (Wendling(), 90.0)):
        x = model.equilibria(drive)[0]
        zeros = V.transmission_zeros(model.jacobian(x), model.input_vector(), model.output_vector())
        assert zeros.real.max() < 0


def test_high_snr_floor_matches_mismatched_filter():
    model = JansenRit()
    x = model.equilibria(50.0)[0]
    col = JansenRitLinear.from_equilibrium(model, x)
    a, b, h = col.matrices()
    b = b / np.sqrt(h @ V.stationary_covariance(a, b) @ h)
    theta = col.log_theta + 0.1 * np.array([1, -1, 1, -1, 1, 0.0])
    at, bt, _ = col.matrices(theta)
    bt = bt / np.sqrt(h @ V.stationary_covariance(at, bt) @ h)
    floor = V.high_snr_floor(at, bt, a, b, h, omega_max=2 * np.pi * 300)
    sim = 1 - V.mismatched_filter_error(at, bt, a, b, h, _snr_to_r(80.0))["r2"]
    for state in (0, 2, 3, 5):  # potentials and their slaved derivatives
        assert abs(floor[state] - sim[state]) < 2e-3 * max(1.0, sim[state])


def test_output_is_recoverable_under_mismatch():
    model = JansenRit()
    x = model.equilibria(50.0)[0]
    col = JansenRitLinear.from_equilibrium(model, x)
    a, b, h = col.matrices()
    at, bt, _ = col.matrices(col.log_theta + 0.15)
    out = V.mismatched_filter_error(at, bt, a, b, h, _snr_to_r(60.0))["output_r2"]
    assert out > 0.999


def test_jansen_rit_loop_gain_symmetry_hides_x1_scale():
    model = JansenRit()
    x = model.equilibria(50.0)[0]
    col = JansenRitLinear.from_equilibrium(model, x)
    shift = np.array([0, 0, 0.2, -0.2, -0.2, 0])  # k_p up, k_e and k_i down
    rows_y = np.vstack([np.eye(6), col.matrices()[2]])
    w = 2 * np.pi * np.linspace(0.5, 80, 50)
    g0 = V._frequency_response(*col.matrices()[:2], rows_y, w)
    g1 = V._frequency_response(*col.matrices(col.log_theta + shift)[:2], rows_y, w)
    assert np.max(np.abs(g0[:, 6] - g1[:, 6])) < 1e-12 * np.max(np.abs(g0[:, 6]))
    assert np.max(np.abs(g1[:, 0] / g0[:, 0] - np.exp(0.2))) < 1e-9  # x1 rescaled
    assert np.max(np.abs(g1[:, 2] - g0[:, 2])) < 1e-12 * np.max(np.abs(g0[:, 2]))  # x3 unchanged


def test_zero_on_imaginary_axis_does_not_prevent_tracking():
    # G(s) = s / (s + 1)^2 has a zero at s = 0; the filtering error is
    # (sqrt(r + 4 r^2) - 2 r) I, which vanishes as r -> 0 (theorem 2)
    a = np.array([[0.0, 1.0], [-1.0, -2.0]])
    b = h = np.array([0.0, 1.0])
    for r in (1e-2, 1e-4, 1e-8):
        rec = V.stationary_recoverability(a, b, h, r)
        expected = (np.sqrt(r + 4 * r * r) - 2 * r) * np.eye(2)
        assert np.allclose(rec.filtered, expected, rtol=1e-6, atol=1e-14)
    assert np.max(rec.filtered) < 1e-3


def test_right_half_plane_zero_bounds_filtering_error():
    # G(s) = (1 - s) / (s + 1)^2: zero at z = 1. Any causal estimate of a state j
    # has error at least 2 Re(z) |G_j(z)|^2 (theorem 2, converse)
    a = np.array([[0.0, 1.0], [-1.0, -2.0]])
    b = np.array([0.0, 1.0])
    h = np.array([1.0, -1.0])
    z = 1.0
    gz = np.linalg.solve(z * np.eye(2) - a, b)
    for r in (1e-4, 1e-8, 1e-12):
        sigma = V.stationary_recoverability(a, b, h, r).filtered
        assert np.all(np.diag(sigma) >= 2 * z * gz**2 * (1 - 1e-6))


def test_wendling_reduction_is_exact():
    from neural_observability.theory.linear_families import WendlingLinear, reduce_wendling

    model = Wendling()
    x = model.equilibria(90.0)[0]
    a, b, h = model.jacobian(x), model.input_vector(), model.output_vector()
    am, bm, hm, embed = reduce_wendling(a, b, h)
    w = 2 * np.pi * np.geomspace(0.5, 300, 40)
    full = V._frequency_response(a, b, np.vstack([np.eye(10), h]), w)
    red = V._frequency_response(am, bm, np.vstack([embed, hm]), w)
    assert np.max(np.abs(full - red) / np.abs(full).max(0)) < 1e-9
    # the full realisation has a structurally unreachable direction, the reduced one does not
    p = V.stationary_covariance(a, b)
    d = np.sqrt(np.diag(p))
    assert np.linalg.eigvalsh(p / np.outer(d, d)).min() < 1e-12
    col = WendlingLinear.from_equilibrium(model, x)
    assert col.minimal_matrices()[0].shape == (8, 8)


@pytest.mark.parametrize("family", [JansenRit, Wendling, NextGenerationEI])
def test_named_symmetries_leave_the_output_unchanged(family):
    from neural_observability.theory.linear_families import linear_column
    from neural_observability.theory.symmetry import named_generators, response_jacobian

    model = family()
    drive = {"Jansen-Rit": 90.0, "Wendling-Chauvel": 90.0, "Next-generation E-I": -0.65}[model.name]
    col = linear_column(model, model.equilibria(drive)[0])
    jac = response_jacobian(col)
    _, gens = named_generators(col)
    for g in gens:
        assert np.linalg.norm(jac @ g) < 1e-7 * np.linalg.norm(jac, 2) * np.linalg.norm(g)


def test_wendling_slow_fast_exchange_is_an_exact_discrete_symmetry():
    # exchanging the slow and fast inhibitory rates and their pyramidal gains,
    # with g_fi * g_i fixed, leaves the output transfer function unchanged
    from neural_observability.theory.linear_families import WendlingLinear

    model = Wendling()
    col = WendlingLinear.from_equilibrium(model, model.equilibria(90.0)[0])
    th = col.log_theta.copy()
    lab = list(col.labels)
    sw = th.copy()
    sw[lab.index("beta")], sw[lab.index("gamma")] = th[lab.index("gamma")], th[lab.index("beta")]
    sw[lab.index("g_s")], sw[lab.index("g_f")] = th[lab.index("g_f")], th[lab.index("g_s")]
    sw[lab.index("g_i")] = th[lab.index("g_i")] + 0.7
    sw[lab.index("g_fi")] = th[lab.index("g_fi")] - 0.7
    w = 2 * np.pi * np.geomspace(0.2, 400, 60)
    g0 = V._frequency_response(*col.matrices(th)[:2], col.matrices(th)[2].reshape(1, -1), w)[:, 0]
    g1 = V._frequency_response(*col.matrices(sw)[:2], col.matrices(sw)[2].reshape(1, -1), w)[:, 0]
    assert np.max(np.abs(g1 - g0) / np.abs(g0)) < 1e-9


def test_kalman_estimates_transform_with_the_similarity():
    # theorem 4: spectrally equivalent parameters give Kalman filters whose
    # transfer functions from the record satisfy F' = S F at any SNR
    model = JansenRit()
    col = JansenRitLinear.from_equilibrium(model, model.equilibria(90.0)[0])
    c = 1.3
    shift = np.log(c) * np.array([0, 0, 1, -1, -1, 0])
    s = np.diag([c, 1, 1, c, 1, 1])
    r = _snr_to_r(10.0)
    w = 2 * np.pi * np.geomspace(0.5, 200, 30)

    def filter_response(theta):
        a, b, h = col.matrices(theta)
        sigma = V.stationary_recoverability(a, b, h, r).filtered
        k = sigma @ h / r
        closed = a - np.outer(k, h)
        return np.array([np.linalg.solve(1j * x * np.eye(6) - closed, k) for x in w])

    f0, f1 = filter_response(col.log_theta), filter_response(col.log_theta + shift)
    assert np.max(np.abs(f1 - f0 @ s.T)) < 1e-8 * np.max(np.abs(f0))


@pytest.mark.parametrize("family", [JansenRit, Wendling, NextGenerationEI])
def test_default_operating_points_are_stable(family):
    model = family()
    drive = {"Jansen-Rit": 90.0, "Wendling-Chauvel": 90.0, "Next-generation E-I": -0.65}[model.name]
    x = model.equilibria(drive)[0]
    assert np.linalg.eigvals(model.jacobian(x)).real.max() < 0
