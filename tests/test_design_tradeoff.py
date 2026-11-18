"""Unit tests for the finite-SNR network acquisition-vs-calibration analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from neural_observability.theory import visibility as V
from neural_observability.theory.atlas import _normalised_column
from neural_observability.theory.families import JansenRit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "simulations"))

from run_design_tradeoff import (  # noqa: E402
    full_network_mismatched_error,
    monte_carlo_single_column_check,
    sample_jr_perturbed_columns,
    separated_mismatched_network,
)


def test_separated_mismatched_matches_full_network_with_nullspace():
    """Theorem 1 coordinate rotation also separates a homogeneously mismatched
    uncoupled network exactly, including spatially blind modes (rank(L) < N)."""
    jr = JansenRit()
    x_star = jr.equilibria(90.0)[0]
    _, a0, b0, h0 = _normalised_column(jr, x_star)
    rng = np.random.default_rng(42)
    (at, bt, ht) = sample_jr_perturbed_columns(jr, 90.0, x_star, 0.10, 1, rng)[0][0]

    n_src, m_ch = 8, 5  # rank(L) <= 5 < 8 -> non-trivial spatial null space
    lead = rng.standard_normal((m_ch, n_src))
    g_s = rng.standard_normal((n_src, n_src))
    s_cov = 0.2 * np.eye(n_src) + g_s @ g_s.T / n_src
    g_r = rng.standard_normal((m_ch, m_ch))
    r_mat = 0.05 * (0.3 * np.eye(m_ch) + g_r @ g_r.T / m_ch)

    sep = separated_mismatched_network(at, bt, ht, a0, b0, h0, lead, r_mat, s_cov)
    a_tr, b_tr, c_tr = V.network_matrices(at, bt, ht, lead, s_cov)
    a_nm, b_nm, c_nm = V.network_matrices(a0, b0, h0, lead, s_cov)
    full = full_network_mismatched_error(
        a_tr, b_tr, c_tr, [ht] * n_src,
        a_nm, b_nm, c_nm, [h0] * n_src,
        r_mat,
    )
    assert np.max(np.abs(sep["state_nmse"] - full["state_nmse"])) < 5e-9
    assert np.max(np.abs(sep["output_nmse"] - full["output_nmse"])) < 5e-9
    assert abs(sep["sensor_signal_nmse"] - full["sensor_signal_nmse"]) < 5e-9


def test_rank_deficient_high_snr_obeys_visibility_weighted_floor_and_zero_sensor_error():
    """When rank(L) < N, infinite-SNR parcel error equals (1 - a_n) + a_n * f_j,
    and sensor-space EEG signal error vanishes while parcel pyramidal-output error stays 1 - a_n."""
    jr = JansenRit()
    x_star = jr.equilibria(90.0)[0]
    _, a0, b0, h0 = _normalised_column(jr, x_star)
    rng = np.random.default_rng(101)
    (at, bt, ht) = sample_jr_perturbed_columns(jr, 90.0, x_star, 0.10, 1, rng)[0][0]

    n_src, m_ch = 8, 3  # 5 null-space modes
    lead = rng.standard_normal((m_ch, n_src))
    s_cov = np.eye(n_src)
    r_mat = 1e-10 * np.eye(m_ch)

    sep = separated_mismatched_network(at, bt, ht, a0, b0, h0, lead, r_mat, s_cov)
    a_n = sep["visible_fraction"]
    floor_x3 = V.high_snr_floor(at, bt, a0, b0, ht, h_nom=h0)[2]
    expected_x3 = (1.0 - a_n) + a_n * floor_x3
    assert np.max(np.abs(sep["state_nmse"][:, 2] - expected_x3)) < 1e-3
    assert np.max(np.abs(sep["output_nmse"] - (1.0 - a_n))) < 1e-4
    assert sep["sensor_signal_nmse"] < 1e-4


def test_separated_mismatched_reduces_to_exact_when_unperturbed():
    jr = JansenRit()
    x_star = jr.equilibria(90.0)[0]
    _, a0, b0, h0 = _normalised_column(jr, x_star)
    rng = np.random.default_rng(7)
    lead = rng.standard_normal((6, 10))
    s_cov = np.eye(10)
    r_mat = 0.02 * np.eye(6)

    sep_mm = separated_mismatched_network(a0, b0, h0, a0, b0, h0, lead, r_mat, s_cov)
    sep_ex = V.separate_network(a0, b0, h0, lead, r_mat, s_cov)
    assert np.max(np.abs(sep_mm["state_r2"] - sep_ex.source_r2())) < 1e-11


def test_full_rank_high_snr_approaches_floor_for_x3_and_zero_for_output():
    jr = JansenRit()
    x_star = jr.equilibria(90.0)[0]
    _, a0, b0, h0 = _normalised_column(jr, x_star)
    rng = np.random.default_rng(11)
    (at, bt, ht) = sample_jr_perturbed_columns(jr, 90.0, x_star, 0.10, 1, rng)[0][0]

    n_src, m_ch = 5, 8  # full source rank
    lead = rng.standard_normal((m_ch, n_src))
    s_cov = np.eye(n_src)
    r_mat = 1e-10 * np.eye(m_ch)

    sep = separated_mismatched_network(at, bt, ht, a0, b0, h0, lead, r_mat, s_cov)
    floor_x3 = V.high_snr_floor(at, bt, a0, b0, ht, h_nom=h0)[2]
    assert np.max(np.abs(sep["state_nmse"][:, 2] - floor_x3)) / floor_x3 < 2e-3
    assert np.max(sep["output_nmse"]) < 1e-4


def test_monte_carlo_agrees_with_lyapunov_at_finite_snr():
    jr = JansenRit()
    x_star = jr.equilibria(90.0)[0]
    _, a0, b0, h0 = _normalised_column(jr, x_star)
    rng = np.random.default_rng(19)
    (at, bt, ht) = sample_jr_perturbed_columns(jr, 90.0, x_star, 0.10, 1, rng)[0][0]
    r_10db = 1.0 / (200.0 * 10.0)
    chk = monte_carlo_single_column_check(at, bt, a0, b0, ht, r_10db, seconds=250.0, seed=123)
    assert chk["rel_diff_x3"] < 0.06
    assert abs(chk["mc_output_nmse"] - chk["exact_output_nmse"]) < 0.015
