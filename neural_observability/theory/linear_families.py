"""Minimal linear parameterisations of linearised neural-mass columns.

The stationary second-order statistics of a linearised column depend on the
physiological parameters only through a few rate constants and loop gains.
Identifiability and the Whittle Fisher information are analysed in these
linear parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_observability.theory.families import JansenRit, Wendling
from neural_observability.theory.visibility import _frequency_response, stationary_covariance


@dataclass(frozen=True)
class LinearColumn:
    name: str
    labels: tuple[str, ...]
    log_theta: np.ndarray
    state_labels: tuple[str, ...]
    psp_states: tuple[int, ...]

    def matrices(self, log_theta: np.ndarray | None = None):
        raise NotImplementedError


@dataclass(frozen=True)
class JansenRitLinear(LinearColumn):
    @classmethod
    def from_equilibrium(cls, model: JansenRit, x_star: np.ndarray) -> "JansenRitLinear":
        jac = model.jacobian(x_star)
        theta = np.log([model.a, model.b, jac[3, 1], jac[4, 0], jac[5, 0], model.gain_e * model.a])
        return cls(
            name="Jansen-Rit",
            labels=("a", "b", "k_p", "k_e", "k_i", "sigma"),
            log_theta=theta,
            state_labels=model.state_labels,
            psp_states=(0, 1, 2),
        )

    def matrices(self, log_theta=None):
        a, b, kp, ke, ki, sigma = np.exp(self.log_theta if log_theta is None else log_theta)
        jac = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [-a * a, kp, -kp, -2 * a, 0, 0],
                [ke, -a * a, 0, 0, -2 * a, 0],
                [ki, 0, -b * b, 0, 0, -2 * b],
            ],
            dtype=float,
        )
        drive = np.zeros(6)
        drive[4] = sigma
        readout = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
        return jac, drive, readout


@dataclass(frozen=True)
class WendlingLinear(LinearColumn):
    @classmethod
    def from_equilibrium(cls, model: Wendling, x_star: np.ndarray) -> "WendlingLinear":
        jac = model.jacobian(x_star)
        theta = np.log(
            [
                1 / model.tau_a,
                1 / model.tau_b,
                1 / model.tau_g,
                jac[5, 1],
                jac[6, 0],
                jac[7, 0],
                jac[8, 0],
                -jac[8, 4],
                jac[9, 0],
                model.gain_e / model.tau_a,
            ]
        )
        return cls(
            name="Wendling-Chauvel",
            labels=("alpha", "beta", "gamma", "g_p", "g_e", "g_s", "g_f", "g_fi", "g_i", "sigma"),
            log_theta=theta,
            state_labels=model.state_labels,
            psp_states=(0, 1, 2, 3, 4),
        )

    def matrices(self, log_theta=None):
        al, be, ga, gp, ge, gs, gf, gfi, gi, sigma = np.exp(
            self.log_theta if log_theta is None else log_theta
        )
        jac = np.zeros((10, 10))
        jac[:5, 5:] = np.eye(5)
        jac[5, 0], jac[5, 5] = -al * al, -2 * al
        jac[5, 1], jac[5, 2], jac[5, 3] = gp, -gp, -gp
        jac[6, 0], jac[6, 1], jac[6, 6] = ge, -al * al, -2 * al
        jac[7, 0], jac[7, 2], jac[7, 7] = gs, -be * be, -2 * be
        jac[8, 0], jac[8, 4], jac[8, 3], jac[8, 8] = gf, -gfi, -ga * ga, -2 * ga
        jac[9, 0], jac[9, 4], jac[9, 9] = gi, -be * be, -2 * be
        drive = np.zeros(10)
        drive[6] = sigma
        readout = np.zeros(10)
        readout[1], readout[2], readout[3] = 1.0, -1.0, -1.0
        return jac, drive, readout

    def minimal_matrices(self, log_theta=None):
        """Minimal eight-state realisation and the map back to the ten states.

        The slow inhibitory PSPs y2 and y4 obey the same second-order kinetics
        and are driven by the same pyramidal input, with gains g_s and g_i, so
        z = y2 - (g_s/g_i) y4 obeys z'' + 2 beta z' + beta^2 z = 0 and vanishes
        in the stationary regime. The minimal state is (y0, y1, y3, y4) and
        their derivatives; y2 and its derivative are reconstructed as
        (g_s/g_i) times y4 and its derivative.
        """
        a, b, h = self.matrices(log_theta)
        return reduce_wendling(a, b, h)


def reduce_wendling(a: np.ndarray, b: np.ndarray, h: np.ndarray):
    """Reduce a ten-state Wendling--Chauvel linearisation to its minimal part.

    Returns ``(a_min, b_min, h_min, embed)`` where ``x_full = embed @ x_min``
    on the reachable subspace. Works for the physiological Jacobian and for
    the linear parameterisation alike, because both share the structure.
    """
    kappa = a[7, 0] / a[9, 0]
    keep = [0, 1, 3, 4, 5, 6, 8, 9]
    embed = np.zeros((10, 8))
    for k, full in enumerate(keep):
        embed[full, k] = 1.0
    embed[2, keep.index(4)] = kappa
    embed[7, keep.index(9)] = kappa
    a_min = a[np.ix_(keep, range(10))] @ embed
    return a_min, b[keep], h @ embed, embed


@dataclass(frozen=True)
class NextGenerationLinear(LinearColumn):
    """Linearised exact mean-field E-I column.

    Parameters: E and I damping (-2 v/tau), the two off-diagonal rate-voltage
    couplings per population, the four synaptic efficacies, the two synaptic
    rates and the drive gain. Readout is the net synaptic current onto E,
    ``J_ee s_e - J_ei s_i``.
    """

    @classmethod
    def from_equilibrium(cls, model, x_star: np.ndarray) -> "NextGenerationLinear":
        jac = model.jacobian(x_star)
        theta = np.log(
            [
                -jac[0, 0], jac[0, 1], -jac[1, 0],
                -jac[2, 2], jac[2, 3], -jac[3, 2],
                model.j_ee, model.j_ei, model.j_ie, model.j_ii,
                1 / model.tau_se, 1 / model.tau_si,
                1 / model.tau_e,
            ]
        )
        return cls(
            name="Next-generation E-I",
            labels=("d_e", "u_e", "w_e", "d_i", "u_i", "w_i", "J_ee", "J_ei", "J_ie", "J_ii", "k_se", "k_si", "sigma"),
            log_theta=theta,
            state_labels=model.state_labels,
            psp_states=(0, 1, 2, 3, 4, 5),
        )

    def matrices(self, log_theta=None):
        de, ue, we, di, ui, wi, jee, jei, jie, jii, kse, ksi, sigma = np.exp(
            self.log_theta if log_theta is None else log_theta
        )
        jac = np.zeros((6, 6))
        jac[0, 0], jac[0, 1], jac[1, 0], jac[1, 1] = -de, ue, -we, -de
        jac[1, 4], jac[1, 5] = jee, -jei
        jac[2, 2], jac[2, 3], jac[3, 2], jac[3, 3] = -di, ui, -wi, -di
        jac[3, 4], jac[3, 5] = jie, -jii
        jac[4, 0], jac[4, 4] = kse, -kse
        jac[5, 2], jac[5, 5] = ksi, -ksi
        drive = np.zeros(6)
        drive[1] = sigma
        readout = np.zeros(6)
        readout[4], readout[5] = jee, -jei
        return jac, drive, readout


def linear_column(model, x_star: np.ndarray) -> LinearColumn:
    """Dispatch to the minimal linear parameterisation of a family instance."""
    name = getattr(model, "name", "")
    if name == "Jansen-Rit":
        return JansenRitLinear.from_equilibrium(model, x_star)
    if name == "Wendling-Chauvel":
        return WendlingLinear.from_equilibrium(model, x_star)
    if name == "Next-generation E-I":
        return NextGenerationLinear.from_equilibrium(model, x_star)
    raise ValueError(f"No linear parameterisation for {name!r}")


def verification_analysis(
    column: LinearColumn,
    snr_db: float,
    f_max_hz: float = 500.0,
    points: int = 6000,
    step: float = 1e-5,
    null_tolerance: float = 1e-9,
    leak_tolerance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Whittle Fisher information and second-order hidden-state error metrics.

    For log-parameters estimated from T seconds of data by Whittle maximum
    likelihood, the expected lost fraction of variance of state j is
    approximately ``tau_j / T`` with ``tau_j = trace(M_j F^{-1})``.
    """
    theta0 = np.asarray(column.log_theta, dtype=float)
    a0, b0, h0 = column.matrices(theta0)
    n = len(a0)
    p0 = stationary_covariance(a0, b0)
    out_var = float(h0 @ p0 @ h0)
    # sensor noise PSD giving the requested SNR in a 100 Hz band
    r = out_var / (200.0 * 10 ** (snr_db / 10.0))
    omegas = np.linspace(0.0, 2 * np.pi * f_max_hz, points)
    rows = np.vstack([np.eye(n), h0.reshape(1, n)])

    def responses(theta):
        a, b, h = column.matrices(theta)
        g = _frequency_response(a, b, rows, omegas)
        return g

    g0 = responses(theta0)
    spec0 = np.abs(g0[:, n]) ** 2 + r
    p = len(theta0)
    d_spec = np.empty((len(omegas), p))
    d_hid = np.empty((len(omegas), n, p), dtype=complex)
    for k in range(p):
        tp, tm = theta0.copy(), theta0.copy()
        tp[k] += step
        tm[k] -= step
        gp, gm = responses(tp), responses(tm)
        d_spec[:, k] = (np.abs(gp[:, n]) ** 2 - np.abs(gm[:, n]) ** 2) / (2 * step)
        d_hid[:, :, k] = (gp[:, :n] / gp[:, [n]] - gm[:, :n] / gm[:, [n]]) / (2 * step)
    fisher = np.trapezoid(
        (d_spec / spec0[:, None])[:, :, None] * (d_spec / spec0[:, None])[:, None, :], omegas, axis=0
    ) / (2 * np.pi)
    weight = np.abs(g0[:, n]) ** 2
    m_mats = np.empty((n, p, p))
    for j in range(n):
        dj = d_hid[:, j, :]
        integrand = np.real(dj.conj()[:, :, None] * dj[:, None, :]) * weight[:, None, None]
        m_mats[j] = np.trapezoid(integrand, omegas, axis=0) / np.pi / p0[j, j]
    eigvals, eigvecs = np.linalg.eigh(fisher)
    null_mask = eigvals < null_tolerance * eigvals.max()
    null_vectors = eigvecs[:, null_mask]
    range_vectors = eigvecs[:, ~null_mask]
    tau = np.empty(n)
    worst = np.empty(n)
    null_leak = np.zeros(n)
    for j in range(n):
        m = m_mats[j]
        scale = max(float(np.trace(m)), np.finfo(float).tiny)
        if null_vectors.shape[1]:
            null_leak[j] = float(np.max(np.diag(null_vectors.T @ m @ null_vectors))) / scale
        if null_leak[j] > leak_tolerance:
            tau[j] = np.inf
            worst[j] = np.inf
            continue
        f_r = range_vectors.T @ fisher @ range_vectors
        m_r = range_vectors.T @ m @ range_vectors
        tau[j] = float(np.trace(m_r @ np.linalg.inv(f_r)))
        whiten = np.linalg.cholesky(np.linalg.inv(f_r))
        worst[j] = float(np.linalg.eigvalsh(whiten.T @ m_r @ whiten)[-1])
    return {
        "fisher_per_second": fisher,
        "fisher_eigenvalues": eigvals,
        "null_directions": null_vectors,
        "null_leak": null_leak,
        "hidden_metric": m_mats,
        "verification_time_s": tau,
        "worst_direction_ratio_s": worst,
        "output_variance": out_var,
        "sensor_psd": r,
    }
