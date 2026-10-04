"""Physically calibrated visibility of neural-mass states from EEG.

Core objects for a linear(ised) stochastic neural mass observed through a
lead field::

    dx = A x dt + B dw,          dy = C x dt + R^{1/2} dv.

* ``P``  stationary state covariance (controllability Gramian of the drive),
* ``W``  noise-weighted observability Gramian,
* visibility spectrum  = eigenvalues of ``P W`` = squared Hankel singular
  values of ``(A, B, R^{-1/2} C)``.

The module also provides the exact spatial x temporal factorisation for
networks of identical columns, residue (pole-weight) bounds on the Hankel
spectrum, steady-state Kalman filtering/smoothing recoverability, and the
Loewner-sandwich constants used for heterogeneous or coupled networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import (
    cholesky,
    eigh,
    solve_continuous_are,
    solve_continuous_lyapunov,
)


# ---------------------------------------------------------------------------
# Gramians and Hankel spectra
# ---------------------------------------------------------------------------


def _sym(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def check_hurwitz(a: np.ndarray) -> float:
    abscissa = float(np.max(np.linalg.eigvals(a).real))
    if abscissa >= 0:
        raise ValueError(f"A is not Hurwitz (spectral abscissa {abscissa:.3e})")
    return abscissa


def stationary_covariance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A P + P A^T + B B^T = 0``."""
    b = np.atleast_2d(np.asarray(b, dtype=float).T).T
    if b.ndim == 1:
        b = b[:, None]
    return _sym(solve_continuous_lyapunov(a, -b @ b.T))


def observability_gramian(a: np.ndarray, c: np.ndarray, r_inv: np.ndarray | float = 1.0) -> np.ndarray:
    """Solve ``A^T W + W A + C^T R^{-1} C = 0``."""
    c = np.atleast_2d(c)
    r_inv = np.atleast_2d(r_inv) if np.ndim(r_inv) else np.eye(c.shape[0]) * float(r_inv)
    return _sym(solve_continuous_lyapunov(a.T, -c.T @ r_inv @ c))


def _psd_factor(m: np.ndarray) -> np.ndarray:
    """Return F with ``m = F F^T`` (Cholesky when possible, else eigen-factor)."""
    try:
        return cholesky(_sym(m), lower=True)
    except np.linalg.LinAlgError:
        values, vectors = eigh(_sym(m))
        return vectors * np.sqrt(np.clip(values, 0.0, None))


def hankel_singular_values(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, r_inv: np.ndarray | float = 1.0
) -> np.ndarray:
    """Hankel singular values by the square-root method after diagonal scaling.

    The diagonal similarity ``x = D z`` does not change Hankel singular values
    but removes the large unit disparities between potentials and their
    derivatives, which keeps the square-root method accurate.
    """
    check_hurwitz(a)
    p = stationary_covariance(a, b)
    scale = np.sqrt(np.clip(np.diag(p), np.finfo(float).tiny, None))
    d_inv = np.diag(1.0 / scale)
    d = np.diag(scale)
    a_z = d_inv @ a @ d
    b_z = d_inv @ np.atleast_2d(np.asarray(b, dtype=float).reshape(len(a), -1))
    c_z = np.atleast_2d(c) @ d
    p_z = stationary_covariance(a_z, b_z)
    w_z = observability_gramian(a_z, c_z, r_inv)
    lp = _psd_factor(p_z)
    lw = _psd_factor(w_z)
    return np.linalg.svd(lw.T @ lp, compute_uv=False)


def backward_hankel_singular_values(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, r_inv: np.ndarray | float = 1.0
) -> np.ndarray:
    """Hankel singular values of the stationary backward (time-reversed) model.

    The backward model ``A_bar = P A^T P^{-1}`` is formed in unit-variance
    coordinates, where ``P`` is well conditioned.
    """
    p_raw = stationary_covariance(a, b)
    scale = np.sqrt(np.diag(p_raw))
    a_z = a * scale[None, :] / scale[:, None]
    b_z = np.asarray(b, dtype=float).reshape(len(a), -1) / scale[:, None]
    c_z = np.atleast_2d(c) * scale[None, :]
    p_z = stationary_covariance(a_z, b_z)
    a_back = p_z @ a_z.T @ np.linalg.inv(p_z)
    return hankel_singular_values(a_back, b_z, c_z, r_inv)


def visibility_spectrum(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, r_inv: np.ndarray | float = 1.0
) -> np.ndarray:
    """Eigenvalues of ``P W`` (dimensionless signal-to-noise per balanced mode)."""
    return hankel_singular_values(a, b, c, r_inv) ** 2


def hankel_singular_values_mp(a, b, c, digits: int = 60) -> np.ndarray:
    """High-precision Hankel singular values for small single-column systems."""
    import mpmath as mp

    mp.mp.dps = digits
    n = len(a)
    am = mp.matrix(np.asarray(a).tolist())
    bm = mp.matrix(np.asarray(b, dtype=float).reshape(n, 1).tolist())
    cm = mp.matrix(np.asarray(c, dtype=float).reshape(1, n).tolist())
    eye = mp.eye(n)

    def kron(x, y):
        out = mp.zeros(x.rows * y.rows, x.cols * y.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                for k in range(y.rows):
                    for l in range(y.cols):
                        out[i * y.rows + k, j * y.cols + l] = x[i, j] * y[k, l]
        return out

    def lyap(m, q):
        # m X + X m^T = -q, vec(X) = -(I kron m + m kron I)^{-1} vec(q)
        big = kron(eye, m) + kron(m, eye)
        vec_q = mp.matrix([q[i % n, i // n] for i in range(n * n)])
        sol = mp.lu_solve(big, -vec_q)
        out = mp.zeros(n, n)
        for i in range(n * n):
            out[i % n, i // n] = sol[i]
        return out

    p = lyap(am, bm * bm.T)
    w = lyap(am.T, cm.T * cm)
    eigenvalues = mp.eig(p * w, left=False, right=False)
    values = sorted((mp.re(v) for v in eigenvalues), reverse=True)
    return np.array([float(mp.sqrt(max(v, 0))) for v in values])


# ---------------------------------------------------------------------------
# Poles, residues and the pole-weight bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalData:
    poles: np.ndarray
    residues: np.ndarray
    weights: np.ndarray  # |residue| / (2 |Re pole|) = Hankel norm of each term


def modal_data(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> ModalData:
    """Partial-fraction data of ``G(s) = c (sI - A)^{-1} b`` (simple poles)."""
    values, right = np.linalg.eig(a)
    left = np.linalg.inv(right)
    b = np.asarray(b, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)
    residues = (c @ right) * (left @ b)
    weights = np.abs(residues) / (2.0 * np.abs(values.real))
    order = np.argsort(weights)[::-1]
    return ModalData(values[order], residues[order], weights[order])


def pole_weight_upper_bounds(weights: np.ndarray) -> np.ndarray:
    """Upper bounds ``sigma_{k+1} <= sum_{i>k} w_(i)`` for k = 0..n-1."""
    w = np.sort(np.asarray(weights))[::-1]
    tail = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
    return tail[: len(w)]


def core_lower_bound(a, b, c, core_size: int) -> float:
    """Lower bound on sigma_1 from the strongest ``core_size`` modal terms.

    ``sigma_1(G) >= sigma_1(G_core) - sum_{i not in core} w_i`` (Weyl), where
    the core keeps the poles with the largest weights. Returns the bound.
    """
    data = modal_data(a, b, c)
    keep = np.arange(core_size)
    # Hankel matrix of the (complex) core subsystem in modal coordinates
    lam = data.poles[keep]
    res = data.residues[keep]
    # modal realisation: A = diag(lam), b_i = 1, c_i = res_i
    p = -np.ones((core_size, core_size)) / (lam[:, None] + lam[None, :].conj())
    w = -np.outer(res.conj(), res) / (lam[:, None].conj() + lam[None, :])
    sv = np.sqrt(np.abs(np.linalg.eigvals(p @ w)))
    sigma_core = float(np.max(sv))
    return sigma_core - float(np.sum(data.weights[core_size:]))


# ---------------------------------------------------------------------------
# Networks: exact spatial x temporal factorisation
# ---------------------------------------------------------------------------


def spatial_spectrum(lead: np.ndarray, sensor_cov: np.ndarray, source_cov: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``S^{1/2} L^T R^{-1} L S^{1/2}`` (spatial SNR spectrum)."""
    r_factor = cholesky(_sym(sensor_cov), lower=True)
    whitened = np.linalg.solve(r_factor, lead)
    s_factor = _psd_factor(source_cov)
    m = whitened @ s_factor
    sv = np.linalg.svd(m, compute_uv=False)
    out = np.zeros(source_cov.shape[0])
    out[: len(sv)] = sv**2
    return np.sort(out)[::-1]


def factorized_visibility(spatial: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Outer-product visibility spectrum, sorted descending."""
    return np.sort(np.outer(spatial, local).ravel())[::-1]


def network_matrices(
    local_a: np.ndarray,
    local_b: np.ndarray,
    local_h: np.ndarray,
    lead: np.ndarray,
    source_cov: np.ndarray,
    coupling: np.ndarray | None = None,
    coupling_row: int | None = None,
    coupling_col: int | None = None,
    local_a_list: list[np.ndarray] | None = None,
):
    """Assemble ``(A, B, C)`` for a network of columns.

    ``coupling`` (N x N) adds ``coupling[i, j]`` from state ``coupling_col`` of
    column j into state ``coupling_row`` of column i. ``local_a_list`` allows
    heterogeneous columns.
    """
    n_src = lead.shape[1]
    q = local_a.shape[0]
    if local_a_list is None:
        a = np.kron(np.eye(n_src), local_a)
    else:
        a = np.zeros((n_src * q, n_src * q))
        for i, block in enumerate(local_a_list):
            a[i * q : (i + 1) * q, i * q : (i + 1) * q] = block
    if coupling is not None:
        e = np.zeros((q, q))
        e[coupling_row, coupling_col] = 1.0
        a = a + np.kron(coupling, e)
    b = np.kron(_psd_factor(source_cov), np.asarray(local_b, dtype=float).reshape(q, 1))
    c = np.kron(lead, np.asarray(local_h, dtype=float).reshape(1, q))
    return a, b, c


def loewner_sandwich(reference: np.ndarray, perturbed: np.ndarray) -> tuple[float, float]:
    """Return (lo, hi) with ``lo * ref <= pert <= hi * ref`` in the Loewner order."""
    factor = _psd_factor(reference)
    inv = np.linalg.pinv(factor)
    values = np.linalg.eigvalsh(_sym(inv @ perturbed @ inv.T))
    return float(values.min()), float(values.max())


# ---------------------------------------------------------------------------
# Stationary Kalman filtering and smoothing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recoverability:
    prior: np.ndarray
    filtered: np.ndarray
    future_only: np.ndarray
    smoothed: np.ndarray

    def physical_r2(self, which: str = "smoothed") -> np.ndarray:
        post = getattr(self, which)
        return 1.0 - np.diag(post) / np.diag(self.prior)

    def spectrum(self, which: str = "smoothed") -> np.ndarray:
        """Eigenvalues of ``I - P^{-1/2} Sigma P^{-1/2}`` (descending)."""
        post = getattr(self, which)
        factor = _psd_factor(self.prior)
        inv = np.linalg.inv(factor)
        values = np.linalg.eigvalsh(_sym(inv @ post @ inv.T))
        return np.sort(1.0 - values)[::-1]


def stationary_recoverability(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, r: np.ndarray | float
) -> Recoverability:
    """Steady-state filter, future-only and smoother error covariances.

    The forward Kalman filter gives ``Sigma_f``. Information carried by future
    observations is the stabilising solution ``Lambda`` of the backward
    information-filter Riccati equation
    ``A^T L + L A - L B B^T L + C^T R^{-1} C = 0`` (Fraser--Potter), so that
    ``Sigma_future = (I + P L)^{-1} P`` and ``Sigma_smooth = (I + Sigma_f L)^{-1} Sigma_f``.
    No inverse of the (often numerically singular) stationary covariance is
    required.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(len(a), -1)
    c = np.atleast_2d(c)
    r = np.atleast_2d(r) if np.ndim(r) else np.eye(c.shape[0]) * float(r)
    # Work in coordinates with unit prior variances; every reported quantity is
    # transformed back exactly, so this only improves conditioning.
    p_raw = stationary_covariance(a, b)
    scale = np.sqrt(np.diag(p_raw))
    a_z = a * scale[None, :] / scale[:, None]
    b_z = b / scale[:, None]
    c_z = c * scale[None, :]
    n = len(a)
    q = b_z @ b_z.T
    p = stationary_covariance(a_z, b_z)
    sigma_f = _sym(solve_continuous_are(a_z.T, c_z.T, q, r))
    lam = _sym(solve_continuous_are(a_z, b_z, c_z.T @ np.linalg.solve(r, c_z), np.eye(b_z.shape[1])))
    sigma_b = _sym(np.linalg.solve(np.eye(n) + p @ lam, p))
    sigma_s = _sym(np.linalg.solve(np.eye(n) + sigma_f @ lam, sigma_f))
    back = np.outer(scale, scale)
    return Recoverability(p * back, sigma_f * back, sigma_b * back, sigma_s * back)


def _spd_inverse(m: np.ndarray) -> np.ndarray:
    values, vectors = eigh(_sym(m))
    floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    values = np.clip(values, floor, None)
    return _sym((vectors / values) @ vectors.T)


def gramian_recoverability_bound(a, b, c, r) -> dict[str, np.ndarray]:
    """Upper bounds on recoverable fractions from the deterministic Gramians.

    future-only:  Sigma_b >= (P^{-1} + W)^{-1}
    filtering:    Sigma_f >= (P^{-1} + W_bar)^{-1}  (backward-model Gramian)
    smoothing:    Sigma_s >= (P^{-1} + W + W_bar)^{-1}
    Returned arrays are the eigenvalues of the bounding recoverable fractions.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(len(a), -1)
    c = np.atleast_2d(c)
    r = np.atleast_2d(r) if np.ndim(r) else np.eye(c.shape[0]) * float(r)
    r_inv = np.linalg.inv(r)
    p = stationary_covariance(a, b)
    w = observability_gramian(a, c, r_inv)
    a_back = p @ a.T @ np.linalg.inv(p)
    w_back = observability_gramian(a_back, c, r_inv)
    factor = _psd_factor(p)

    def fractions(info):
        values = np.linalg.eigvalsh(_sym(factor.T @ info @ factor))
        values = np.clip(values, 0.0, None)
        return np.sort(values / (1.0 + values))[::-1]

    return {
        "future_only": fractions(w),
        "filtering": fractions(w_back),
        "smoothing": fractions(w + w_back),
    }


# ---------------------------------------------------------------------------
# Information budgets
# ---------------------------------------------------------------------------


def gaussian_information(visibility: np.ndarray) -> float:
    """``0.5 * sum log(1 + lambda)`` in nats."""
    return float(0.5 * np.sum(np.log1p(np.clip(visibility, 0.0, None))))


def hidden_information(spatial: np.ndarray, local: np.ndarray, kept_local_modes: int) -> dict[str, float]:
    """Exact information in local modes beyond ``kept_local_modes`` and its bound."""
    local = np.sort(np.asarray(local))[::-1]
    tail = local[kept_local_modes:]
    exact = gaussian_information(np.outer(spatial, tail).ravel())
    bound = 0.5 * float(np.sum(spatial)) * float(np.sum(tail))
    total = gaussian_information(np.outer(spatial, local).ravel())
    return {"hidden_nats": exact, "hidden_bound_nats": bound, "total_nats": total}


def snr_thresholds_db(local: np.ndarray) -> np.ndarray:
    """Spatial SNR (dB) at which local mode k reaches visibility 1, relative to mode 1."""
    local = np.sort(np.asarray(local))[::-1]
    return 10.0 * np.log10(local[0] / np.clip(local, np.finfo(float).tiny, None))


def all_subsets_bound(weights: np.ndarray, k: int) -> float:
    """Brute-force check of the pole-weight bound (keep any k terms)."""
    w = np.asarray(weights)
    best = np.inf
    for keep in combinations(range(len(w)), k):
        mask = np.ones(len(w), dtype=bool)
        mask[list(keep)] = False
        best = min(best, float(np.sum(w[mask])))
    return best


# ---------------------------------------------------------------------------
# Zeros, output recoverability and model mismatch
# ---------------------------------------------------------------------------


def transmission_zeros(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Finite invariant zeros of the single-input single-output system."""
    from scipy.linalg import eig as _eig

    n = len(a)
    m = np.block([[a, np.reshape(b, (n, 1))], [np.reshape(c, (1, n)), np.zeros((1, 1))]])
    e = np.block([[np.eye(n), np.zeros((n, 1))], [np.zeros((1, n)), np.zeros((1, 1))]])
    values = _eig(m, e, right=False)
    return values[np.isfinite(values)]


def output_r2(rec: Recoverability, h: np.ndarray, which: str = "filtered") -> float:
    post = getattr(rec, which)
    h = np.asarray(h, dtype=float).reshape(-1)
    return float(1.0 - (h @ post @ h) / (h @ rec.prior @ h))


def mismatched_filter_error(
    a_true: np.ndarray,
    b_true: np.ndarray,
    a_nom: np.ndarray,
    b_nom: np.ndarray,
    h: np.ndarray,
    r: float,
    h_nom: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Error of the nominal steady-state Kalman filter run on the true system.

    Returns the true prior covariance, the error covariance, per-state R^2 and
    the output R^2. Both models share state coordinates; ``h`` is the true
    readout and ``h_nom`` (default ``h``) the readout assumed by the filter.
    """
    n = len(a_true)
    h = np.asarray(h, dtype=float).reshape(1, n)
    h_n = h if h_nom is None else np.asarray(h_nom, dtype=float).reshape(1, n)
    p_true = stationary_covariance(a_true, b_true)
    scale = np.sqrt(np.diag(p_true))
    s_inv = 1.0 / scale

    def to_z(a):
        return a * scale[None, :] * s_inv[:, None]

    at, an = to_z(a_true), to_z(a_nom)
    bt = np.asarray(b_true, dtype=float).reshape(n, 1) * s_inv[:, None]
    bn = np.asarray(b_nom, dtype=float).reshape(n, 1) * s_inv[:, None]
    hz = h * scale[None, :]
    hz_n = h_n * scale[None, :]
    sigma = solve_continuous_are(an.T, hz_n.T, bn @ bn.T, np.array([[r]]))
    gain = sigma @ hz_n.T / r
    big_a = np.block([[at, np.zeros((n, n))], [gain @ hz, an - gain @ hz_n]])
    big_b = np.block([[bt, np.zeros((n, 1))], [np.zeros((n, 1)), gain * np.sqrt(r)]])
    cov = _sym(solve_continuous_lyapunov(big_a, -big_b @ big_b.T))
    pxx, pxh, phh = cov[:n, :n], cov[:n, n:], cov[n:, n:]
    err = _sym(pxx - pxh - pxh.T + phh)
    back = np.outer(scale, scale)
    err_phys = err * back
    prior = _sym(pxx) * back
    ht, hn = hz.reshape(-1), hz_n.reshape(-1)
    out_err = ht @ pxx @ ht - 2 * ht @ pxh @ hn + hn @ phh @ hn
    return {
        "prior": prior,
        "error": err_phys,
        "r2": 1.0 - np.diag(err_phys) / np.diag(prior),
        "output_r2": float(1.0 - out_err / (ht @ pxx @ ht)),
    }


def _frequency_response(a, b, c_rows, omegas):
    """Return array (len(omegas), rows) of c (i w I - A)^{-1} b."""
    n = len(a)
    b = np.asarray(b, dtype=float).reshape(n)
    omegas = np.asarray(omegas, dtype=float)
    stack = 1j * omegas[:, None, None] * np.eye(n)[None, :, :] - np.asarray(a)[None, :, :]
    rhs = np.broadcast_to(b.astype(complex), (len(omegas), n))[..., None]
    solved = np.linalg.solve(stack, rhs)[..., 0]
    return solved @ np.asarray(c_rows).T


def high_snr_floor(
    a_true,
    b_true,
    a_nom,
    b_nom,
    h,
    omega_max: float = 2 * np.pi * 2000.0,
    points: int = 8000,
    h_nom=None,
) -> np.ndarray:
    """Closed-form SNR-independent floor of hidden-state error under mismatch.

    floor_j = (1/2pi) int |H_j^true - H_j^nom|^2 |G_y^true|^2 dw / Var(x_j)
    with H_j = G_j / G_y. Returned as the lost fraction of variance (1 - R^2).
    ``h`` is the true readout; ``h_nom`` (default ``h``) the assumed one.
    """
    n = len(a_true)
    h = np.asarray(h, dtype=float).reshape(1, n)
    h_nom = h if h_nom is None else np.asarray(h_nom, dtype=float).reshape(1, n)
    omegas = np.concatenate([[0.0], np.geomspace(1e-2, omega_max, points)])
    gt = _frequency_response(a_true, b_true, np.vstack([np.eye(n), h]), omegas)
    gn = _frequency_response(a_nom, b_nom, np.vstack([np.eye(n), h_nom]), omegas)
    ht = gt[:, :n] / gt[:, [n]]
    hn = gn[:, :n] / gn[:, [n]]
    integrand = np.abs(ht - hn) ** 2 * np.abs(gt[:, [n]]) ** 2
    # (1/2pi) * 2 * int_0^inf  (integrand is even in w)
    variance = np.trapezoid(integrand, omegas, axis=0) / np.pi
    p_true = stationary_covariance(a_true, b_true)
    return variance / np.diag(p_true)


def whittle_fisher(spectra_grad: np.ndarray, spectrum: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    """Fisher information per second of the Whittle likelihood.

    ``spectra_grad``: (len(omegas), p) derivatives of the output spectrum;
    ``spectrum``: (len(omegas),) output spectrum (including sensor noise).
    F = (1/4pi) int_{-inf}^{inf} (dS/S)(dS/S)^T dw = (1/2pi) int_0^inf ...
    """
    rel = spectra_grad / spectrum[:, None]
    integrand = rel[:, :, None] * rel[:, None, :]
    return np.trapezoid(integrand, omegas, axis=0) / (2 * np.pi)


# ---------------------------------------------------------------------------
# Spatial--temporal separation for networks of identical columns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeparatedNetwork:
    """Exact decomposition of a network estimation problem into scalar ones."""

    spatial_snr: np.ndarray       # mu_i, length N (zeros for spatially blind modes)
    mixing: np.ndarray            # c = S^{1/2} V, N x N
    local_prior: np.ndarray       # P0
    local_posteriors: np.ndarray  # Sigma(mu_i), N x q x q

    def source_posterior(self, source: int) -> np.ndarray:
        w = self.mixing[source] ** 2
        return np.tensordot(w, self.local_posteriors, axes=1)

    def source_r2(self) -> np.ndarray:
        """R^2 of every local state at every source, shape (N, q)."""
        weights = self.mixing**2
        post = np.einsum("ni,ijj->nj", weights, self.local_posteriors)
        prior = weights.sum(1)[:, None] * np.diag(self.local_prior)[None, :]
        return 1.0 - post / prior

    def ceiling_r2(self) -> np.ndarray:
        """Infinite-SNR limit of output recoverability per source (spatial ceiling)."""
        weights = self.mixing**2
        visible = weights[:, self.spatial_snr > 0].sum(1)
        return visible / weights.sum(1)


def separate_network(
    local_a: np.ndarray,
    local_b: np.ndarray,
    local_h: np.ndarray,
    lead: np.ndarray,
    sensor_psd: np.ndarray,
    source_cov: np.ndarray,
    which: str = "filtered",
    rank_tolerance: float = 1e-10,
) -> SeparatedNetwork:
    """Apply the separation theorem: N independent single-column problems.

    ``sensor_psd`` is the (two-sided) sensor-noise intensity matrix ``R``.
    """
    r_factor = cholesky(_sym(sensor_psd), lower=True)
    s_half = _psd_factor(source_cov)
    whitened = np.linalg.solve(r_factor, lead) @ s_half
    _, sv, vt = np.linalg.svd(whitened, full_matrices=True)
    n_src = lead.shape[1]
    mu = np.zeros(n_src)
    mu[: len(sv)] = sv**2
    mu[mu < rank_tolerance * max(mu.max(), np.finfo(float).tiny)] = 0.0
    mixing = s_half @ vt.T
    p0 = stationary_covariance(local_a, local_b)
    posts = np.empty((n_src, len(local_a), len(local_a)))
    cache: dict[float, np.ndarray] = {}
    for i, value in enumerate(mu):
        if value == 0.0:
            posts[i] = p0
            continue
        key = float(value)
        if key not in cache:
            rec = stationary_recoverability(local_a, local_b, np.sqrt(value) * np.asarray(local_h), 1.0)
            cache[key] = getattr(rec, which)
        posts[i] = cache[key]
    return SeparatedNetwork(mu, mixing, p0, posts)
