"""Minimality, exact symmetries and spectral equivalence of linear columns.

For a family of linearised columns parameterised by log-parameters theta,
two parameter values are spectrally equivalent when they produce the same
output spectrum. For minimal, stable, minimum-phase realisations this means
the same transfer function up to sign, and the realisations are then related
by a unique similarity S (state-space isomorphism theorem). This module

* checks minimality with a scaled Popov--Belevitch--Hautus test,
* finds the continuous symmetry group from the exact rank of the map from
  parameters to transfer-function coefficients (rational arithmetic),
* recovers the infinitesimal similarity X for every symmetry direction,
* searches globally for spectrally equivalent parameters off the continuous
  orbits (discrete ambiguities), and
* tests whether symmetry directions are realised by physiological parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from neural_observability.theory import visibility as V
from neural_observability.theory.linear_families import (
    JansenRitLinear,
    NextGenerationLinear,
    WendlingLinear,
    reduce_wendling,
)


# ---------------------------------------------------------------------------
# Minimal realisations and minimality checks
# ---------------------------------------------------------------------------


def minimal_column(column, log_theta=None):
    """Return ``(a, b, h, embed)`` of the minimal realisation of a column."""
    if isinstance(column, WendlingLinear):
        return column.minimal_matrices(log_theta)
    a, b, h = column.matrices(log_theta)
    return a, b, h, np.eye(len(a))


def minimal_physiological(model, x_star):
    """Minimal realisation of a physiological family at an operating point."""
    a, b, h = model.jacobian(x_star), model.input_vector(), model.output_vector()
    if model.name == "Wendling-Chauvel":
        return reduce_wendling(a, b, h)
    return a, b, h, np.eye(len(a))


def pbh_margins(a: np.ndarray, b: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """Smallest PBH singular values (controllability, observability).

    Columns (rows) of the test matrices are normalised before the singular
    values are taken; values near machine precision indicate a non-minimal
    realisation.
    """
    n = len(a)
    ctrl, obs = np.inf, np.inf
    for lam in np.linalg.eigvals(a):
        m = np.hstack([lam * np.eye(n) - a, np.reshape(b, (n, 1))])
        m = m / np.linalg.norm(m, axis=0, keepdims=True).clip(1e-300)
        sv = np.linalg.svd(m, compute_uv=False)
        ctrl = min(ctrl, sv[-1] / sv[0])
        m = np.vstack([lam * np.eye(n) - a, np.reshape(h, (1, n))])
        m = m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-300)
        sv = np.linalg.svd(m, compute_uv=False)
        obs = min(obs, sv[-1] / sv[0])
    return float(ctrl), float(obs)


# ---------------------------------------------------------------------------
# Transfer-function coefficients and exact rank
# ---------------------------------------------------------------------------


def _sympy_family(column):
    """Symbolic minimal realisation (A, b, h) and parameter symbols."""
    import sympy as sp

    if isinstance(column, JansenRitLinear):
        a, b_, kp, ke, ki, sg = sp.symbols("a b k_p k_e k_i sigma", positive=True)
        A = sp.Matrix([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                       [-a**2, kp, -kp, -2 * a, 0, 0], [ke, -a**2, 0, 0, -2 * a, 0],
                       [ki, 0, -b_**2, 0, 0, -2 * b_]])
        B = sp.Matrix([0, 0, 0, 0, sg, 0])
        H = sp.Matrix([[0, 1, -1, 0, 0, 0]])
        return [a, b_, kp, ke, ki, sg], A, B, H, None
    if isinstance(column, WendlingLinear):
        al, be, ga, gp, ge, gs, gf, gfi, gi, sg = sp.symbols(
            "alpha beta gamma g_p g_e g_s g_f g_fi g_i sigma", positive=True)
        kap = gs / gi
        # minimal state (y0, y1, y3, y4, y0', y1', y3', y4')
        A = sp.zeros(8, 8)
        for i in range(4):
            A[i, i + 4] = 1
        A[4, 0], A[4, 4] = -al**2, -2 * al
        A[4, 1], A[4, 2], A[4, 3] = gp, -gp, -gp * kap
        A[5, 0], A[5, 1], A[5, 5] = ge, -al**2, -2 * al
        A[6, 0], A[6, 3], A[6, 2], A[6, 6] = gf, -gfi, -ga**2, -2 * ga
        A[7, 0], A[7, 3], A[7, 7] = gi, -be**2, -2 * be
        B = sp.zeros(8, 1)
        B[5] = sg
        H = sp.Matrix([[0, 1, -1, -kap, 0, 0, 0, 0]])
        embed = sp.zeros(10, 8)
        for k, full in enumerate([0, 1, 3, 4, 5, 6, 8, 9]):
            embed[full, k] = 1
        embed[2, 3] = kap
        embed[7, 7] = kap
        return [al, be, ga, gp, ge, gs, gf, gfi, gi, sg], A, B, H, embed
    if isinstance(column, NextGenerationLinear):
        syms = sp.symbols("d_e u_e w_e d_i u_i w_i J_ee J_ei J_ie J_ii k_se k_si sigma", positive=True)
        de, ue, we, di, ui, wi, jee, jei, jie, jii, kse, ksi, sg = syms
        A = sp.zeros(6, 6)
        A[0, 0], A[0, 1], A[1, 0], A[1, 1] = -de, ue, -we, -de
        A[1, 4], A[1, 5] = jee, -jei
        A[2, 2], A[2, 3], A[3, 2], A[3, 3] = -di, ui, -wi, -di
        A[3, 4], A[3, 5] = jie, -jii
        A[4, 0], A[4, 4] = kse, -kse
        A[5, 2], A[5, 5] = ksi, -ksi
        B = sp.zeros(6, 1)
        B[1] = sg
        H = sp.Matrix([[0, 0, 0, 0, jee, -jei]])
        return list(syms), A, B, H, None
    raise TypeError(type(column))


_SYMBOLIC_CACHE: dict = {}


def _symbolic_jacobian(column):
    """Symbolic Jacobian of transfer coefficients w.r.t. log-parameters (cached)."""
    import sympy as sp

    key = type(column)
    if key not in _SYMBOLIC_CACHE:
        params, A, B, H, _ = _sympy_family(column)
        s = sp.symbols("s")
        n = A.shape[0]
        M = s * sp.eye(n) - A
        den = sp.Poly(sp.expand(M.det(method="berkowitz")), s)
        num = sp.Poly(sp.expand((H * M.adjugate(method="berkowitz") * B)[0]), s)
        lead = den.LC()
        coeffs = [c / lead for c in den.all_coeffs()[1:]] + [c / lead for c in num.all_coeffs()]
        jac = sp.Matrix([[sp.diff(c, p) * p for p in params] for c in coeffs])
        _SYMBOLIC_CACHE[key] = (params, jac, len(coeffs), n)
    return _SYMBOLIC_CACHE[key]


def exact_nullspace_at(column, log_theta=None) -> np.ndarray:
    """Exact null space of the coefficient Jacobian at the column's parameters.

    The floating-point parameter values are converted to exact rationals and
    the null space is computed in rational arithmetic; rows are returned as
    floating-point vectors normalised to unit maximum modulus.
    """
    import sympy as sp

    params, jac, _, _ = _symbolic_jacobian(column)
    theta = np.asarray(column.log_theta if log_theta is None else log_theta, dtype=float)
    vals = {p: sp.Rational(float(v)) for p, v in zip(params, np.exp(theta))}
    null = jac.subs(vals).nullspace()
    rows = []
    for v in null:
        f = np.array([float(x) for x in v])
        rows.append(f / np.abs(f).max())
    return np.array(rows).reshape(len(rows), len(params))


def exact_symmetry_rank(column, points: int = 3, seed: int = 0, spread: float = 0.3):
    """Exact rank and rational null space of the coefficient Jacobian.

    The Jacobian of the monic denominator and numerator coefficients of the
    output transfer function with respect to the log-parameters is evaluated
    exactly (rational arithmetic) at ``points`` random rational parameter
    values near the column's own values. Its generic rank equals the number of
    locally identifiable parameter combinations; the null space gives the
    infinitesimal symmetries. Evaluation at random points gives the generic
    rank with probability one (Sedoglavic-type test).
    """
    import sympy as sp

    params, A, B, H, embed = _sympy_family(column)
    s = sp.symbols("s")
    n = A.shape[0]
    M = s * sp.eye(n) - A
    den = sp.Poly(sp.expand(M.det(method="berkowitz")), s)
    num = sp.Poly(sp.expand((H * M.adjugate(method="berkowitz") * B)[0]), s)
    lead = den.LC()
    coeffs = [c / lead for c in den.all_coeffs()[1:]] + [c / lead for c in num.all_coeffs()]
    # d coeff / d log p = p * d coeff / d p
    jac = sp.Matrix([[sp.diff(c, p) * p for p in params] for c in coeffs])
    rng = np.random.default_rng(seed)
    base = np.exp(np.asarray(column.log_theta))
    ranks, nulls = [], []
    for _ in range(points):
        vals = {p: sp.Rational(int(round(v * np.exp(spread * rng.standard_normal()) * 1000)), 1000)
                for p, v in zip(params, base)}
        jn = jac.subs(vals)
        ranks.append(int(jn.rank()))
        nulls.append([list(v) for v in jn.nullspace()])
    return {"parameters": [str(p) for p in params], "ranks": ranks, "nullspaces": nulls,
            "coefficients": len(coeffs), "state_dimension": n}


def integer_basis(vectors) -> np.ndarray:
    """Clean rational null-space vectors into a small-integer basis (rows)."""
    import sympy as sp

    if not vectors:
        return np.zeros((0, 0))
    m = sp.Matrix([[sp.nsimplify(x) for x in v] for v in vectors])
    m = m.rref()[0]
    rows = []
    for i in range(m.rows):
        row = m.row(i)
        if all(x == 0 for x in row):
            continue
        den = sp.ilcm(*[sp.fraction(x)[1] for x in row])
        rows.append([int(x * den) for x in row])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Infinitesimal similarities for symmetry directions
# ---------------------------------------------------------------------------


def _scaling(a, b):
    """Diagonal scaling to unit stationary standard deviations."""
    return np.sqrt(np.diag(V.stationary_covariance(a, b)))


def _solve_similarity_system(a0, b0, a1, rhs_a, rhs_b, h_left, rhs_h):
    """Least-squares solution S of  S a0 - a1 S = rhs_a,  S b0 = rhs_b,  h_left S = rhs_h.

    Blocks are weighted by the norms of their coefficient matrices. Returns S
    and the largest relative block residual.
    """
    n = len(a0)
    eye = np.eye(n)
    blocks = [(np.kron(a0.T, eye) - np.kron(eye, a1), np.asarray(rhs_a).reshape(-1, order="F")),
              (np.kron(np.reshape(b0, (1, n)), eye), np.asarray(rhs_b).reshape(-1)),
              (np.kron(eye, np.reshape(h_left, (1, n))), np.asarray(rhs_h).reshape(-1))]
    big = np.vstack([m / np.linalg.norm(m) for m, _ in blocks])
    rhs = np.concatenate([r / np.linalg.norm(m) for m, r in blocks])
    sol, *_ = np.linalg.lstsq(big, rhs, rcond=None)
    resid = max(np.linalg.norm(m @ sol - r) / (np.linalg.norm(m, 2) * np.linalg.norm(sol) + np.linalg.norm(r))
                for m, r in blocks)
    return sol.reshape(n, n, order="F"), float(resid)


def lie_generator(column, direction: np.ndarray, step: float = 1e-6):
    """Matrix X with dA = XA - AX, db = Xb, dh = -hX along ``direction``.

    Derivatives are taken of the minimal realisation, in coordinates with unit
    stationary variances. Returns X (original coordinates), the relative
    residual, and the induced change of the reconstruction map for every
    state of the original realisation, ``d(embed) + embed X``, with each row
    divided by the norm of the corresponding row of ``embed`` (rows that
    vanish correspond to states whose reconstruction is invariant).
    """
    theta = np.asarray(column.log_theta, dtype=float)
    a0, b0, h0, e0 = minimal_column(column, theta)
    d = _scaling(a0, b0)

    def scaled(th):
        a, b, h, e = minimal_column(column, th)
        return a * d[None, :] / d[:, None], b / d, h * d, e * d[None, :]

    az, bz, hz, ez = scaled(theta)
    ap, bp, hp, ep = scaled(theta + step * direction)
    am, bm, hm, em = scaled(theta - step * direction)
    da, db, dh, de = [(p - m) / (2 * step) for p, m in ((ap, am), (bp, bm), (hp, hm), (ep, em))]
    # X a - a X = da;  X b = db;  -h X = dh  ->  h X = -dh
    xz, resid = _solve_similarity_system(az, bz, az, da, db, hz, -dh)
    x = (xz * d[:, None]) / d[None, :]
    change = de + ez @ xz
    pz = V.stationary_covariance(az, bz)
    full_sd = np.sqrt(np.diag(ez @ pz @ ez.T)).clip(1e-300)
    recon = np.sqrt(np.clip(np.diag(change @ pz @ change.T), 0.0, None)) / full_sd
    return x, resid, recon


def similarity_between(column, theta_from, theta_to, sign: float = 1.0):
    """Similarity S with A1 = S A0 S^-1, b1 = sign S b0, h1 = h0 S^-1.

    Solved in coordinates with unit stationary variances of the source
    realisation. Returns S, the relative residual and the relative change of
    every reconstructed original state, ``embed_to S - embed_from``.
    """
    a0, b0, h0, e0 = minimal_column(column, theta_from)
    a1, b1, h1, e1 = minimal_column(column, theta_to)
    d0, d1 = _scaling(a0, b0), _scaling(a1, b1)
    a0z, b0z, h0z = a0 * d0[None, :] / d0[:, None], b0 / d0, h0 * d0
    a1z, b1z, h1z = a1 * d1[None, :] / d1[:, None], b1 / d1, h1 * d1
    # S a0 - a1 S = 0,  S b0 = sign b1,  h1 S = h0
    sz, resid = _solve_similarity_system(a0z, b0z, a1z, np.zeros_like(a0z), sign * b1z, h1z, h0z)
    s = (sz * d1[:, None]) / d0[None, :]
    # change of each reconstructed state, relative to its stationary s.d.
    change = e1 @ s - e0
    p0 = V.stationary_covariance(a0, b0)
    full_sd = np.sqrt(np.diag(e0 @ p0 @ e0.T)).clip(1e-300)
    rel = np.sqrt(np.clip(np.diag(change @ p0 @ change.T), 0.0, None)) / full_sd
    return s, resid, rel


# ---------------------------------------------------------------------------
# Global search for spectrally equivalent parameters
# ---------------------------------------------------------------------------


def reconstruction_difference(column, theta_true, theta_assumed, omega_max: float = 2 * np.pi * 4000.0):
    """Lost fraction of variance of every state when ``theta_assumed`` is used.

    This is the high-SNR floor of equation (3) between two parameter values:
    zero for state j exactly when both give the same output-to-state filter
    H_j on the support of the output spectrum. Uses the original (possibly
    non-minimal) realisation, whose transfer functions are well defined.
    """
    at, bt, ht = column.matrices(theta_true)
    an, bn, hn = column.matrices(theta_assumed)
    pt = V.stationary_covariance(at, bt)
    bt = bt / np.sqrt(ht @ pt @ ht)
    pn = V.stationary_covariance(an, bn)
    bn = bn / np.sqrt(hn @ pn @ hn)
    return V.high_snr_floor(at, bt, an, bn, ht, omega_max=omega_max, h_nom=hn)


def _response(column, theta, omegas):
    a, b, h, _ = minimal_column(column, theta)
    return V._frequency_response(a, b, h.reshape(1, -1), omegas)[:, 0]


NAMED_GENERATORS = {
    # log-parameter directions of exact one-parameter scaling groups, with the
    # minimal-realisation states they rescale
    "Jansen-Rit": {
        "x1 scale": ({"k_p": 1, "k_e": -1, "k_i": -1}, "interneuron PSP x1 and its derivative"),
    },
    "Wendling-Chauvel": {
        "y0 scale": ({"g_p": 1, "g_e": -1, "g_s": -1, "g_f": -1, "g_i": -1},
                     "interneuron PSP y0 and its derivative"),
        "y4 scale": ({"g_i": 1, "g_fi": -1}, "inhibitory PSP onto fast interneurons y4 and its derivative"),
    },
    "Next-generation E-I": {
        "E rate scale": ({"u_e": 1, "w_e": -1, "J_ee": -1, "J_ie": -1}, "E rate and E synaptic activation"),
        "I rate scale": ({"u_i": 1, "w_i": -1, "J_ei": -1, "J_ii": -1}, "I rate and I synaptic activation"),
        "I voltage scale": ({"u_i": -1, "w_i": 1, "J_ie": 1, "J_ii": 1}, "I mean voltage"),
    },
}


def named_generators(column) -> tuple[list[str], np.ndarray]:
    family = {JansenRitLinear: "Jansen-Rit", WendlingLinear: "Wendling-Chauvel",
              NextGenerationLinear: "Next-generation E-I"}[type(column)]
    names, rows = [], []
    for key, (coeffs, _) in NAMED_GENERATORS[family].items():
        v = np.zeros(len(column.labels))
        for label, value in coeffs.items():
            v[column.labels.index(label)] = value
        names.append(key)
        rows.append(v)
    return names, np.array(rows)


def response_jacobian(column, theta=None, f_hz=None, step: float = 1e-6) -> np.ndarray:
    """Jacobian of the relative output transfer function w.r.t. log-parameters."""
    theta = np.asarray(column.log_theta if theta is None else theta, dtype=float)
    f_hz = np.geomspace(0.2, 400.0, 120) if f_hz is None else f_hz
    omegas = 2 * np.pi * np.asarray(f_hz)
    g0 = _response(column, theta, omegas)
    cols = []
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += step
        tm[k] -= step
        d = (_response(column, tp, omegas) - _response(column, tm, omegas)) / (2 * step) / g0
        cols.append(np.concatenate([d.real, d.imag]))
    return np.column_stack(cols)


def equivalence_search(column, generators: np.ndarray, starts: int = 400, spread: float = 0.7,
                       seed: int = 0, f_hz=None, tolerance: float = 1e-9, exact_similarity: float = 1e-9):
    """Multistart search for parameters with the same transfer function.

    Starting points are drawn uniformly within +-``spread`` of the true
    log-parameters. Each solution with relative transfer-function residual
    below ``tolerance`` (either sign of G) is classified by its distance from
    the affine continuous orbit ``theta0 + span(generators)`` in log space.
    """
    from scipy.optimize import least_squares

    theta0 = np.asarray(column.log_theta, dtype=float)
    f_hz = np.geomspace(0.2, 400.0, 80) if f_hz is None else f_hz
    omegas = 2 * np.pi * np.asarray(f_hz)
    g0 = _response(column, theta0, omegas)
    weight = 1.0 / np.abs(g0)
    if generators.size:
        q, _ = np.linalg.qr(generators.T)
        project = np.eye(len(theta0)) - q @ q.T
    else:
        project = np.eye(len(theta0))
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(starts):
        start = theta0 + rng.uniform(-spread, spread, len(theta0))
        best = None
        for sign in (1.0, -1.0):
            def resid(th, sign=sign):
                try:
                    g = _response(column, th, omegas)
                except np.linalg.LinAlgError:
                    return np.full(2 * len(omegas), 1e3)
                d = (sign * g - g0) * weight
                return np.concatenate([d.real, d.imag])
            try:
                sol = least_squares(resid, start, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000,
                                    bounds=(theta0 - 3.0, theta0 + 3.0))
            except (ValueError, np.linalg.LinAlgError):
                continue
            cost = float(np.sqrt(np.mean(sol.fun**2)))
            if best is None or cost < best[0]:
                best = (cost, sol.x, sign)
        if best is None:
            continue
        cost, th, sign = best
        a, _, _, _ = minimal_column(column, th)
        stable = bool(np.linalg.eigvals(a).real.max() < 0)
        entry = {"residual": cost, "sign": sign, "stable": stable,
                 "off_orbit": float(np.linalg.norm(project @ (th - theta0))), "theta": th.tolist()}
        if cost < tolerance and stable:
            _, s_resid, _ = similarity_between(column, theta0, th, sign)
            entry["similarity_residual"] = s_resid
            entry["state_change"] = reconstruction_difference(column, theta0, th).tolist()
        found.append(entry)
    converged = [f for f in found if "state_change" in f]
    exact = [f for f in converged if f["similarity_residual"] < exact_similarity]
    near = [f for f in converged if f["similarity_residual"] >= exact_similarity]

    def max_change(rows):
        return np.array([f["state_change"] for f in rows]).max(0).tolist() if rows else []

    return {"starts": starts, "transfer_tolerance": tolerance, "exact_similarity_tolerance": exact_similarity,
            "exact_equivalents": len(exact), "near_equivalents": len(near),
            "max_state_change_exact": max_change(exact), "max_state_change_near": max_change(near),
            "solutions": converged, "residuals": [f["residual"] for f in found]}


# ---------------------------------------------------------------------------
# Physiological realisability of symmetry directions
# ---------------------------------------------------------------------------


PHYSIOLOGICAL = {
    "Jansen-Rit": ("gain_e", "gain_i", "a", "b", "c1", "c2", "c3", "c4", "e0", "v0", "r"),
    "Wendling-Chauvel": ("gain_e", "gain_s", "gain_f", "tau_a", "tau_b", "tau_g",
                         "c1", "c2", "c3", "c4", "c5", "c6", "c7", "e0", "v0", "r"),
    "Next-generation E-I": ("tau_e", "tau_i", "tau_se", "tau_si", "delta_e", "delta_i",
                            "eta_i", "j_ee", "j_ei", "j_ie", "j_ii"),
}


def physiological_jacobian(model, drive: float, x_star: np.ndarray, column_factory, step: float = 1e-5):
    """Jacobian of linear log-parameters w.r.t. physiological log-parameters.

    Columns: every physiological parameter (log scale; additive for eta_i),
    the mean drive (additive) and the drive-noise amplitude (log). The
    operating point is re-solved on the branch nearest ``x_star``.
    """
    from neural_observability.theory.atlas import _closest_equilibrium

    names = PHYSIOLOGICAL[model.name]

    def theta_of(m, d, noise_log):
        xs = _closest_equilibrium(m, d, x_star)
        if xs is None:
            raise RuntimeError("operating point lost")
        th = np.array(column_factory(m, xs).log_theta, dtype=float)
        th[-1] += noise_log  # drive gain is the last linear parameter in every family
        return th

    cols = []
    for name in names:
        v = getattr(model, name)
        if name == "eta_i":
            plus, minus = replace(model, **{name: v + step}), replace(model, **{name: v - step})
        else:
            plus, minus = replace(model, **{name: v * np.exp(step)}), replace(model, **{name: v * np.exp(-step)})
        cols.append((theta_of(plus, drive, 0.0) - theta_of(minus, drive, 0.0)) / (2 * step))
    scale = max(abs(drive), 1.0)
    cols.append((theta_of(model, drive + step * scale, 0.0) - theta_of(model, drive - step * scale, 0.0)) / (2 * step))
    cols.append((theta_of(model, drive, step) - theta_of(model, drive, -step)) / (2 * step))
    return np.column_stack(cols), list(names) + ["mean drive", "noise amplitude"]


@dataclass
class RealisabilityResult:
    rank: int
    linear_parameters: int
    generator_residuals: list[float]


def realisability(jac_phys: np.ndarray, generators: np.ndarray, tol: float = 1e-8) -> RealisabilityResult:
    """Rank of the physiological map and distance of generators from its range."""
    u, sv, _ = np.linalg.svd(jac_phys, full_matrices=False)
    rank = int(np.sum(sv > tol * sv[0]))
    basis = u[:, :rank]
    resid = []
    for g in generators:
        g = g / np.linalg.norm(g)
        resid.append(float(np.linalg.norm(g - basis @ (basis.T @ g))))
    return RealisabilityResult(rank, jac_phys.shape[0], resid)
