"""Neural-mass families used to test the visibility theory.

Each family exposes the same interface:

``drift(x, p_mean)``          nonlinear vector field for a column (batched over rows)
``equilibria(p_mean)``        list of fixed points
``jacobian(x_star)``          local Jacobian
``input_vector()``            column of the noise/drive input (rank-one drive)
``output_vector()``           EEG-relevant readout (rank-one output)
``state_labels``              human-readable labels

All three families have a single stochastic drive and a single cortical
readout, which is the structural fact behind the residue bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.optimize import brentq, fsolve


def _sigmoid(v: np.ndarray, e0: float, v0: float, r: float) -> np.ndarray:
    exponent = np.clip(r * (v0 - v), -700.0, 700.0)
    return 2.0 * e0 / (1.0 + np.exp(exponent))


def _sigmoid_prime(v: np.ndarray, e0: float, v0: float, r: float) -> np.ndarray:
    s = _sigmoid(v, e0, v0, r)
    return r * s * (1.0 - s / (2.0 * e0))


# ---------------------------------------------------------------------------
# Jansen--Rit (Jansen & Rit 1995)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JansenRit:
    gain_e: float = 3.25
    gain_i: float = 22.0
    a: float = 100.0
    b: float = 50.0
    c1: float = 135.0
    c2: float = 108.0
    c3: float = 33.75
    c4: float = 33.75
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    name: str = "Jansen-Rit"
    state_labels: tuple[str, ...] = (
        "interneuron PSP driven by pyramidal firing x1",
        "excitatory PSP on pyramidal cells x2",
        "inhibitory PSP on pyramidal cells x3",
        "dx1/dt",
        "dx2/dt",
        "dx3/dt",
    )

    @property
    def dimension(self) -> int:
        return 6

    def s(self, v):
        return _sigmoid(v, self.e0, self.v0, self.r)

    def ds(self, v):
        return _sigmoid_prime(v, self.e0, self.v0, self.r)

    def drift(self, x: np.ndarray, p_mean: float) -> np.ndarray:
        x = np.atleast_2d(x)
        x1, x2, x3, v1, v2, v3 = x.T
        a, b = self.a, self.b
        out = np.empty_like(x)
        out[:, 0] = v1
        out[:, 1] = v2
        out[:, 2] = v3
        out[:, 3] = self.gain_e * a * self.s(x2 - x3) - 2 * a * v1 - a * a * x1
        out[:, 4] = self.gain_e * a * (p_mean + self.c2 * self.s(self.c1 * x1)) - 2 * a * v2 - a * a * x2
        out[:, 5] = self.gain_i * b * self.c4 * self.s(self.c3 * x1) - 2 * b * v3 - b * b * x3
        return out

    def equilibria(self, p_mean: float) -> list[np.ndarray]:
        ge, gi, a, b = self.gain_e, self.gain_i, self.a, self.b

        def rest(x1):
            x2 = ge / a * (p_mean + self.c2 * self.s(self.c1 * x1))
            x3 = gi / b * self.c4 * self.s(self.c3 * x1)
            return x2, x3

        def residual(x1):
            x2, x3 = rest(x1)
            return x1 - ge / a * self.s(x2 - x3)

        grid = np.linspace(0.0, ge / a * 2 * self.e0, 20001)
        values = residual(grid)
        roots = []
        for lo, hi, flo, fhi in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if flo == 0.0 or flo * fhi < 0:
                root = brentq(residual, lo, hi, xtol=1e-15)
                if not roots or abs(root - roots[-1]) > 1e-10:
                    roots.append(root)
        result = []
        for x1 in roots:
            x2, x3 = rest(x1)
            result.append(np.array([x1, x2, x3, 0.0, 0.0, 0.0]))
        return result

    def jacobian(self, x_star: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x_star[:3]
        a, b = self.a, self.b
        kp = self.gain_e * a * self.ds(x2 - x3)
        ke = self.gain_e * a * self.c1 * self.c2 * self.ds(self.c1 * x1)
        ki = self.gain_i * b * self.c3 * self.c4 * self.ds(self.c3 * x1)
        return np.array(
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

    def input_vector(self) -> np.ndarray:
        vec = np.zeros(6)
        vec[4] = self.gain_e * self.a
        return vec

    def output_vector(self) -> np.ndarray:
        return np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])

    def synaptic_rates(self) -> tuple[float, ...]:
        return (self.a, self.b)


# ---------------------------------------------------------------------------
# Wendling--Chauvel model with fast somatic inhibition
# (Wendling et al. 2002; background parameters of Koksal Ersoz et al. 2020)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Wendling:
    gain_e: float = 5.0
    gain_s: float = 40.0  # slow dendritic inhibition (B)
    gain_f: float = 35.0  # fast somatic inhibition (G)
    tau_a: float = 0.010
    tau_b: float = 0.050
    tau_g: float = 0.003
    c1: float = 135.0
    c2: float = 108.0
    c3: float = 35.0
    c4: float = 25.0
    c5: float = 450.0
    c6: float = 121.0
    c7: float = 121.0
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    name: str = "Wendling-Chauvel"
    state_labels: tuple[str, ...] = (
        "interneuron PSP driven by pyramidal firing y0",
        "excitatory PSP on pyramidal cells y1",
        "slow dendritic inhibitory PSP y2",
        "fast somatic inhibitory PSP y3",
        "inhibitory PSP onto fast interneurons y4",
        "dy0/dt",
        "dy1/dt",
        "dy2/dt",
        "dy3/dt",
        "dy4/dt",
    )

    @property
    def dimension(self) -> int:
        return 10

    def s(self, v):
        return _sigmoid(v, self.e0, self.v0, self.r)

    def ds(self, v):
        return _sigmoid_prime(v, self.e0, self.v0, self.r)

    def drift(self, x: np.ndarray, p_mean: float) -> np.ndarray:
        x = np.atleast_2d(x)
        y0, y1, y2, y3, y4, z0, z1, z2, z3, z4 = x.T
        ta, tb, tg = self.tau_a, self.tau_b, self.tau_g
        out = np.empty_like(x)
        out[:, :5] = x[:, 5:]
        out[:, 5] = self.gain_e / ta * self.s(y1 - y2 - y3) - 2 / ta * z0 - y0 / ta**2
        out[:, 6] = self.gain_e / ta * (p_mean + self.c2 * self.s(self.c1 * y0)) - 2 / ta * z1 - y1 / ta**2
        out[:, 7] = self.gain_s / tb * self.c4 * self.s(self.c3 * y0) - 2 / tb * z2 - y2 / tb**2
        out[:, 8] = self.gain_f / tg * self.c7 * self.s(self.c5 * y0 - self.c6 * y4) - 2 / tg * z3 - y3 / tg**2
        out[:, 9] = self.gain_s / tb * self.s(self.c3 * y0) - 2 / tb * z4 - y4 / tb**2
        return out

    def equilibria(self, p_mean: float) -> list[np.ndarray]:
        ge, gs, gf = self.gain_e, self.gain_s, self.gain_f
        ta, tb, tg = self.tau_a, self.tau_b, self.tau_g

        def rest(y0):
            y1 = ge * ta * (p_mean + self.c2 * self.s(self.c1 * y0))
            y2 = gs * tb * self.c4 * self.s(self.c3 * y0)
            y4 = gs * tb * self.s(self.c3 * y0)
            y3 = gf * tg * self.c7 * self.s(self.c5 * y0 - self.c6 * y4)
            return y1, y2, y3, y4

        def residual(y0):
            y1, y2, y3, _ = rest(y0)
            return y0 - ge * ta * self.s(y1 - y2 - y3)

        grid = np.linspace(0.0, ge * ta * 2 * self.e0, 40001)
        values = residual(grid)
        roots = []
        for lo, hi, flo, fhi in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if flo == 0.0 or flo * fhi < 0:
                root = brentq(residual, lo, hi, xtol=1e-15)
                if not roots or abs(root - roots[-1]) > 1e-10:
                    roots.append(root)
        result = []
        for y0 in roots:
            y1, y2, y3, y4 = rest(y0)
            result.append(np.array([y0, y1, y2, y3, y4, 0, 0, 0, 0, 0], dtype=float))
        return result

    def jacobian(self, x_star: np.ndarray) -> np.ndarray:
        y0, y1, y2, y3, y4 = x_star[:5]
        ta, tb, tg = self.tau_a, self.tau_b, self.tau_g
        jac = np.zeros((10, 10))
        jac[:5, 5:] = np.eye(5)
        dpyr = self.ds(y1 - y2 - y3)
        # pyramidal PSP
        jac[5, 1] = self.gain_e / ta * dpyr
        jac[5, 2] = -self.gain_e / ta * dpyr
        jac[5, 3] = -self.gain_e / ta * dpyr
        jac[5, 0] = -1 / ta**2
        jac[5, 5] = -2 / ta
        # excitatory feedback
        jac[6, 0] = self.gain_e / ta * self.c2 * self.c1 * self.ds(self.c1 * y0)
        jac[6, 1] = -1 / ta**2
        jac[6, 6] = -2 / ta
        # slow inhibition
        jac[7, 0] = self.gain_s / tb * self.c4 * self.c3 * self.ds(self.c3 * y0)
        jac[7, 2] = -1 / tb**2
        jac[7, 7] = -2 / tb
        # fast inhibition
        dfast = self.ds(self.c5 * y0 - self.c6 * y4)
        jac[8, 0] = self.gain_f / tg * self.c7 * self.c5 * dfast
        jac[8, 4] = -self.gain_f / tg * self.c7 * self.c6 * dfast
        jac[8, 3] = -1 / tg**2
        jac[8, 8] = -2 / tg
        # inhibition of fast interneurons
        jac[9, 0] = self.gain_s / tb * self.c3 * self.ds(self.c3 * y0)
        jac[9, 4] = -1 / tb**2
        jac[9, 9] = -2 / tb
        return jac

    def input_vector(self) -> np.ndarray:
        vec = np.zeros(10)
        vec[6] = self.gain_e / self.tau_a
        return vec

    def output_vector(self) -> np.ndarray:
        vec = np.zeros(10)
        vec[1], vec[2], vec[3] = 1.0, -1.0, -1.0
        return vec

    def synaptic_rates(self) -> tuple[float, ...]:
        return (1 / self.tau_a, 1 / self.tau_b, 1 / self.tau_g)


# ---------------------------------------------------------------------------
# Exact mean-field ("next-generation") E-I model with first-order synapses
# (Montbrio, Pazo & Roxin 2015; synaptic kinetics as in Devalle et al. 2017)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NextGenerationEI:
    # Default operating point: damped ~33 Hz E-I (gamma) focus at eta_e = -0.65.
    tau_e: float = 0.010
    tau_i: float = 0.010
    tau_se: float = 0.005
    tau_si: float = 0.020
    delta_e: float = 1.0
    delta_i: float = 1.0
    eta_i: float = -8.4
    j_ee: float = 10.3
    j_ei: float = 21.3
    j_ie: float = 22.2
    j_ii: float = 5.8
    name: str = "Next-generation E-I"
    state_labels: tuple[str, ...] = (
        "E firing rate",
        "E mean voltage",
        "I firing rate",
        "I mean voltage",
        "E synaptic activation",
        "I synaptic activation",
    )

    @property
    def dimension(self) -> int:
        return 6

    def drift(self, x: np.ndarray, p_mean: float) -> np.ndarray:
        """``p_mean`` is the mean excitability ``eta_e`` of the E population."""
        x = np.atleast_2d(x)
        re, ve, ri, vi, se, si = x.T
        te, ti = self.tau_e, self.tau_i
        out = np.empty_like(x)
        out[:, 0] = (self.delta_e / (np.pi * te) + 2 * re * ve) / te
        out[:, 1] = (ve**2 + p_mean - (np.pi * te * re) ** 2 + te * (self.j_ee * se - self.j_ei * si)) / te
        out[:, 2] = (self.delta_i / (np.pi * ti) + 2 * ri * vi) / ti
        out[:, 3] = (vi**2 + self.eta_i - (np.pi * ti * ri) ** 2 + ti * (self.j_ie * se - self.j_ii * si)) / ti
        out[:, 4] = (-se + re) / self.tau_se
        out[:, 5] = (-si + ri) / self.tau_si
        return out

    def equilibria(self, p_mean: float) -> list[np.ndarray]:
        found: list[np.ndarray] = []
        rng = np.random.default_rng(0)
        for _ in range(200):
            guess_re = rng.uniform(1, 200)
            guess_ri = rng.uniform(1, 200)
            start = np.array([
                guess_re, -self.delta_e / (2 * np.pi * self.tau_e * guess_re),
                guess_ri, -self.delta_i / (2 * np.pi * self.tau_i * guess_ri),
                guess_re, guess_ri,
            ])
            sol, info, ier, _ = fsolve(lambda z: self.drift(z, p_mean)[0], start, full_output=True, xtol=1e-13)
            if ier != 1 or sol[0] <= 0 or sol[2] <= 0:
                continue
            if np.max(np.abs(self.drift(sol, p_mean))) > 1e-6:
                continue
            if all(np.linalg.norm(sol - f) > 1e-6 * (1 + np.linalg.norm(f)) for f in found):
                found.append(sol)
        found.sort(key=lambda z: z[0])
        return found

    def jacobian(self, x_star: np.ndarray) -> np.ndarray:
        re, ve, ri, vi, _, _ = x_star
        te, ti = self.tau_e, self.tau_i
        jac = np.zeros((6, 6))
        jac[0, 0] = 2 * ve / te
        jac[0, 1] = 2 * re / te
        jac[1, 0] = -2 * np.pi**2 * te * re
        jac[1, 1] = 2 * ve / te
        jac[1, 4] = self.j_ee
        jac[1, 5] = -self.j_ei
        jac[2, 2] = 2 * vi / ti
        jac[2, 3] = 2 * ri / ti
        jac[3, 2] = -2 * np.pi**2 * ti * ri
        jac[3, 3] = 2 * vi / ti
        jac[3, 4] = self.j_ie
        jac[3, 5] = -self.j_ii
        jac[4, 0] = 1 / self.tau_se
        jac[4, 4] = -1 / self.tau_se
        jac[5, 2] = 1 / self.tau_si
        jac[5, 5] = -1 / self.tau_si
        return jac

    def input_vector(self) -> np.ndarray:
        vec = np.zeros(6)
        vec[1] = 1.0 / self.tau_e
        return vec

    def output_vector(self) -> np.ndarray:
        """Net synaptic current onto excitatory cells (LFP/EEG proxy)."""
        vec = np.zeros(6)
        vec[4], vec[5] = self.j_ee, -self.j_ei
        return vec

    def synaptic_rates(self) -> tuple[float, ...]:
        return (1 / self.tau_se, 1 / self.tau_si, 1 / self.tau_e, 1 / self.tau_i)


FAMILIES: dict[str, Callable[..., object]] = {
    "jansen_rit": JansenRit,
    "wendling": Wendling,
    "next_generation": NextGenerationEI,
}


def perturbed(model, rng: np.random.Generator, relative_sd: float, fields: tuple[str, ...]):
    """Return a copy of a family instance with log-normal parameter jitter."""
    changes = {
        name: float(getattr(model, name) * np.exp(relative_sd * rng.standard_normal()))
        for name in fields
    }
    return replace(model, **changes)
