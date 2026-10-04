"""Bootstrap particle filtering of nonlinear stochastic neural-mass columns.

Many filtering problems (conditions) are run in parallel on one device: each
condition has its own true trajectory, observation noise level and assumed
(filter) parameters. All arithmetic is float64.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import torch


def _sig(v, e0, v0, r):
    return 2.0 * e0 / (1.0 + torch.exp(torch.clamp(r * (v0 - v), -700.0, 700.0)))


class TorchFamily:
    """Batched drift for one family; parameters have shape (runs, 1)."""

    def __init__(self, models: Sequence, drives: Sequence[float], device):
        self.name = models[0].name
        self.device = device
        keys = list(asdict(models[0]).keys())
        numeric = [k for k in keys if isinstance(getattr(models[0], k), (int, float))]
        self.p = {
            k: torch.tensor([[float(getattr(m, k))] for m in models], dtype=torch.float64, device=device)
            for k in numeric
        }
        self.drive = torch.tensor([[float(d)] for d in drives], dtype=torch.float64, device=device)
        self.dim = models[0].dimension
        if self.name == "Jansen-Rit":
            self.noise_index = 4
            self.noise_gain = self.p["gain_e"] * self.p["a"]
            self.readout = torch.zeros((len(models), self.dim), dtype=torch.float64, device=device)
            self.readout[:, 1], self.readout[:, 2] = 1.0, -1.0
        elif self.name == "Wendling-Chauvel":
            self.noise_index = 6
            self.noise_gain = self.p["gain_e"] / self.p["tau_a"]
            self.readout = torch.zeros((len(models), self.dim), dtype=torch.float64, device=device)
            self.readout[:, 1], self.readout[:, 2], self.readout[:, 3] = 1.0, -1.0, -1.0
        elif self.name == "Next-generation E-I":
            self.noise_index = 1
            self.noise_gain = 1.0 / self.p["tau_e"]
            self.readout = torch.zeros((len(models), self.dim), dtype=torch.float64, device=device)
            self.readout[:, 4] = self.p["j_ee"][:, 0]
            self.readout[:, 5] = -self.p["j_ei"][:, 0]
        else:
            raise ValueError(self.name)

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (runs, particles, dim)."""
        p, d = self.p, self.drive
        out = torch.empty_like(x)
        if self.name == "Jansen-Rit":
            e0, v0, r = p["e0"], p["v0"], p["r"]
            x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
            a, b = p["a"], p["b"]
            out[..., 0:3] = x[..., 3:6]
            out[..., 3] = p["gain_e"] * a * _sig(x2 - x3, e0, v0, r) - 2 * a * x[..., 3] - a * a * x1
            out[..., 4] = p["gain_e"] * a * (d + p["c2"] * _sig(p["c1"] * x1, e0, v0, r)) - 2 * a * x[..., 4] - a * a * x2
            out[..., 5] = p["gain_i"] * b * p["c4"] * _sig(p["c3"] * x1, e0, v0, r) - 2 * b * x[..., 5] - b * b * x3
        elif self.name == "Wendling-Chauvel":
            e0, v0, r = p["e0"], p["v0"], p["r"]
            ta, tb, tg = p["tau_a"], p["tau_b"], p["tau_g"]
            y0, y1, y2, y3, y4 = (x[..., i] for i in range(5))
            z = [x[..., 5 + i] for i in range(5)]
            out[..., 0:5] = x[..., 5:10]
            out[..., 5] = p["gain_e"] / ta * _sig(y1 - y2 - y3, e0, v0, r) - 2 / ta * z[0] - y0 / ta**2
            out[..., 6] = p["gain_e"] / ta * (d + p["c2"] * _sig(p["c1"] * y0, e0, v0, r)) - 2 / ta * z[1] - y1 / ta**2
            out[..., 7] = p["gain_s"] / tb * p["c4"] * _sig(p["c3"] * y0, e0, v0, r) - 2 / tb * z[2] - y2 / tb**2
            out[..., 8] = p["gain_f"] / tg * p["c7"] * _sig(p["c5"] * y0 - p["c6"] * y4, e0, v0, r) - 2 / tg * z[3] - y3 / tg**2
            out[..., 9] = p["gain_s"] / tb * _sig(p["c3"] * y0, e0, v0, r) - 2 / tb * z[4] - y4 / tb**2
        else:
            te, ti = p["tau_e"], p["tau_i"]
            re, ve, ri, vi, se, si = (x[..., i] for i in range(6))
            out[..., 0] = (p["delta_e"] / (np.pi * te) + 2 * re * ve) / te
            out[..., 1] = (ve**2 + d - (np.pi * te * re) ** 2 + te * (p["j_ee"] * se - p["j_ei"] * si)) / te
            out[..., 2] = (p["delta_i"] / (np.pi * ti) + 2 * ri * vi) / ti
            out[..., 3] = (vi**2 + p["eta_i"] - (np.pi * ti * ri) ** 2 + ti * (p["j_ie"] * se - p["j_ii"] * si)) / ti
            out[..., 4] = (-se + re) / p["tau_se"]
            out[..., 5] = (-si + ri) / p["tau_si"]
        return out

    def step(self, x: torch.Tensor, dt: float, noise_sd: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        x = x + self.drift(x) * dt
        kick = torch.randn(x.shape[:-1], dtype=torch.float64, device=self.device, generator=gen)
        x[..., self.noise_index] += (self.noise_gain * noise_sd) * np.sqrt(dt) * kick
        if self.name == "Next-generation E-I":
            x[..., 0] = torch.clamp(x[..., 0], min=1e-6)
            x[..., 2] = torch.clamp(x[..., 2], min=1e-6)
        return x

    def output(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.readout[:, None, :]).sum(-1)


def simulate_truth(truth: TorchFamily, x0: torch.Tensor, noise_sd: torch.Tensor, dt: float,
                   substeps: int, samples: int, gen: torch.Generator) -> torch.Tensor:
    """Return true states at observation times, shape (runs, samples, dim)."""
    x = x0[:, None, :].clone()
    states = torch.empty((x0.shape[0], samples, x0.shape[1]), dtype=torch.float64, device=x0.device)
    for k in range(samples):
        for _ in range(substeps):
            x = truth.step(x, dt, noise_sd, gen)
        states[:, k] = x[:, 0]
    return states


def extended_kalman_filter(model: TorchFamily, observations: torch.Tensor, obs_sd: torch.Tensor,
                           x_init: torch.Tensor, noise_sd: torch.Tensor, dt: float, substeps: int,
                           warmup_steps: int, fd_step: float = 1e-6):
    """Continuous--discrete extended Kalman filter, batched over conditions.

    The mean follows the Euler scheme of the drift; the covariance follows
    P <- P + dt (J P + P J^T + q q^T), with J the finite-difference Jacobian
    of the drift at the current mean. Observations update the state through
    the linear readout. Returns filtered means (runs, samples, dim).
    """
    runs, samples = observations.shape
    dim = x_init.shape[-1]
    dev = x_init.device
    x = x_init[:, None, :].clone()
    q = torch.zeros((runs, dim), dtype=torch.float64, device=dev)
    q[:, model.noise_index] = (model.noise_gain * noise_sd)[:, 0] if torch.is_tensor(model.noise_gain) \
        else model.noise_gain * noise_sd[:, 0]
    qq = q[:, :, None] * q[:, None, :]
    p = torch.zeros((runs, dim, dim), dtype=torch.float64, device=dev)
    eye = torch.eye(dim, dtype=torch.float64, device=dev)

    def jac(xm):
        base = model.drift(xm)  # (runs, 1, dim)
        pert = xm + fd_step * (torch.abs(xm) + 1e-3) * eye[None, :, :]  # (runs, dim, dim): row k perturbs state k
        scale = fd_step * (torch.abs(xm) + 1e-3)  # (runs, 1, dim)
        return ((model.drift(pert) - base) / scale.transpose(1, 2)).transpose(1, 2)  # (runs, dim, dim)

    def predict(xm, pm):
        j = jac(xm)
        xm = xm + model.drift(xm) * dt
        pm = pm + dt * (j @ pm + pm @ j.transpose(1, 2) + qq)
        if model.name == "Next-generation E-I":
            xm[..., 0] = torch.clamp(xm[..., 0], min=1e-6)
            xm[..., 2] = torch.clamp(xm[..., 2], min=1e-6)
        return xm, 0.5 * (pm + pm.transpose(1, 2))

    for _ in range(warmup_steps):
        x, p = predict(x, p)
    h = model.readout  # (runs, dim)
    r = obs_sd[:, 0] ** 2
    means = torch.empty((runs, samples, dim), dtype=torch.float64, device=dev)
    for k in range(samples):
        for _ in range(substeps):
            x, p = predict(x, p)
        ph = (p @ h[:, :, None])[..., 0]  # (runs, dim)
        s = (h * ph).sum(-1) + r
        gain = ph / s[:, None]
        innov = observations[:, k] - (x[:, 0] * h).sum(-1)
        x = x + (gain * innov[:, None])[:, None, :]
        p = p - gain[:, :, None] * ph[:, None, :]
        p = 0.5 * (p + p.transpose(1, 2))
        means[:, k] = x[:, 0]
    return means


def error_decomposition(est: torch.Tensor, truth: torch.Tensor) -> dict[str, np.ndarray]:
    """Normalised error of estimates (runs, time, dim) split into bias and fluctuation.

    nmse_total = nmse_bias + nmse_fluctuation, all divided by the variance of
    the true state over the scoring window.
    """
    var = truth.var(1, unbiased=False)
    bias = (est.mean(1) - truth.mean(1)) ** 2
    fluct = (((est - est.mean(1, keepdim=True)) - (truth - truth.mean(1, keepdim=True))) ** 2).mean(1)
    return {"nmse_total": ((bias + fluct) / var).cpu().numpy(), "nmse_bias": (bias / var).cpu().numpy(),
            "nmse_fluctuation": (fluct / var).cpu().numpy()}


def systematic_resample(weights: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    runs, n = weights.shape
    u = (torch.rand((runs, 1), dtype=torch.float64, device=weights.device, generator=gen)
         + torch.arange(n, dtype=torch.float64, device=weights.device)[None, :]) / n
    cdf = torch.cumsum(weights, dim=1)
    cdf[:, -1] = 1.0
    return torch.searchsorted(cdf, u).clamp(max=n - 1)


def particle_filter(model: TorchFamily, observations: torch.Tensor, obs_sd: torch.Tensor,
                    x_init: torch.Tensor, noise_sd: torch.Tensor, dt: float, substeps: int,
                    particles: int, warmup_steps: int, gen: torch.Generator):
    """Filtered means (runs, samples, dim) and effective sample sizes (runs, samples)."""
    runs, samples = observations.shape
    x = x_init[:, None, :].repeat(1, particles, 1)
    x = x * (1.0 + 0.05 * torch.randn(x.shape, dtype=torch.float64, device=x.device, generator=gen))
    for _ in range(warmup_steps):
        x = model.step(x, dt, noise_sd, gen)
    means = torch.empty((runs, samples, x.shape[-1]), dtype=torch.float64, device=x.device)
    ess = torch.empty((runs, samples), dtype=torch.float64, device=x.device)
    for k in range(samples):
        for _ in range(substeps):
            x = model.step(x, dt, noise_sd, gen)
        resid = observations[:, k:k + 1] - model.output(x)
        logw = -0.5 * (resid / obs_sd) ** 2
        bad = ~torch.isfinite(logw) | ~torch.isfinite(x).all(-1)
        logw = torch.where(bad, torch.full_like(logw, -torch.inf), logw)
        top = logw.max(dim=1, keepdim=True).values
        top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
        w = torch.exp(logw - top)
        total = w.sum(dim=1, keepdim=True)
        w = torch.where(total > 0, w / total.clamp(min=1e-300), torch.full_like(w, 1.0 / w.shape[1]))
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        means[:, k] = (w[..., None] * x).sum(1)
        ess[:, k] = 1.0 / (w**2).sum(1)
        idx = systematic_resample(w, gen)
        x = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
    return means, ess
