"""Limiting error of the mismatched filter for the fitted recordings and the forecast.

Equation (2.3) is the high-SNR limit of the error of the assumed model's filter
run on the true system when the assumed model satisfies theorem 3 (stable,
minimal, no transfer zeros in the closed right half-plane). For every fitted
model, and for every posterior draw of the forecast in which the drawn model is
the assumed one, the transfer zeros are computed. Where the assumed model has a
right-half-plane zero, the error of its steady-state Kalman filter on the true
system is evaluated directly from the joint stationary covariance of true and
estimated states, at increasing SNR. The same direct evaluation is made for every
fit as a check of the closed form.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_finite_record_fits as F
from neural_observability.theory import visibility as V
from neural_observability.theory.symmetry import minimal_column

SNRS_DB = (40, 60, 80, 100, 120, 140)
LIMIT_DB = 140
SOURCE = Path("results/simulations/finite_record_fits/verification_fit.json")
OUT = Path("results/simulations/finite_record_fits/verification_filter.json")


def normalised(col, theta):
    a, b, h = col.matrices(theta)
    return a, b / np.sqrt(h @ V.stationary_covariance(a, b) @ h), h


def max_zero(col, theta):
    a, b, h, _ = minimal_column(col, theta)
    z = V.transmission_zeros(a, b, h)
    return float(z.real.max()) if len(z) else -np.inf


def exact_errors(col, theta_true, theta_assumed, snrs=SNRS_DB):
    at, bt, ht = normalised(col, theta_true)
    an, bn, hn = normalised(col, theta_assumed)
    out = {}
    for s in snrs:
        try:
            res = V.mismatched_filter_error(at, bt, an, bn, ht, 1.0 / (200.0 * 10 ** (s / 10.0)), h_nom=hn)
            out[f"{s:g}"] = (1.0 - res["r2"]).tolist()
        except (ValueError, np.linalg.LinAlgError):
            out[f"{s:g}"] = None
    return out


def limiting_error(col, theta_true, theta_assumed):
    """Equation (2.3) when it applies, otherwise the exact filter error at LIMIT_DB."""
    mz = max_zero(col, theta_assumed)
    if mz < 0:
        return F.realised_loss(col, theta_true, theta_assumed), mz, False
    exact = exact_errors(col, theta_true, theta_assumed, (LIMIT_DB,))[f"{LIMIT_DB:g}"]
    return np.asarray(exact), mz, True


def generating_parameters(col):
    theta0 = np.asarray(col.log_theta, dtype=float)
    _, b, _ = F.normalised_minimal(col, theta0)
    theta = theta0.copy()
    theta[-1] += np.log(np.linalg.norm(b) / np.linalg.norm(minimal_column(col, theta0)[1]))
    return theta


def forecast_draws(family, seconds_list, draws, seed):
    """Repeat the forecast of run_finite_record_fits with the same random draws."""
    from neural_observability.theory.linear_families import verification_analysis

    col = F.column_for(family)
    theta0 = np.asarray(col.log_theta, dtype=float)
    a, b, h = col.matrices(theta0)
    theta_true = theta0.copy()
    theta_true[-1] -= np.log(np.sqrt(h @ V.stationary_covariance(a, b) @ h))
    col_true = type(col)(col.name, col.labels, theta_true, col.state_labels, col.psp_states)
    fisher = verification_analysis(col_true, snr_db=F.SNR_DB)["fisher_per_second"]
    rng = np.random.default_rng(seed)
    out = {}
    for seconds in seconds_list:
        cov = np.linalg.inv(np.eye(len(theta0)) / F.PRIOR_SD**2 + seconds * fisher)
        chol = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-15 * np.eye(len(cov)))
        est, atlas, rhp = [], [], 0
        for _ in range(draws):
            delta = chol @ rng.standard_normal(len(theta0))
            try:
                loss, mz, direct = limiting_error(col, theta_true, theta_true + delta)
                atlas_loss = F.realised_loss(col, theta_true + delta, theta_true)
            except (ValueError, np.linalg.LinAlgError):
                continue
            est.append(loss)
            atlas.append(atlas_loss)
            rhp += int(direct)
        out[f"{seconds:g}"] = {"fitted_record": np.mean(est, 0).tolist(), "atlas_roles": np.mean(atlas, 0).tolist(),
                               "draws": len(est), "draws_with_rhp_zero": rhp}
    return out


def main():
    data = json.loads(SOURCE.read_text())
    seed = data["config"]["seed"]
    result = {"limit_snr_db": LIMIT_DB, "families": {}}
    for i, (family, v) in enumerate(data["families"].items()):
        col = F.column_for(family)
        theta_true = generating_parameters(col)
        fits = []
        for r in v["replicates"]:
            theta_fit = theta_true + np.asarray(r["theta_error"])
            loss, mz, direct = limiting_error(col, theta_true, theta_fit)
            fits.append({"seconds": r["seconds"], "rep": r["rep"], "max_zero_real": mz, "direct": direct,
                         "closed_form": r["loss"], "limit": np.asarray(loss).tolist(),
                         "exact_by_snr": exact_errors(col, theta_true, theta_fit)})
        seconds = sorted({r["seconds"] for r in fits})
        summary = {}
        for s in seconds:
            losses = np.array([f["limit"] for f in fits if f["seconds"] == s])
            summary[f"{s:g}"] = {"mean": losses.mean(0).tolist(),
                                 "se": (losses.std(0, ddof=1) / np.sqrt(len(losses))).tolist(), "n": len(losses)}
        fc = forecast_draws(family, seconds, 256, seed + i)
        result["families"][family] = {"state_labels": v["state_labels"], "fits": fits, "realised": summary,
                                      "forecast": fc}
        print(family, {k: (np.round(summary[k]["mean"], 4).tolist(), np.round(fc[k]["fitted_record"], 4).tolist(),
                           fc[k]["draws_with_rhp_zero"]) for k in summary}, flush=True)
    OUT.write_text(json.dumps(result))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
