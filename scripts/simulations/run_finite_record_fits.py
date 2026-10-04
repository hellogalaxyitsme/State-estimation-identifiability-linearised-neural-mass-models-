"""Check the information-based finite-record forecast against actual estimation.

For representative operating points, stationary recordings of T seconds are
simulated exactly from the linearised column (sampled at 1 kHz, white sensor
noise at 10 dB), the linear parameters are estimated by maximising the
Whittle likelihood of the exact sampled-process spectrum plus a log-normal
prior (s.d. 0.2 in log space, centred on a random draw from that prior) by
Fisher scoring with staged likelihood weighting, and the hidden-state loss of
the fitted model is computed with the closed-form high-SNR floor. The mean
realised loss is compared with the forecast obtained from the Whittle Fisher
information and the same prior. run_finite_record_filter.py replaces the floor
by the directly evaluated filter error where the fitted model has a
right-half-plane zero.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import expm, cholesky
from scipy.signal import lfilter

from neural_observability.theory import visibility as V
from neural_observability.theory.atlas import wendling_2002
from neural_observability.theory.families import JansenRit, NextGenerationEI
from neural_observability.theory.linear_families import linear_column
from neural_observability.theory.symmetry import minimal_column

FS = 1000.0
SNR_DB = 10.0
PRIOR_SD = 0.2
POINTS = {"jansen_rit": (JansenRit, 90.0), "wendling_2002": (wendling_2002, 90.0),
          "next_generation": (NextGenerationEI, -0.65)}


def column_for(family):
    factory, drive = POINTS[family]
    model = factory()
    return linear_column(model, model.equilibria(drive)[0])


def normalised_minimal(col, theta):
    a, b, h, _ = minimal_column(col, theta)
    p = V.stationary_covariance(a, b)
    return a, b / np.sqrt(h @ p @ h), h


def sensor_variance():
    r = 1.0 / (200.0 * 10 ** (SNR_DB / 10.0))  # continuous PSD for unit output variance
    return r * FS


def discrete_system(a, b):
    ad = expm(a / FS)
    p = V.stationary_covariance(a, b)
    qd = p - ad @ p @ ad.T
    return ad, 0.5 * (qd + qd.T), p


def simulate(a, b, h, seconds, rng):
    ad, qd, p = discrete_system(a, b)
    n = len(a)
    lam, vec = np.linalg.eig(ad)
    vinv = np.linalg.inv(vec)
    w, u = np.linalg.eigh(qd)
    lq = u * np.sqrt(np.clip(w, 0, None))
    samples = int(seconds * FS)
    eps = rng.standard_normal((samples, n)) @ lq.T
    x0 = cholesky(p + 1e-300 * np.eye(n), lower=True) @ rng.standard_normal(n) if np.all(np.linalg.eigvalsh(p) > 0) \
        else np.zeros(n)
    z0 = vinv @ x0
    drive = eps @ vinv.T  # modal innovations
    z = np.empty((samples, n), dtype=complex)
    for k in range(n):
        z[:, k], _ = lfilter([1.0], [1.0, -lam[k]], drive[:, k], zi=[lam[k] * z0[k]])
    y = (z @ (h @ vec)).real
    return y + np.sqrt(sensor_variance()) * rng.standard_normal(samples)


def discrete_spectrum(a, b, h, omegas):
    ad, qd, _ = discrete_system(a, b)
    w, u = np.linalg.eigh(qd)
    lq = u * np.sqrt(np.clip(w, 0, None))
    n = len(a)
    zs = np.exp(1j * omegas)
    resp = np.linalg.solve(zs[:, None, None] * np.eye(n)[None] - ad[None], np.broadcast_to(lq, (len(zs), n, n)))
    g = np.einsum("i,fij->fj", h, resp)
    return (np.abs(g) ** 2).sum(1) + sensor_variance()


def binned_periodogram(y, max_bins=4096):
    n = len(y)
    f = np.fft.rfft(y - y.mean())
    per = np.abs(f[1:n // 2]) ** 2 / n
    om = 2 * np.pi * np.arange(1, n // 2) / n
    m = max(1, int(np.ceil(len(per) / max_bins)))
    k = (len(per) // m) * m
    return om[:k].reshape(-1, m).mean(1), per[:k].reshape(-1, m).mean(1), m


def _stable(col, theta):
    try:
        return np.linalg.eigvals(minimal_column(col, theta)[0]).real.max() < -1e-6
    except (np.linalg.LinAlgError, ValueError):
        return False


class FitResult:
    def __init__(self, x, fun, success, iterations):
        self.x, self.fun, self.success, self.nit = x, fun, success, iterations


def fit(col, y, prior_mean, rng, starts: int = 6):
    """Maximum a posteriori estimate (Whittle likelihood + log-normal prior).

    The negative log posterior is minimised by Fisher scoring with
    Levenberg-Marquardt damping. To avoid local optima of long records, the
    likelihood is introduced by continuation: it is first weighted by
    4^-K (the information of a record about 15-60 s long) and the weight is
    raised fourfold at each stage, each stage starting from the previous
    optimum. Starting points are the prior mean and small perturbations of it;
    the run with the lowest final objective is kept. The returned objective is
    divided by the number of periodogram ordinates.
    """
    om, per, m = binned_periodogram(y)
    count = m * len(om)
    p = len(prior_mean)

    def inside(theta):
        return np.max(np.abs(theta - prior_mean)) <= 5 * PRIOR_SD  # five prior standard deviations

    def spectrum(theta):
        try:
            a, b, h = minimal_column(col, theta)[:3]
            if np.linalg.eigvals(a).real.max() >= -1e-6:
                return None
            spec = discrete_spectrum(a, b, h, om)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return None
        if not np.all(np.isfinite(spec)) or np.any(spec <= 0):
            return None
        return spec

    def neg_log_post(theta, weight, spec=None):
        if not inside(theta):
            return np.inf
        spec = spectrum(theta) if spec is None else spec
        if spec is None:
            return np.inf
        return weight * m * np.sum(np.log(spec) + per / spec) + 0.5 * np.sum((theta - prior_mean) ** 2) / PRIOR_SD**2

    def objective(theta):
        value = neg_log_post(theta, 1.0)
        return value / count if np.isfinite(value) else 1e6

    def score_and_fisher(theta, spec, step=1e-5):
        jac = np.empty((len(om), p))
        for k in range(p):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += step
            tm[k] -= step
            sp, sm = spectrum(tp), spectrum(tm)
            if sp is None or sm is None:
                return None, None
            jac[:, k] = (np.log(sp) - np.log(sm)) / (2 * step)
        grad = m * jac.T @ (1.0 - per / spec)
        fisher = m * jac.T @ jac
        return grad, fisher

    def scoring(theta, weight, max_iter=200):
        prior_prec = np.eye(p) / PRIOR_SD**2
        spec = spectrum(theta)
        value = neg_log_post(theta, weight, spec)
        lam, it, done = 1e-3, 0, False
        while it < max_iter and spec is not None:
            it += 1
            g, fisher = score_and_fisher(theta, spec)
            if g is None:
                break
            g = weight * g + prior_prec @ (theta - prior_mean)
            hess = weight * fisher + prior_prec
            newton = np.linalg.solve(hess, g)
            if g @ newton < 1e-6:  # Newton decrement below 1e-6 nats
                done = True
                break
            accepted = False
            for _ in range(30):
                step = np.linalg.solve(hess + lam * np.diag(np.diag(hess)), -g)
                cand = theta + step
                spec_c = spectrum(cand) if inside(cand) else None
                val_c = neg_log_post(cand, weight, spec_c) if spec_c is not None else np.inf
                if val_c < value:
                    theta, spec, value = cand, spec_c, val_c
                    lam = max(lam / 3, 1e-9)
                    accepted = True
                    break
                lam *= 4
            if not accepted:
                done = True  # no descent direction at machine precision
                break
        return theta, value, done, it

    candidates = [prior_mean] if _stable(col, prior_mean) else []
    tries = 0
    while len(candidates) < starts and tries < 400:
        tries += 1
        spread = 0.1 * (1 + tries // 50)  # widen the search if nearby points are unstable
        c = prior_mean + spread * rng.standard_normal(p)
        if _stable(col, c):
            candidates.append(c)
    seconds = len(y) / FS
    stages = int(max(0, np.ceil(np.log(seconds / 15.0) / np.log(4.0))))
    weights = [4.0 ** -k for k in range(stages, -1, -1)]
    best = None
    for start in candidates:
        theta, total_it = start.copy(), 0
        for w in weights:
            theta, value, done, it = scoring(theta, w)
            total_it += it
        res = FitResult(theta, value / count, done, total_it)
        if best is None or res.fun < best.fun:
            best = res
    return best, len(candidates), objective


def realised_loss(col, theta_true, theta_fit):
    at, bt, ht = col.matrices(theta_true)
    an, bn, hn = col.matrices(theta_fit)
    bt = bt / np.sqrt(ht @ V.stationary_covariance(at, bt) @ ht)
    bn = bn / np.sqrt(hn @ V.stationary_covariance(an, bn) @ hn)
    return V.high_snr_floor(at, bt, an, bn, ht, h_nom=hn)


def one_replicate(task):
    family, seconds, rep, seed = task
    col = column_for(family)
    theta0 = np.asarray(col.log_theta, dtype=float)
    rng = np.random.default_rng(seed)
    a, b, h = normalised_minimal(col, theta0)
    # the drive gain is a free parameter; data are generated with unit output variance
    theta_true = theta0.copy()
    theta_true[-1] += np.log(np.linalg.norm(b) / np.linalg.norm(minimal_column(col, theta0)[1]))
    y = simulate(a, b, h, seconds, rng)
    prior_mean = theta_true + PRIOR_SD * rng.standard_normal(len(theta0))
    t = time.time()
    res, n_starts, objective = fit(col, y, prior_mean, rng)
    loss = realised_loss(col, theta_true, res.x)
    # diagnostic only: the objective at the true parameters (never used by the estimator);
    # a fit whose objective exceeds it has not reached the posterior mode
    obj_true = float(objective(theta_true))
    return {"family": family, "seconds": seconds, "rep": rep, "converged": bool(res.success),
            "starts": n_starts, "stable_fit": bool(_stable(col, res.x)),
            "objective": float(res.fun), "objective_at_truth": obj_true,
            "reached_mode": bool(res.fun <= obj_true + 1e-9),
            "fit_seconds": time.time() - t, "loss": loss.tolist(),
            "theta_error": (res.x - theta_true).tolist(), "prior_error": (prior_mean - theta_true).tolist()}


def forecast(family, seconds_list, draws, seed):
    from neural_observability.theory.linear_families import verification_analysis

    col = column_for(family)
    theta0 = np.asarray(col.log_theta, dtype=float)
    a, b, h = col.matrices(theta0)
    scale = np.sqrt(h @ V.stationary_covariance(a, b) @ h)
    theta_true = theta0.copy()
    theta_true[-1] -= np.log(scale)
    col_true = type(col)(col.name, col.labels, theta_true, col.state_labels, col.psp_states)
    fisher = verification_analysis(col_true, snr_db=SNR_DB)["fisher_per_second"]
    rng = np.random.default_rng(seed)
    out = {}
    for seconds in seconds_list:
        cov = np.linalg.inv(np.eye(len(theta0)) / PRIOR_SD**2 + seconds * fisher)
        chol = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-15 * np.eye(len(cov)))
        fitted_side, truth_side = [], []
        for _ in range(draws):
            delta = chol @ rng.standard_normal(len(theta0))
            try:
                fitted_side.append(realised_loss(col, theta_true, theta_true + delta))
                truth_side.append(realised_loss(col, theta_true + delta, theta_true))
            except (ValueError, np.linalg.LinAlgError):
                continue
        out[f"{seconds:g}"] = {"estimate_error_convention": np.mean(fitted_side, 0).tolist(),
                               "atlas_convention": np.mean(truth_side, 0).tolist(), "draws": len(fitted_side)}
    return out, list(col.state_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(POINTS))
    parser.add_argument("--seconds", default="60,600,3600")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--out", default="results/simulations/finite_record_fits")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seconds = [float(s) for s in args.seconds.split(",")]
    families = args.families.split(",")
    tasks = [(f, s, r, args.seed + 100000 * i + 1000 * j + r)
             for i, f in enumerate(families) for j, s in enumerate(seconds) for r in range(args.reps)]
    with ProcessPoolExecutor(args.workers) as pool:
        reps = list(pool.map(one_replicate, tasks, chunksize=1))
    result = {"config": vars(args), "families": {}}
    for i, f in enumerate(families):
        fc, labels = forecast(f, seconds, 256, args.seed + i)
        rows = [r for r in reps if r["family"] == f]
        summary = {}
        for s in seconds:
            sub = [r for r in rows if r["seconds"] == s]
            losses = np.array([r["loss"] for r in sub])
            mode = np.array([r["reached_mode"] for r in sub])
            summary[f"{s:g}"] = {"realised_mean": losses.mean(0).tolist(),
                                 "reached_mode": int(mode.sum()),
                                 "realised_mean_mode_only": losses[mode].mean(0).tolist() if mode.any() else None,
                                 "realised_se_mode_only": (losses[mode].std(0, ddof=1) / np.sqrt(mode.sum())).tolist()
                                 if mode.sum() > 1 else None,
                                 "realised_median": np.median(losses, 0).tolist(),
                                 "realised_se": (losses.std(0, ddof=1) / np.sqrt(len(losses))).tolist(),
                                 "converged": int(sum(r["converged"] for r in sub)), "n": len(sub)}
        result["families"][f] = {"state_labels": labels, "forecast": fc, "realised": summary, "replicates": rows}
        print(f, json.dumps({k: (np.round(v["realised_mean"], 3).tolist(),
                                 np.round(fc[k]["estimate_error_convention"], 3).tolist()) for k, v in summary.items()}),
              flush=True)
    (out / "verification_fit.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
