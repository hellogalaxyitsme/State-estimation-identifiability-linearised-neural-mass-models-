"""Nonlinear filtering of the full stochastic models.

Main mode: for each regime, the true model is either the filter's own model or
a perturbed one (every listed parameter changed by +-5% or +-10%, random
signs, eight draws per level, each truth reused at every SNR). Each condition
is filtered by a bootstrap particle filter and by an extended Kalman filter.
Errors are reported as normalised mean squared errors split into a mean-level
(bias) part and a fluctuation part, alongside the linear-theory predictions.

Convergence mode: for a subset of conditions, repeats the particle filter
with 10^3, 10^4 and 10^5 particles and three seeds, with a finer integration
step, and reports the same metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from neural_observability.theory import visibility as V
from neural_observability.theory.atlas import wendling_2002
from neural_observability.theory.families import JansenRit, NextGenerationEI, Wendling
from neural_observability.theory.nonlinear_filtering import (
    TorchFamily,
    error_decomposition,
    extended_kalman_filter,
    particle_filter,
    simulate_truth,
)

WC_FIELDS = ("gain_e", "gain_s", "gain_f", "tau_a", "tau_b", "tau_g", "c1", "c2", "c3", "c4", "c5", "c6", "c7")
JR_FIELDS = ("gain_e", "gain_i", "a", "b", "c1", "c2", "c3", "c4")
REGIMES = {
    "jansen_rit_focus": {"factory": JansenRit, "drive": 90.0, "branch": 0, "noise_sd": 2.0, "fields": JR_FIELDS},
    "jansen_rit_limit_cycle": {"factory": JansenRit, "drive": 200.0, "branch": 0, "noise_sd": 2.0, "fields": JR_FIELDS},
    "wendling_background": {"factory": Wendling, "drive": 90.0, "branch": 0, "noise_sd": 1.5, "fields": WC_FIELDS},
    "wendling_2002": {"factory": wendling_2002, "drive": 90.0, "branch": 0, "noise_sd": 2.0, "fields": WC_FIELDS},
    "next_generation_gamma": {"factory": NextGenerationEI, "drive": -0.65, "branch": 0, "noise_sd": 0.03,
                              "fields": ("tau_e", "tau_i", "tau_se", "tau_si", "delta_e", "delta_i",
                                         "j_ee", "j_ei", "j_ie", "j_ii")},
}


def _normalised(model, x_star):
    a = model.jacobian(x_star)
    if np.linalg.eigvals(a).real.max() >= 0:
        return None
    b, h = model.input_vector(), model.output_vector()
    p = V.stationary_covariance(a, b)
    return a, b / np.sqrt(h @ p @ h), h


def linear_predictions(model, x_star, snr_db, truth=None, truth_x=None):
    nom = _normalised(model, x_star)
    if nom is None:
        return None
    a, b, h = nom
    r = 1.0 / (200.0 * 10 ** (snr_db / 10.0))
    if truth is None:
        rec = V.stationary_recoverability(a, b, h, r)
        return {"nmse": (1 - rec.physical_r2("filtered")).tolist(), "output_r2": V.output_r2(rec, h, "filtered")}
    tr = _normalised(truth, truth_x)
    if tr is None:
        return None
    at, bt, ht = tr
    res = V.mismatched_filter_error(at, bt, a, b, ht, r, h_nom=h)
    floor = V.high_snr_floor(at, bt, a, b, ht, h_nom=h)
    return {"nmse": (1 - res["r2"]).tolist(), "output_r2": res["output_r2"], "high_snr_floor": floor.tolist()}


def draw_truths(spec, nominal, x_nom, drive, levels, draws, rng):
    truths = []
    for level in levels:
        for draw in range(draws):
            signs = rng.choice([-1.0, 1.0], size=len(spec["fields"]))
            truth = replace(nominal, **{f: getattr(nominal, f) * (1 + level * s) for f, s in zip(spec["fields"], signs)})
            eqs = truth.equilibria(drive)
            if not eqs:
                truths.append({"level": level, "draw": draw, "excluded": "no fixed point"})
                continue
            xt = min(eqs, key=lambda e: np.max(np.abs(e - x_nom) / (np.abs(x_nom) + 1e-9)))
            truths.append({"level": level, "draw": draw, "truth": truth, "x_truth": xt})
    return truths


def run_conditions(name, spec, conditions, nominal, x_nom, drive, args, device, seed, particles,
                   dt, truth_dt, with_ekf):
    runs = len(conditions)
    gen = torch.Generator(device=device).manual_seed(seed)
    truth_family = TorchFamily([c["truth"] for c in conditions], [drive] * runs, device)
    filter_family = TorchFamily([nominal] * runs, [drive] * runs, device)
    noise = torch.full((runs, 1), spec["noise_sd"], dtype=torch.float64, device=device)
    x0 = torch.tensor(np.array([c["x_truth"] for c in conditions]), dtype=torch.float64, device=device)
    fs = 1000.0
    burn = int(args.burn * fs)
    total = burn + int(args.seconds * fs)
    states = simulate_truth(truth_family, x0, noise, truth_dt, int(round(1e-3 / truth_dt)), total, gen)
    y_clean = (states * truth_family.readout[:, None, :]).sum(-1)
    var_y = y_clean[:, burn:].var(dim=1, keepdim=True)
    snr_lin = torch.tensor([[10 ** (c["snr_db"] / 10.0)] for c in conditions], dtype=torch.float64, device=device)
    obs_sd = torch.sqrt(5.0 * var_y / snr_lin)  # white noise to 500 Hz; SNR defined in a 100 Hz band
    y = y_clean + obs_sd * torch.randn(y_clean.shape, dtype=torch.float64, device=device, generator=gen)
    x_init = torch.tensor(np.array([x_nom] * runs), dtype=torch.float64, device=device)
    substeps = int(round(1e-3 / dt))
    t0 = time.time()
    means, ess = particle_filter(filter_family, y, obs_sd, x_init, noise, dt, substeps,
                                 particles, warmup_steps=int(0.5 / dt), gen=gen)
    pf_seconds = time.time() - t0
    estimates = {"pf": means}
    if with_ekf:
        estimates["ekf"] = extended_kalman_filter(filter_family, y, obs_sd, x_init, noise, dt, substeps,
                                                  warmup_steps=int(0.5 / dt))
    score = slice(burn, total)
    half = burn + (total - burn) // 2
    truth_s = states[:, score]
    y_true = y_clean[:, score]
    out = []
    metrics = {}
    for key, est in estimates.items():
        e = est[:, score]
        metrics[key] = {
            "all": error_decomposition(e, truth_s),
            "first_half": error_decomposition(est[:, burn:half], states[:, burn:half]),
            "second_half": error_decomposition(est[:, half:total], states[:, half:total]),
            "output_r2": (1 - (((e * filter_family.readout[:, None, :]).sum(-1) - y_true) ** 2).mean(1)
                          / y_true.var(1)).cpu().numpy(),
        }
    ess_np = ess[:, score].cpu().numpy()
    for i, c in enumerate(conditions):
        row = {"regime": name, "family": nominal.name, "snr_db": c["snr_db"], "mismatch_level": c["level"],
               "draw": c.get("draw", 0), "rep": c.get("rep", 0), "particles": particles, "dt": dt, "truth_dt": truth_dt, "seed": seed,
               "state_labels": list(nominal.state_labels), "output_variance": float(var_y[i, 0]),
               "ess_median": float(np.median(ess_np[i])), "ess_p05": float(np.percentile(ess_np[i], 5)),
               "ess_min": float(np.min(ess_np[i])), "pf_seconds_batch": pf_seconds,
               "state_variance": truth_s[i].var(0, unbiased=False).cpu().numpy().tolist()}
        for key, m in metrics.items():
            row[key] = {part: {k: v[i].tolist() for k, v in m[part].items()} for part in ("all", "first_half", "second_half")}
            row[key]["output_r2"] = float(m["output_r2"][i])
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "convergence"), default="main")
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--snr", default="0,10,20,30,40")
    parser.add_argument("--levels", default="0.05,0.10")
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--burn", type=float, default=2.0)
    parser.add_argument("--particles", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--out", default="results/simulations/nonlinear_filtering")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snrs = [float(s) for s in args.snr.split(",")]
    levels = [float(s) for s in args.levels.split(",")]
    results, excluded = [], []
    fname = "pf_results.json" if args.mode == "main" else "pf_convergence.json"
    for r_index, name in enumerate(args.regimes.split(",")):
        spec = REGIMES[name]
        start = time.time()
        nominal = spec["factory"]()
        drive = spec["drive"]
        x_nom = nominal.equilibria(drive)[spec["branch"]]
        rng = np.random.default_rng(args.seed + r_index)
        truths = draw_truths(spec, nominal, x_nom, drive, levels, args.draws, rng)
        excluded += [{"regime": name, **{k: v for k, v in t.items() if k in ("level", "draw", "excluded")}}
                     for t in truths if "excluded" in t]
        truths = [t for t in truths if "excluded" not in t]
        if args.mode == "main":
            conditions = []
            for snr in snrs:
                conditions.append({"snr_db": snr, "level": 0.0, "truth": nominal, "x_truth": x_nom})
                conditions += [{"snr_db": snr, **t} for t in truths]
            rows = run_conditions(name, spec, conditions, nominal, x_nom, drive, args, device,
                                  args.seed + 17 * r_index, args.particles, 1e-4, 1e-4, with_ekf=True)
        else:
            subset = [t for t in truths if abs(t["level"] - 0.10) < 1e-9][:2]
            conditions = []
            # three replicates of every condition in one batch: each row has its own
            # simulated data and its own particle cloud
            for rep in range(3):
                for snr in (10.0, 40.0):
                    conditions.append({"snr_db": snr, "level": 0.0, "truth": nominal, "x_truth": x_nom, "rep": rep})
                    conditions += [{"snr_db": snr, **t, "rep": rep} for t in subset]
            rows = []
            for particles in (1000, 10000, 100000):
                # the same seed for every particle count, so all counts see the same data
                rows += run_conditions(name, spec, conditions, nominal, x_nom, drive, args, device,
                                       args.seed + 17 * r_index, particles, 1e-4, 5e-5,
                                       with_ekf=(particles == 1000))
            rows += run_conditions(name, spec, conditions, nominal, x_nom, drive, args, device,
                                   args.seed + 17 * r_index, 10000, 5e-5, 5e-5, with_ekf=False)
        for row in rows:
            lin = (linear_predictions(nominal, x_nom, row["snr_db"]) if row["mismatch_level"] == 0.0 else None)
            if row["mismatch_level"] != 0.0:
                t = next(t for t in truths if t["level"] == row["mismatch_level"] and t["draw"] == row["draw"])
                lin = linear_predictions(nominal, x_nom, row["snr_db"], t["truth"], t["x_truth"])
            row["linear_prediction"] = lin
        results += rows
        with open(out / fname, "w") as handle:
            json.dump({"config": vars(args), "excluded_truths": excluded, "results": results}, handle)
        print(f"{name}: {len(rows)} rows, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
