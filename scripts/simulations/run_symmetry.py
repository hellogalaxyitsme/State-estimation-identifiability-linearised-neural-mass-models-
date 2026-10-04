"""Exact symmetries, minimality and spectral equivalence of the three families.

For each family, at the default operating point and at jittered atlas points:

* minimality of the original and reduced realisations,
* exact (rational) rank of the parameter-to-transfer-function map,
* the named one-parameter scaling groups and any further null directions,
  with their infinitesimal similarities and the states they change,
* realisability of every null direction by physiological parameters,
* a multistart search for exactly and nearly equivalent parameter values.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neural_observability.theory import symmetry as S
from neural_observability.theory import visibility as V
from neural_observability.theory.atlas import sample_operating_points, wendling_2002
from neural_observability.theory.families import JansenRit, NextGenerationEI, Wendling
from neural_observability.theory.linear_families import linear_column

DEFAULTS = {"jansen_rit": (JansenRit, 90.0), "wendling": (Wendling, 90.0), "wendling_2002": (wendling_2002, 90.0),
            "next_generation": (NextGenerationEI, -0.65)}


def stationary_correlation_floor(a, b):
    p = V.stationary_covariance(a, b)
    d = np.sqrt(np.diag(p))
    return float(np.linalg.eigvalsh(p / np.outer(d, d)).min())


def analyse(family: str, model, drive: float, x_star, starts: int, seed: int, exact_null: int) -> dict:
    col = linear_column(model, x_star)
    a, b, h = col.matrices()
    am, bm, hm, embed = S.minimal_column(col)
    p = V.stationary_covariance(a, b)
    out = {
        "output_share": {lab: float(h[j] ** 2 * p[j, j] / (h @ p @ h))
                         for j, lab in enumerate(model.state_labels) if h[j] != 0},
        "family": family, "drive": drive, "parameters": list(col.labels), "log_theta": col.log_theta.tolist(),
        "state_labels": list(model.state_labels),
        "full_dimension": len(a), "minimal_dimension": len(am),
        "full_correlation_floor": stationary_correlation_floor(a, b),
        "minimal_correlation_floor": stationary_correlation_floor(am, bm),
        "minimal_pbh": S.pbh_margins(am, bm, hm),
    }
    jac = S.response_jacobian(col)
    sv = np.linalg.svd(jac, compute_uv=False)
    out["response_singular_values"] = sv.tolist()
    names, gens = S.named_generators(col)
    out["named_generators"] = {}
    for name, g in zip(names, gens):
        x, resid, recon = S.lie_generator(col, g)
        out["named_generators"][name] = {
            "direction": dict(zip(col.labels, g.tolist())),
            "response_change": float(np.linalg.norm(jac @ g) / (np.linalg.norm(jac, 2) * np.linalg.norm(g))),
            "similarity_diagonal": np.diag(x).tolist(),
            "similarity_offdiagonal_max": float(np.abs(x - np.diag(np.diag(x))).max()),
            "lie_residual": resid,
            "reconstruction_change": recon.tolist(),
        }
    # further null directions: exact null space at this operating point
    # (rational arithmetic), with the named generators projected out
    null = S.exact_nullspace_at(col)
    q, _ = np.linalg.qr(gens.T)
    rest = null - (null @ q) @ q.T
    u, s_rest, _ = np.linalg.svd(rest.T, full_matrices=False)
    extra = u[:, s_rest > 1e-8].T
    out["null_dimension"] = int(len(null))
    out["named_in_exact_null"] = float(np.linalg.norm(gens.T - null.T @ np.linalg.lstsq(null.T, gens.T, rcond=None)[0]))
    out["extra_generators"] = []
    for g in extra:
        g = g / np.abs(g).max()
        x, resid, recon = S.lie_generator(col, g)
        out["extra_generators"].append({
            "direction": dict(zip(col.labels, g.tolist())),
            "similarity_diagonal": np.diag(x).tolist(),
            "similarity_offdiagonal_max": float(np.abs(x - np.diag(np.diag(x))).max()),
            "similarity": x.tolist(),
            "lie_residual": resid,
            "reconstruction_change": recon.tolist(),
        })
    try:
        jp, phys_names = S.physiological_jacobian(model, drive, x_star, linear_column)
        all_null = np.vstack([gens, extra]) if len(extra) else gens
        rr = S.realisability(jp, all_null)
        out["physiological"] = {"rank": rr.rank, "linear_parameters": rr.linear_parameters,
                                "generator_residuals": rr.generator_residuals, "parameters": phys_names}
    except RuntimeError as exc:
        out["physiological"] = {"error": str(exc)}
    if starts:
        srch = S.equivalence_search(col, gens, starts=starts, seed=seed)
        srch.pop("residuals")
        out["equivalence_search"] = srch
    return out


def worker(task):
    family, kind, index, seed, starts, exact_null = task
    factory, drive = DEFAULTS[family]
    if kind == "default":
        model = factory()
        x = model.equilibria(drive)[0]
    else:
        pts = sample_operating_points(family, index + 1, seed=seed, jitter_sd=0.15, min_damping=0.5)
        model, drive, x = pts[index]["model"], pts[index]["drive"], pts[index]["x_star"]
    t = time.time()
    res = analyse(family, model, drive, x, starts, seed=seed + index, exact_null=exact_null)
    res["kind"], res["index"], res["seconds"] = kind, index, time.time() - t
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/simulations/symmetry")
    parser.add_argument("--points", type=int, default=6)
    parser.add_argument("--starts-default", type=int, default=400)
    parser.add_argument("--starts-points", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exact = {}
    for family, (factory, drive) in DEFAULTS.items():
        model = factory()
        col = linear_column(model, model.equilibria(drive)[0])
        t = time.time()
        res = S.exact_symmetry_rank(col, points=3, seed=args.seed)
        exact[family] = {"parameters": res["parameters"], "ranks": res["ranks"],
                         "coefficients": res["coefficients"], "minimal_dimension": res["state_dimension"],
                         "null_dimension": [len(n) for n in res["nullspaces"]], "seconds": time.time() - t}
        print(family, exact[family], flush=True)
    nulls = {f: len(exact[f]["parameters"]) - min(exact[f]["ranks"]) for f in DEFAULTS}
    tasks = [(f, "default", 0, args.seed, args.starts_default, nulls[f]) for f in DEFAULTS]
    tasks += [(f, "atlas", i, args.seed + 101, args.starts_points, nulls[f]) for f in DEFAULTS for i in range(args.points)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(worker, tasks))
    for r in results:
        srch = r.get("equivalence_search", {})
        print(r["family"], r["kind"], r["index"], "null", r["null_dimension"], "phys rank",
              r["physiological"].get("rank"), "exact eq", srch.get("exact_equivalents"),
              "near eq", srch.get("near_equivalents"), f"{r['seconds']:.0f}s", flush=True)
    (out / "symmetry_results.json").write_text(json.dumps({"config": vars(args), "exact_rank": exact,
                                                            "points": results}, indent=1))


if __name__ == "__main__":
    main()
