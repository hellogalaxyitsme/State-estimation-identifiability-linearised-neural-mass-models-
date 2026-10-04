"""Summary statistics of the nonlinear filtering study (writes results/simulations/nonlinear_filtering/pf_summary.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PF = ROOT / "results" / "simulations" / "nonlinear_filtering"
REGIMES = [("jansen_rit_focus", 2), ("jansen_rit_limit_cycle", 2), ("wendling_2002", 3),
           ("wendling_background", 2), ("next_generation_gamma", 5)]


def med(rows, key, part, metric, idx):
    vals = [r[key][part][metric][idx] for r in rows]
    return float(np.median(vals)) if vals else float("nan")


def regime_summary(rows, idx):
    ex0 = [r for r in rows if r["mismatch_level"] == 0 and r["snr_db"] == 0.0]
    mm = [r for r in rows if abs(r["mismatch_level"] - 0.10) < 1e-9 and r["snr_db"] == 40.0]
    mm_ekf = [r for r in mm if np.isfinite(r["ekf"]["all"]["nmse_total"][idx])]
    lin_ex = ex0[0]["linear_prediction"]["nmse"][idx] if ex0 and ex0[0]["linear_prediction"] else float("nan")
    lin_mm = [r["linear_prediction"]["nmse"][idx] for r in mm if r["linear_prediction"]]
    mm_all = [r for r in rows if r["mismatch_level"] > 0]
    ratio = [r["pf"]["all"]["nmse_fluctuation"][idx] / max(r["ekf"]["all"]["nmse_fluctuation"][idx], 1e-12)
             for r in mm_all if np.isfinite(r["ekf"]["all"]["nmse_fluctuation"][idx])]
    return {
        "exact_0dB": {"pf": ex0[0]["pf"]["all"]["nmse_total"][idx], "ekf": ex0[0]["ekf"]["all"]["nmse_total"][idx],
                      "linear": lin_ex},
        "exact_by_snr": {f"{r['snr_db']:g}": [r["pf"]["all"]["nmse_total"][idx], r["ekf"]["all"]["nmse_total"][idx],
                                              r["linear_prediction"]["nmse"][idx] if r["linear_prediction"] else None]
                         for r in sorted([r for r in rows if r["mismatch_level"] == 0], key=lambda r: r["snr_db"])},
        "mm10_40dB": {"total": med(mm, "pf", "all", "nmse_total", idx), "bias": med(mm, "pf", "all", "nmse_bias", idx),
                      "fluct": med(mm, "pf", "all", "nmse_fluctuation", idx),
                      "ekf_total": med(mm_ekf, "ekf", "all", "nmse_total", idx),
                      "ekf_bias": med(mm_ekf, "ekf", "all", "nmse_bias", idx),
                      "ekf_fluct": med(mm_ekf, "ekf", "all", "nmse_fluctuation", idx),
                      "linear": float(np.median(lin_mm)) if lin_mm else float("nan"),
                      "output_r2": float(np.median([r["pf"]["output_r2"] for r in mm])),
                      "ekf_output_r2": float(np.median([r["ekf"]["output_r2"] for r in mm_ekf])) if mm_ekf
                      else float("nan"),
                      "ekf_diverged": len(mm) - len(mm_ekf),
                      "ess_p05": float(np.median([r["ess_p05"] for r in mm])),
                      "n": len(mm)},
        "fluct_by_snr": {f"{v:g}": {lev: med([r for r in rows if abs(r["mismatch_level"] - lev) < 1e-9
                                              and r["snr_db"] == v], "pf", "all", "nmse_fluctuation", idx)
                                    for lev in (0.05, 0.10)}
                         for v in sorted({r["snr_db"] for r in rows})},
        "halves_40dB_10pct": [med(mm, "pf", "first_half", "nmse_fluctuation", idx),
                              med(mm, "pf", "second_half", "nmse_fluctuation", idx)],
        "unstable_truths": sorted({(r["mismatch_level"], r["draw"]) for r in rows
                                   if r["mismatch_level"] > 0 and not r["linear_prediction"]}),
        "degenerate_rows": [(r["mismatch_level"], r["draw"], r["snr_db"], r["ess_p05"], r["pf"]["output_r2"],
                             r["ekf"]["output_r2"]) for r in rows if r["ess_p05"] < 100],
        "ekf_diverged_rows": [(r["mismatch_level"], r["draw"], r["snr_db"]) for r in rows
                              if not np.isfinite(r["ekf"]["all"]["nmse_total"][idx])],
        "pf_vs_ekf_fluct_ratio_mm": [float(x) for x in ratio],
        "pf_vs_ekf_within_1p5": [int(sum(1 / 1.5 < x < 1.5 for x in ratio)), len(ratio), len(mm_all)],
    }


def convergence_summary(conv, regime, idx):
    rows = [r for r in conv if r["regime"] == regime]
    out = {}
    if not rows:
        return out
    mismatched = [x["draw"] for x in rows if x["mismatch_level"] > 0]
    conditions = (("exact", lambda r: r["mismatch_level"] == 0),
                  ("mismatch 1", lambda r: r["mismatch_level"] > 0 and r["draw"] == min(mismatched)),
                  ("mismatch 2", lambda r: r["mismatch_level"] > 0 and r["draw"] == max(mismatched)))
    for cond, sel in conditions:
        for snr in (10.0, 40.0):
            stats = {p: [r["pf"]["all"]["nmse_fluctuation"][idx] for r in rows if sel(r) and r["snr_db"] == snr
                         and r["particles"] == p and r["truth_dt"] == 5e-5 and r["dt"] == 1e-4]
                     for p in (1000, 10000, 100000)}
            fine = [r["pf"]["all"]["nmse_fluctuation"][idx] for r in rows if sel(r) and r["snr_db"] == snr
                    and r["dt"] == 5e-5]
            ekf = [r["ekf"]["all"]["nmse_fluctuation"][idx] for r in rows if sel(r) and r["snr_db"] == snr and "ekf" in r]
            ess = [r["ess_p05"] for r in rows if sel(r) and r["snr_db"] == snr and r["particles"] == 100000]
            out[f"{regime}|{cond}|{snr:g}"] = {str(k): v for k, v in stats.items()} | {
                "fine_dt": fine, "ekf": ekf, "ess_p05_1e5": ess}
    return out


def main():
    rows = json.loads((PF / "pf_results.json").read_text())["results"]
    conv_path = PF / "pf_convergence.json"
    conv = json.loads(conv_path.read_text())["results"] if conv_path.exists() else []
    summary = {}
    for regime, idx in REGIMES:
        regime_rows = [r for r in rows if r["regime"] == regime]
        if regime_rows:
            summary[regime] = regime_summary(regime_rows, idx)
    if conv:
        summary["convergence"] = {}
        for regime, idx in REGIMES:
            summary["convergence"].update(convergence_summary(conv, regime, idx))
    (PF / "pf_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {PF / 'pf_summary.json'}")


if __name__ == "__main__":
    main()
