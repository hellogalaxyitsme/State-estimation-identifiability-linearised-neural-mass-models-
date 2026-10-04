"""Summary statistics of the operating-point atlas (writes results/simulations/atlas/atlas_summary.json)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "results" / "simulations" / "atlas"
FAMILIES = {"jansen_rit": "Jansen-Rit", "wendling": "Wendling-Chauvel (background)",
            "wendling_2002": "Wendling-Chauvel (active fast inhibition)", "next_generation": "Mean field"}
LABELS = {"jansen_rit": ["x1", "x2", "x3", "dx1", "dx2", "dx3"],
          "wendling": ["y0", "y1", "y2", "y3", "y4", "dy0", "dy1", "dy2", "dy3", "dy4"],
          "next_generation": ["re", "ve", "ri", "vi", "se", "si"]}
LABELS["wendling_2002"] = LABELS["wendling"]


def load(family):
    rows = [json.loads(line) for line in open(ATLAS / f"{family}_records.jsonl")]
    return rows, [r for r in rows if "error" not in r]


def pct(values, ps=(25, 50, 75), axis=0):
    v = np.asarray(values, dtype=float)
    return [np.nanpercentile(v, p, axis=axis).tolist() for p in ps]


def classify_error(text: str) -> str:
    if "infs or NaNs" in text:
        return "linear gain numerically zero"
    if "Schur" in text or "Riccati" in text:
        return "Riccati solver failure"
    return text[:60]


def summarise_family(fam: str, config: dict, inclusion: dict) -> dict:
    rows, good = load(fam)
    n = len(LABELS[fam])
    s: dict = {"inclusion": inclusion.get(fam, {}), "analysed": len(rows), "ok": len(good),
               "errors": dict(Counter(classify_error(r["error"]) for r in rows if "error" in r))}
    s["minimum_phase_fraction"] = float(np.mean([r["minimum_phase"] for r in good]))
    s["max_zero_real_part"] = float(np.nanmax([r["max_zero_real_part"] for r in good]))
    s["minimal_dimension"] = sorted({r["minimal_dimension"] for r in good})
    s["minimal_correlation_floor"] = pct([r["minimal_correlation_floor"] for r in good], (0, 50))
    share = np.array([r["output_share"] for r in good])
    s["output_share"] = {LABELS[fam][j]: pct(share[:, j], (10, 50, 90)) for j in range(n) if np.any(share[:, j] > 0)}
    s["dominant_frequency_hz"] = pct([r["dominant_frequency_hz"] for r in good], (5, 50, 95))
    hsv = np.array([r["hsv_forward_rel"][:3] for r in good]) ** 2
    s["hankel_ratio_k2_k3_median"] = np.median(hsv[:, 1:3], 0).tolist()
    curves = {}
    for key in ("filtered", "smoothed", "future_only"):
        arr = np.array([r["curves"][key] for r in good])
        q25, med, q75 = pct(arr)
        curves[key] = {"median": med, "q25": q25, "q75": q75}
    for key in ("output_filtered", "output_smoothed"):
        q25, med, q75 = pct([r["curves"][key] for r in good])
        curves[key] = {"median": med, "q25": q25, "q75": q75}
    s["curves"] = curves
    s["filtered_high_snr_median"] = {}
    for snr in config["high_snr_db"]:
        vals = [r["filtered_high_snr"][f"{snr:g}"] for r in good if r["filtered_high_snr"][f"{snr:g}"] is not None]
        s["filtered_high_snr_median"][f"{snr:g}"] = {"median": np.median(vals, 0).tolist(), "n": len(vals),
                                                      "p10": np.percentile(vals, 10, 0).tolist()}
    two = {}
    for frac in config["second_drive_fractions"]:
        two[f"{frac:g}"] = {}
        for i, snr in enumerate(config["second_drive_snr_db"]):
            vals = [r["two_drive"][f"{frac:g}"][i] for r in good if r["two_drive"][f"{frac:g}"][i] is not None]
            q25, med, q75 = pct(vals)
            two[f"{frac:g}"][f"{snr:g}"] = {"median": med, "q25": q25, "q75": q75, "n": len(vals)}
    s["two_drive"] = two
    # theorem 3 concerns the assumed model, here the atlas point: floors and forecasts use the points
    # whose minimal realisation has no transfer zeros in the closed right half-plane
    eligible = [r for r in good if r["minimum_phase"]]
    s["eligible_points"] = len(eligible)
    floors, excl = {}, {}
    for level in config["mismatch_levels"]:
        key = f"{level:g}"
        arr = np.array([row for r in eligible for row in r["mismatch_floors"][key]])
        q25, med, q75 = pct(arr)
        floors[key] = {"n": int(len(arr)), "median": med, "q25": q25, "q75": q75,
                       "p90": np.percentile(arr, 90, 0).tolist(), "max": arr.max(0).tolist(),
                       "fraction_over_1": np.mean(arr > 1.0, 0).tolist(),
                       "fraction_over_0.1": np.mean(arr > 0.1, 0).tolist()}
        excl[key] = dict(sum((Counter(r["mismatch_exclusions"][key]) for r in eligible), Counter()))
        tmp = [r["truth_minimum_phase_fraction"][key] for r in eligible if r["truth_minimum_phase_fraction"][key] is not None]
        floors[key]["truth_minimum_phase_fraction"] = float(np.mean(tmp)) if tmp else None
    s["floors"] = floors
    s["mismatch_exclusions"] = excl
    # closed form versus exact mismatched filter, and quadrature
    check = {}
    for snr in config["floor_check_snr_db"]:
        rel, failed = [], 0
        for r in eligible:
            for key, c in r["floor_checks"].items():
                entry = c.get(f"{snr:g}dB", {})
                if "loss" not in entry:
                    failed += 1
                    continue
                f, e = np.array(c["formula_loss"]), np.array(entry["loss"])
                rel.append(np.abs(e - f) / np.maximum(np.abs(f), 1e-3))
        rel = np.array(rel)
        check[f"{snr:g}"] = {"n": int(len(rel)), "failed": failed, "median_rel_diff": np.median(rel, 0).tolist(),
                             "p90_rel_diff": np.percentile(rel, 90, 0).tolist()}
    quad = []
    for r in eligible:
        for c in r["floor_checks"].values():
            f, q16 = np.array(c["formula_loss"]), np.array(c["quadrature"]["wmax_x16"])
            quad.append(np.abs(q16 - f) / np.maximum(np.abs(f), 1e-3))
    s["floor_vs_exact_filter"] = check
    s["quadrature_rel_change_x16"] = {"median": np.median(quad, 0).tolist(), "p90": np.percentile(quad, 90, 0).tolist(),
                                      "max": np.max(quad, 0).tolist(), "n": len(quad)}
    outr2 = [c["60dB"]["output_r2"] for r in eligible for k, c in r["floor_checks"].items()
             if k == "0.2" and "output_r2" in c.get("60dB", {})]
    s["output_r2_20pct_60dB"] = pct(outr2, (5, 50, 95)) if outr2 else None
    ver, rej = {}, Counter()
    for t in config["record_seconds"]:
        key = f"{t:g}"
        vals = [r["verification"]["loss_by_record_seconds"][key]["mean_loss"] for r in eligible
                if r["verification"]["loss_by_record_seconds"][key]["mean_loss"] is not None]
        q25, med, q75 = pct(vals)
        ver[key] = {"median": med, "q25": q25, "q75": q75, "n": len(vals)}
        for r in eligible:
            rej.update(r["verification"]["loss_by_record_seconds"][key]["rejected"])
    s["verification_loss"] = ver
    s["verification_rejected_draws"] = dict(rej)
    s["verification_draws_total"] = len(eligible) * len(config["record_seconds"]) * config["verification_draws"]
    ranks = [(r.get("physiological_rank"), r.get("linear_parameter_count")) for r in good]
    s["physiological_full_rank_fraction"] = float(np.mean([a == b for a, b in ranks if a is not None]))
    s["physiological_rank_errors"] = sum(r.get("physiological_rank") is None for r in good)
    return s


def main():
    config = json.loads((ATLAS / "config.json").read_text())
    inclusion = json.loads((ATLAS / "inclusion.json").read_text())
    summary = {"snr_db": config["snr_db"], "config": config}
    for fam in FAMILIES:
        if (ATLAS / f"{fam}_records.jsonl").exists():
            summary[fam] = summarise_family(fam, config, inclusion)
    (ATLAS / "atlas_summary.json").write_text(json.dumps(summary, indent=1))
    snrs = list(config["snr_db"])
    for fam, s in summary.items():
        if fam in ("snr_db", "config"):
            continue
        lab = LABELS[fam]
        print(f"\n### {fam}: inclusion {s['inclusion']} analysed {s['analysed']} ok {s['ok']} errors {s['errors']}")
        print(f"  min-phase {s['minimum_phase_fraction']:.3f} max zero Re {s['max_zero_real_part']:.1f} minimal dims {s['minimal_dimension']} corr floor {np.round(s['minimal_correlation_floor'], 9)}")
        print("  output share (10/50/90)", {k: np.round(v, 4).tolist() for k, v in s["output_share"].items()})
        print("  freq", np.round(s["dominant_frequency_hz"], 1), "hankel k2,k3", np.round(s["hankel_ratio_k2_k3_median"], 4))
        for key in ("filtered", "future_only"):
            med = np.array(s["curves"][key]["median"])
            for snr in (0, 20, 40):
                print(f"  {key:11s} {snr:3d}dB", dict(zip(lab, np.round(med[snrs.index(snr)], 2))))
        print("  output filtered 0/20/40", np.round(np.array(s["curves"]["output_filtered"]["median"])[[snrs.index(0), snrs.index(20), snrs.index(40)]], 3))
        for snr, v in s["filtered_high_snr_median"].items():
            print(f"  filtered {snr}dB median", dict(zip(lab, np.round(v["median"], 3))), "p10", np.round(v["p10"], 3).tolist(), "n", v["n"])
        for frac, d in s["two_drive"].items():
            print(f"  2nd drive {frac}: 80dB median", dict(zip(lab, np.round(d["80"]["median"], 3))), "n", d["80"]["n"])
        for level, f in s["floors"].items():
            print(f"  floor {level} n={f['n']} median", dict(zip(lab, np.round(f["median"], 3))))
            print("        q75", dict(zip(lab, np.round(f["q75"], 3))), ">1", dict(zip(lab, np.round(f["fraction_over_1"], 2))))
            print(f"        truth min-phase {f['truth_minimum_phase_fraction']} excl {s['mismatch_exclusions'][level]}")
        for snr, c in s["floor_vs_exact_filter"].items():
            print(f"  exact filter {snr}dB n={c['n']} failed={c['failed']} median rel diff", np.round(c["median_rel_diff"], 4).tolist())
        print("  quadrature x16 rel change median", np.round(s["quadrature_rel_change_x16"]["median"], 4).tolist(), "max", np.round(s["quadrature_rel_change_x16"]["max"], 4).tolist())
        print("  output r2 20% 60dB", s["output_r2_20pct_60dB"])
        for t, v in s["verification_loss"].items():
            print(f"  loss T={t:>6s}", dict(zip(lab, np.round(v["median"], 3))))
        print("  rejected draws", s["verification_rejected_draws"], "of", s["verification_draws_total"],
              "phys full rank", s["physiological_full_rank_fraction"], "rank errors", s["physiological_rank_errors"])


if __name__ == "__main__":
    main()
