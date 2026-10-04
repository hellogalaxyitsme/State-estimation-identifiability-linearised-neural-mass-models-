"""Predicted versus observed visibility of known intracerebral dipoles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mne
from neural_observability.head_models.referencing import average_reference_basis
from neural_observability.known_sources.localize_mi import (
    parse_session,
    read_tsv,
    session_field,
    squared_cosine,
    whitened_energy,
    write_freesurfer_bem,
)

def nearest_distance(points: np.ndarray, target: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(points - target[None, :], axis=1)))


def analyse_subject(root: Path, subject: str, subjects_dir: Path, skull: float):
    der = root / "derivatives"
    eeg_dir = der / "epochs" / subject / "eeg"
    electrodes = read_tsv(eeg_dir / f"{subject}_task-seegstim_electrodes.tsv")
    names = [row["name"] for row in electrodes]
    positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in electrodes])
    contacts = {
        row["name"]: np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        for row in read_tsv(der / "epochs" / subject / "ieeg" / f"{subject}_task-seegstim_space-T1w_electrodes.tsv")
    }
    fwd_template = mne.read_forward_solution(der / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif", verbose=False)
    trans = fwd_template["mri_head_t"]
    write_freesurfer_bem(der / "sourcemodelling" / subject / "anat", subject, subjects_dir)
    model = mne.make_bem_model(subject, ico=4, conductivity=(0.3, skull, 0.3), subjects_dir=subjects_dir, verbose=False)
    bem = mne.make_bem_solution(model, verbose=False)
    scalp = model[-1]["rr"] if model[-1]["id"] == 4 else model[0]["rr"]
    scalp_head = mne.transforms.apply_trans(trans, scalp)
    sessions = []
    for meta_path in sorted(eeg_dir.glob(f"{subject}_task-seegstim_run-*_epochs.json")):
        run = meta_path.name.split("_run-")[1].split("_")[0]
        meta = json.loads(meta_path.read_text())
        first, second, current = parse_session(meta["Description"])
        rows = read_tsv(eeg_dir / f"{subject}_task-seegstim_run-{run}_epochs.tsv")
        zero_time = float(rows[0]["zero_time"])
        channels = read_tsv(eeg_dir / f"{subject}_task-seegstim_run-{run}_channels.tsv")
        if [c["name"] for c in channels] != names:
            raise RuntimeError(f"{subject} run {run}: channel order differs from electrodes table")
        good = np.array([c["status"] == "good" for c in channels])
        sfreq = float(channels[0]["sampling_frequency"])
        epochs = np.load(eeg_dir / f"{subject}_task-seegstim_run-{run}_epochs.npy", mmap_mode="r")
        y_peak, sigma, basis, t_peak, n_trials = session_field(
            np.asarray(epochs), good, sfreq, zero_time, (0.0, 0.005), (-0.25, -0.05)
        )
        p1, p2 = contacts[first], contacts[second]
        spacing = float(np.linalg.norm(p2 - p1))
        orientation = (p2 - p1) / spacing
        midpoint = 0.5 * (p1 + p2)
        info = mne.create_info(names, sfreq, "eeg")
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(names, positions)), coord_frame="head")
        info.set_montage(montage)
        dip = mne.Dipole(np.array([0.0]), midpoint[None, :], np.array([current * spacing]), orientation[None, :], np.array([100.0]))
        fwd, _ = mne.make_forward_dipole(dip, bem, info, trans=trans, verbose=False)
        gain_full = np.asarray(fwd["sol"]["data"])[:, 0]  # V per A m along the contact axis, all electrodes
        predicted = basis.T @ gain_full[good] * current * spacing
        sessions.append({
            "subject": subject,
            "run": run,
            "site": f"{first}-{second}",
            "current_a": current,
            "contact_spacing_m": spacing,
            "dipole_moment_am": current * spacing,
            "n_trials": n_trials,
            "good_channels": int(good.sum()),
            "peak_time_s": t_peak,
            "mu_obs": whitened_energy(y_peak, sigma),
            "mu_pred": whitened_energy(predicted, sigma),
            "depth_below_scalp_m": nearest_distance(scalp_head, midpoint),
            "nearest_electrode_m": nearest_distance(positions[good], midpoint),
            "y_peak": y_peak.tolist(),
            "predicted": predicted.tolist(),
            "gain_full": gain_full.tolist(),
            "good": good.tolist(),
        })
    return sessions


def spearman(x, y):
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def endpoints(sessions: list[dict], seed: int = 20261004) -> dict:
    rng = np.random.default_rng(seed)
    obs = np.log10([s["mu_obs"] for s in sessions])
    pred = np.log10([s["mu_pred"] for s in sessions])
    subjects = np.array([s["subject"] for s in sessions])
    cur = np.log10([s["current_a"] for s in sessions])

    def ols(x, y):
        xm = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(xm, y, rcond=None)
        resid = y - xm @ coef
        return coef, float(np.sqrt(np.mean(resid**2))), float(1 - resid.var() / y.var())

    coef, rmse, r2 = ols(pred, obs)
    uniq = sorted(set(subjects))
    loso = [ols(pred[subjects != s], obs[subjects != s])[0][1] for s in uniq]
    boot = []
    for _ in range(10000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(subjects == s) for s in pick])
        if np.ptp(pred[idx]) > 0:
            boot.append(ols(pred[idx], obs[idx])[0][1])
    # intensity law within site
    pairs = []
    by_site: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(sessions):
        by_site.setdefault((s["subject"], s["site"]), []).append(i)
    for idxs in by_site.values():
        if len(idxs) >= 2:
            lo, hi = sorted(idxs, key=lambda i: sessions[i]["current_a"])[:: len(idxs) - 1]
            if sessions[hi]["current_a"] > sessions[lo]["current_a"]:
                pairs.append((cur[hi] - cur[lo], obs[hi] - obs[lo]))
    pairs = np.array(pairs)
    slope_intensity = float(np.sum(pairs[:, 0] * pairs[:, 1]) / np.sum(pairs[:, 0] ** 2)) if len(pairs) else float("nan")
    # geometry beyond current
    geo_obs = obs - 2 * cur
    geo_pred = pred - 2 * cur
    depth = np.array([s["depth_below_scalp_m"] for s in sessions])
    near = np.array([s["nearest_electrode_m"] for s in sessions])
    within = [spearman(geo_pred[subjects == s], geo_obs[subjects == s]) for s in uniq if np.sum(subjects == s) >= 4]
    # known-site topography
    effects = []
    for i, s in enumerate(sessions):
        others = [j for j, t in enumerate(sessions) if t["subject"] == s["subject"] and t["site"] != s["site"]]
        if not others:
            continue
        y = np.array(s["y_peak"])
        good_i = np.array(s["good"])
        basis_i = average_reference_basis(int(good_i.sum()))
        own = squared_cosine(y, np.array(s["predicted"]))
        wrong = np.mean([
            squared_cosine(y, basis_i.T @ np.array(sessions[j]["gain_full"])[good_i]) for j in others
        ])
        effects.append((s["subject"], own - wrong, own))
    eff_subject = np.array([np.mean([e[1] for e in effects if e[0] == s]) for s in uniq])
    flips = np.array([[1 if (k >> b) & 1 else -1 for b in range(len(uniq))] for k in range(2 ** len(uniq))])
    p_sign = float(np.mean(flips @ eff_subject >= eff_subject.sum() - 1e-15))
    return {
        "n_sessions": len(sessions),
        "n_subjects": len(uniq),
        "calibration": {
            "intercept": float(coef[0]), "slope": float(coef[1]), "rmse_log10": rmse, "r2": r2,
            "slope_leave_one_subject_out_range": [float(min(loso)), float(max(loso))],
            "slope_cluster_bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "log10_mu_obs_range": [float(obs.min()), float(obs.max())],
        },
        "intensity_law": {"pairs": int(len(pairs)), "slope": slope_intensity, "expected": 2.0},
        "geometry_beyond_current": {
            "spearman_pooled_model": spearman(geo_pred, geo_obs),
            "spearman_pooled_depth": spearman(-depth, geo_obs),
            "spearman_pooled_inverse_square_nearest": spearman(-np.log10(near), geo_obs),
            "spearman_within_subject_mean": float(np.mean(within)),
            "within_subject_values": within,
        },
        "known_site_topography": {
            "sessions": len(effects),
            "mean_effect": float(np.mean([e[1] for e in effects])),
            "positive_sessions": int(np.sum([e[1] > 0 for e in effects])),
            "mean_own_squared_cosine": float(np.mean([e[2] for e in effects])),
            "subject_mean_effects": eff_subject.tolist(),
            "sign_flip_p_one_sided": p_sign,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--work", default="derived/localize_mi_subjects")
    parser.add_argument("--out", default="results/known_sources/localize_mi")
    parser.add_argument("--skulls", default="0.006,0.003,0.012")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data)
    result = {"by_skull_conductivity": {}}
    for skull in [float(s) for s in args.skulls.split(",")]:
        subjects_dir = Path(args.work) / f"skull_{skull:g}"
        subjects_dir.mkdir(parents=True, exist_ok=True)
        sessions = []
        for subject in sorted(p.name for p in (root / "derivatives" / "epochs").glob("sub-*")):
            sessions.extend(analyse_subject(root, subject, subjects_dir, skull))
            print(f"skull {skull}: {subject} done ({len(sessions)} sessions)", flush=True)
        result["by_skull_conductivity"][f"{skull:g}"] = {"endpoints": endpoints(sessions), "sessions": sessions}
        (out / "localize_mi_results.json").write_text(json.dumps(result))
        print(json.dumps(result["by_skull_conductivity"][f"{skull:g}"]["endpoints"], indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
