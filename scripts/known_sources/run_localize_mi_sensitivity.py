"""Measurement checks for the known-current analysis (Localize-MI).

Adds to the main analysis (run_localize_mi.py), at the nominal skull
conductivity:

* amplitude-clipping check of the single trials in the peak window,
* split-trial peak selection: the latency is chosen on one half of the
  trials and the field is measured on the other half (and vice versa),
* correction of whitened energy for its noise expectation (the dimension d),
* covariances estimated from independent halves for prediction and
  observation, and an unwhitened (Euclidean) visibility,
* participant-cluster bootstrap intervals for the calibration slope and the
  current exponent, residual dispersion and ordinary least-squares reference
  intervals (sessions treated as independent),
* example fields and stimulus-locked field power of a representative session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mne
from neural_observability.known_sources.localize_mi import (
    oas_covariance,
    parse_session,
    read_tsv,
    squared_cosine,
    write_freesurfer_bem,
)
from neural_observability.head_models.referencing import average_reference_basis


def half_field(data, basis, times, peak_mask, base_mask, select_idx, measure_idx):
    """Peak latency from ``select_idx`` trials, field and covariance from ``measure_idx``."""
    sel = basis.T @ data[select_idx].mean(0)
    k = np.flatnonzero(peak_mask)[int(np.argmax(np.sum(sel[:, peak_mask] ** 2, 0)))]
    meas = basis.T @ data[measure_idx].mean(0)
    base = np.einsum("ck,tcs->tsk", basis, data[measure_idx][:, :, base_mask]).reshape(-1, basis.shape[1])
    cov = oas_covariance(base) / len(measure_idx)
    return meas[:, k], cov, float(times[k])


def energy(v, cov):
    return float(v @ np.linalg.solve(cov, v))


def analyse_subject(root: Path, subject: str, subjects_dir: Path, skull: float, rng):
    der = root / "derivatives"
    eeg_dir = der / "epochs" / subject / "eeg"
    electrodes = read_tsv(eeg_dir / f"{subject}_task-seegstim_electrodes.tsv")
    names = [row["name"] for row in electrodes]
    positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in electrodes])
    contacts = {row["name"]: np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                for row in read_tsv(der / "epochs" / subject / "ieeg" / f"{subject}_task-seegstim_space-T1w_electrodes.tsv")}
    fwd_template = mne.read_forward_solution(der / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif", verbose=False)
    trans = fwd_template["mri_head_t"]
    write_freesurfer_bem(der / "sourcemodelling" / subject / "anat", subject, subjects_dir)
    model = mne.make_bem_model(subject, ico=4, conductivity=(0.3, skull, 0.3), subjects_dir=subjects_dir, verbose=False)
    bem = mne.make_bem_solution(model, verbose=False)
    sessions = []
    for meta_path in sorted(eeg_dir.glob(f"{subject}_task-seegstim_run-*_epochs.json")):
        run = meta_path.name.split("_run-")[1].split("_")[0]
        meta = json.loads(meta_path.read_text())
        first, second, current = parse_session(meta["Description"])
        rows = read_tsv(eeg_dir / f"{subject}_task-seegstim_run-{run}_epochs.tsv")
        zero_time = float(rows[0]["zero_time"])
        channels = read_tsv(eeg_dir / f"{subject}_task-seegstim_run-{run}_channels.tsv")
        good = np.array([c["status"] == "good" for c in channels])
        sfreq = float(channels[0]["sampling_frequency"])
        epochs = np.asarray(np.load(eeg_dir / f"{subject}_task-seegstim_run-{run}_epochs.npy", mmap_mode="r"))
        data = epochs[:, good, :]
        n_trials = data.shape[0]
        basis = average_reference_basis(int(good.sum()))
        d = basis.shape[1]
        times = np.arange(data.shape[-1]) / sfreq - zero_time
        base_mask = (times >= -0.25) & (times < -0.05)
        peak_mask = (times >= 0.0) & (times <= 0.005)
        # clipping: runs of >= 3 identical consecutive samples in the peak window,
        # on channels whose peak-window amplitude exceeds 10 baseline s.d.
        win = data[:, :, peak_mask]
        flat = np.abs(np.diff(win, axis=-1)) == 0
        runs3 = (flat[..., :-1] & flat[..., 1:]).any(-1)
        base_sd = data[:, :, base_mask].std(-1)
        large = np.abs(win).max(-1) > 10 * base_sd
        clipped = runs3 & large
        # full-sample estimate
        avg = basis.T @ data.mean(0)
        k = np.flatnonzero(peak_mask)[int(np.argmax(np.sum(avg[:, peak_mask] ** 2, 0)))]
        y_full = avg[:, k]
        base_all = np.einsum("ck,tcs->tsk", basis, data[:, :, base_mask]).reshape(-1, d)
        cov_full = oas_covariance(base_all) / n_trials
        # split halves (random partition, fixed seed)
        perm = rng.permutation(n_trials)
        ha, hb = np.sort(perm[: n_trials // 2]), np.sort(perm[n_trials // 2:])
        y_b, cov_b, t_b = half_field(data, basis, times, peak_mask, base_mask, ha, hb)
        y_a, cov_a, t_a = half_field(data, basis, times, peak_mask, base_mask, hb, ha)
        # forward model
        p1, p2 = contacts[first], contacts[second]
        spacing = float(np.linalg.norm(p2 - p1))
        info = mne.create_info(names, sfreq, "eeg")
        info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, positions)), coord_frame="head"))
        dip = mne.Dipole(np.array([0.0]), (0.5 * (p1 + p2))[None, :], np.array([current * spacing]),
                         ((p2 - p1) / spacing)[None, :], np.array([100.0]))
        fwd, _ = mne.make_forward_dipole(dip, bem, info, trans=trans, verbose=False)
        gain = np.asarray(fwd["sol"]["data"])[:, 0]
        pred = basis.T @ gain[good] * current * spacing
        sessions.append({
            "subject": subject, "run": run, "site": f"{first}-{second}", "current_a": current,
            "n_trials": n_trials, "good_channels": int(good.sum()), "dimension": d,
            "peak_time_s": float(times[k]), "peak_time_half_a_s": t_a, "peak_time_half_b_s": t_b,
            "clipped_channel_trials": int(clipped.sum()), "channel_trials": int(clipped.size),
            "mu_obs": energy(y_full, cov_full), "mu_pred": energy(pred, cov_full),
            # split-trial: field measured on trials not used to pick the latency
            "mu_obs_split": 0.5 * (energy(y_a, cov_a) + energy(y_b, cov_b)),
            "mu_obs_split_corrected": 0.5 * (energy(y_a, cov_a) + energy(y_b, cov_b)) - d,
            "mu_pred_split": 0.5 * (energy(pred, cov_a) + energy(pred, cov_b)),
            # observed field of each half whitened with the other half's covariance,
            # so that prediction and observation do not share a covariance estimate
            "mu_pred_crossed": 0.5 * (energy(pred, cov_a) + energy(pred, cov_b)),
            "mu_obs_crossed": 0.5 * (energy(y_a, cov_b) + energy(y_b, cov_a)),
            "euclid_obs": float(y_full @ y_full), "euclid_noise": float(np.trace(cov_full)),
            "euclid_pred": float(pred @ pred),
            "topo_own_split": 0.5 * (squared_cosine(y_a, pred) + squared_cosine(y_b, pred)),
            "y_peak": y_full.tolist(), "predicted": pred.tolist(), "good": good.tolist(),
            "gain_full": gain.tolist(), "positions": positions.tolist(),
            "field_power": (np.sum(avg ** 2, 0)[(times >= -0.02) & (times <= 0.01)]).tolist(),
            "field_power_times": times[(times >= -0.02) & (times <= 0.01)].tolist(),
        })
    return sessions


def ols(x, y):
    xm = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(xm, y, rcond=None)
    resid = y - xm @ coef
    return coef, resid


def endpoints(sessions, obs_key, pred_key, rng, boot=10000):
    obs = np.array([s[obs_key] for s in sessions], dtype=float)
    pred = np.array([s[pred_key] for s in sessions], dtype=float)
    keep = (obs > 0) & (pred > 0)
    subj = np.array([s["subject"] for s in sessions])[keep]
    lo, lp = np.log10(obs[keep]), np.log10(pred[keep])
    cur = np.log10([s["current_a"] for s in sessions])[keep]
    sites = np.array([s["subject"] + "|" + s["site"] for s in sessions])[keep]
    coef, resid = ols(lp, lo)
    n = len(lo)
    s_res = float(np.sqrt(np.sum(resid**2) / (n - 2)))
    xm = np.column_stack([np.ones(n), lp])
    xtx = np.linalg.inv(xm.T @ xm)
    lev = np.einsum("ij,jk,ik->i", xm, xtx, xm)
    from scipy.stats import t as tdist

    half_pi = float(tdist.ppf(0.975, n - 2) * s_res * np.sqrt(1 + np.mean(lev)))
    uniq = sorted(set(subj))

    def intensity_slope(idx, labels=None):
        labels = sites[idx] if labels is None else labels
        pairs = []
        for site in sorted(set(labels)):
            j = idx[labels == site]
            if len(j) >= 2:
                jl, jh = j[np.argmin(cur[j])], j[np.argmax(cur[j])]
                if cur[jh] > cur[jl]:
                    pairs.append((cur[jh] - cur[jl], lo[jh] - lo[jl]))
        if not pairs:
            return np.nan, 0
        p = np.array(pairs)
        return float(np.sum(p[:, 0] * p[:, 1]) / np.sum(p[:, 0] ** 2)), len(p)

    slope_i, n_pairs = intensity_slope(np.arange(n))
    b_cal, b_int = [], []
    for _ in range(boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(subj == s) for s in pick])
        draw = np.concatenate([np.full(np.sum(subj == s), k) for k, s in enumerate(pick)])
        if np.ptp(lp[idx]) > 0:
            b_cal.append(ols(lp[idx], lo[idx])[0][1])
        # a participant drawn twice contributes its sites twice
        si, _ = intensity_slope(idx, np.array([f"{sites[i]}#{k}" for i, k in zip(idx, draw)]))
        if np.isfinite(si):
            b_int.append(si)
    geo_o, geo_p = lo - 2 * cur, lp - 2 * cur
    from scipy.stats import spearmanr

    return {
        "n": int(n), "calibration_slope": float(coef[1]), "calibration_intercept": float(coef[0]),
        "calibration_slope_ci95": [float(np.percentile(b_cal, 2.5)), float(np.percentile(b_cal, 97.5))],
        "r2": float(1 - resid.var() / lo.var()), "residual_sd_log10": s_res,
        "residual_sd_fold": float(10**s_res),
        "prediction_interval_half_width_log10": half_pi, "prediction_interval_fold": float(10**half_pi),
        "current_exponent": slope_i, "current_pairs": n_pairs,
        "current_exponent_ci95": [float(np.percentile(b_int, 2.5)), float(np.percentile(b_int, 97.5))],
        "geometry_spearman": float(spearmanr(geo_p, geo_o).statistic),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--work", default="derived/localize_mi_subjects")
    parser.add_argument("--out", default="results/known_sources/sensitivity")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    root = Path(args.data)
    subjects_dir = Path(args.work) / "skull_0.006"
    sessions = []
    for subject in sorted(p.name for p in (root / "derivatives" / "epochs").glob("sub-*")):
        sessions.extend(analyse_subject(root, subject, subjects_dir, 0.006, rng))
        print(subject, len(sessions), flush=True)
    variants = {
        "full_trials": ("mu_obs", "mu_pred"),
        "split_trial": ("mu_obs_split", "mu_pred_split"),
        "split_trial_noise_corrected": ("mu_obs_split_corrected", "mu_pred_split"),
        "independent_covariances": ("mu_obs_crossed", "mu_pred_crossed"),
        "euclidean": ("euclid_obs", "euclid_pred"),
    }
    for s in sessions:
        s["euclid_obs_corrected"] = s["euclid_obs"] - s["euclid_noise"]
    variants["euclidean_noise_corrected"] = ("euclid_obs_corrected", "euclid_pred")
    result = {"endpoints": {k: endpoints(sessions, o, p, np.random.default_rng(args.seed + i))
                            for i, (k, (o, p)) in enumerate(variants.items())}}
    clipped = sum(s["clipped_channel_trials"] for s in sessions)
    total = sum(s["channel_trials"] for s in sessions)
    result["clipping"] = {"clipped_channel_trials": clipped, "channel_trials": total,
                          "sessions_with_any": int(sum(s["clipped_channel_trials"] > 0 for s in sessions))}
    result["peak_latency_ms"] = {
        "full_trials": np.percentile([1e3 * s["peak_time_s"] for s in sessions], [0, 50, 100]).tolist(),
        "half_disagreement_ms_median": float(np.median([1e3 * abs(s["peak_time_half_a_s"] - s["peak_time_half_b_s"])
                                                         for s in sessions])),
    }
    result["noise_term_relative"] = {
        "median_d_over_mu_obs": float(np.median([s["dimension"] / s["mu_obs"] for s in sessions])),
        "max_d_over_mu_obs": float(np.max([s["dimension"] / s["mu_obs"] for s in sessions])),
    }
    result["trials"] = [int(np.min([s["n_trials"] for s in sessions])), int(np.max([s["n_trials"] for s in sessions]))]
    result["good_channels"] = [int(np.min([s["good_channels"] for s in sessions])),
                               int(np.max([s["good_channels"] for s in sessions]))]
    result["sessions"] = sessions
    (out / "localize_mi_sensitivity.json").write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=1))


if __name__ == "__main__":
    main()
