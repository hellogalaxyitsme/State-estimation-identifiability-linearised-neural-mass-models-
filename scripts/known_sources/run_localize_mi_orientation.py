"""Diagnostic of the assumed dipole orientation in the Localize-MI analysis.

Asks whether the assumed shaft-aligned dipole orientation limits the absolute
visibility prediction, by comparing the shaft-aligned prediction with an
orientation-averaged one, q^2 trace(G^T Sigma^{-1} G)/3, for free-orientation
gain G at the same location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
from scipy.stats import spearmanr

from neural_observability.known_sources.localize_mi import parse_session, read_tsv, session_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/localize_mi/extracted", help="Localize-MI dataset root")
    parser.add_argument("--work", default="derived/localize_mi_subjects",
                        help="head models written by run_localize_mi.py")
    parser.add_argument("--out", default="results/known_sources/localize_mi")
    args = parser.parse_args()
    der = Path(args.data) / "derivatives"
    subjects_dir = Path(args.work) / "skull_0.006"
    out = []
    for subject in sorted(p.name for p in (der / "epochs").glob("sub-*")):
        eeg_dir = der / "epochs" / subject / "eeg"
        electrodes = read_tsv(eeg_dir / f"{subject}_task-seegstim_electrodes.tsv")
        names = [r["name"] for r in electrodes]
        pos = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in electrodes])
        contacts = {r["name"]: np.array([float(r["x"]), float(r["y"]), float(r["z"])])
                    for r in read_tsv(der / "epochs" / subject / "ieeg" / f"{subject}_task-seegstim_space-T1w_electrodes.tsv")}
        trans = mne.read_forward_solution(der / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif", verbose=False)["mri_head_t"]
        model = mne.make_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir, verbose=False)
        bem = mne.make_bem_solution(model, verbose=False)
        info = mne.create_info(names, 8000.0, "eeg")
        info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, pos)), coord_frame="head"))
        for meta_path in sorted(eeg_dir.glob(f"{subject}_task-seegstim_run-*_epochs.json")):
            run = meta_path.name.split("_run-")[1].split("_")[0]
            first, second, current = parse_session(json.loads(meta_path.read_text())["Description"])
            channels = read_tsv(eeg_dir / f"{subject}_task-seegstim_run-{run}_channels.tsv")
            good = np.array([c["status"] == "good" for c in channels])
            zero = float(read_tsv(eeg_dir / f"{subject}_task-seegstim_run-{run}_epochs.tsv")[0]["zero_time"])
            epochs = np.load(eeg_dir / f"{subject}_task-seegstim_run-{run}_epochs.npy", mmap_mode="r")
            y, sigma, basis, _, _ = session_field(np.asarray(epochs), good, 8000.0, zero, (0.0, 0.005), (-0.25, -0.05))
            p1, p2 = contacts[first], contacts[second]
            d = float(np.linalg.norm(p2 - p1))
            mid = 0.5 * (p1 + p2)
            ori = (p2 - p1) / d
            gains = []
            for axis in np.eye(3):
                dip = mne.Dipole(np.array([0.0]), mid[None], np.array([1.0]), axis[None], np.array([100.0]))
                fwd, _ = mne.make_forward_dipole(dip, bem, info, trans=trans, verbose=False)
                gains.append(np.asarray(fwd["sol"]["data"])[:, 0])
            g = basis.T @ np.column_stack(gains)[good]  # channels x 3
            whitened = np.linalg.solve(sigma, g)
            fisher = g.T @ whitened  # 3 x 3
            q2 = (current * d) ** 2
            out.append({
                "subject": subject, "run": run, "current_a": current,
                "mu_obs": float(y @ np.linalg.solve(sigma, y)),
                "mu_shaft": q2 * float(ori @ fisher @ ori),
                "mu_orientation_average": q2 * float(np.trace(fisher)) / 3.0,
                "mu_best_orientation": q2 * float(np.linalg.eigvalsh(fisher)[-1]),
                "observed_vs_shaft_cosine2": float((ori @ np.linalg.lstsq(g, y, rcond=None)[0]) ** 2
                                                  / max(1e-30, np.sum(np.linalg.lstsq(g, y, rcond=None)[0] ** 2))),
            })
        print(subject, len(out), flush=True)
    obs = np.log10([r["mu_obs"] for r in out])
    cur = np.log10([r["current_a"] for r in out])
    summary = {}
    for key in ("mu_shaft", "mu_orientation_average", "mu_best_orientation"):
        pred = np.log10([r[key] for r in out])
        slope, intercept = np.polyfit(pred, obs, 1)
        summary[key] = {
            "calibration_slope": float(slope),
            "r2": float(np.corrcoef(pred, obs)[0, 1] ** 2),
            "spearman_geometry_beyond_current": float(spearmanr(pred - 2 * cur, obs - 2 * cur).statistic),
        }
    summary["median_cos2_fitted_vs_shaft_orientation"] = float(np.median([r["observed_vs_shaft_cosine2"] for r in out]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "orientation_diagnostic.json").write_text(json.dumps({"summary": summary, "sessions": out}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
