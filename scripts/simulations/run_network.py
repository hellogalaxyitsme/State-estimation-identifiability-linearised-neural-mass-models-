"""Cortical network study: spatial--temporal separation on a template head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from neural_observability.theory import visibility as V
from neural_observability.theory.families import JansenRit
from neural_observability.head_models.template import load_or_build_forward
from neural_observability.theory.information import spatial_covariance
from neural_observability.theory.network import JRParameters, build_network_linearization


def snr_to_psd(lead: np.ndarray, source_cov: np.ndarray, snr_db: float) -> float:
    """Sensor PSD giving the requested mean channel SNR in a 100 Hz band.

    Columns are normalised to unit output variance, so channel signal variance
    is diag(L S L^T).
    """
    signal = float(np.mean(np.diag(lead @ source_cov @ lead.T)))
    return signal / (200.0 * 10 ** (snr_db / 10.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--cache-dir", default="derived/cache")
    parser.add_argument("--montages", default="19,32,64,128,256")
    parser.add_argument("--snr", default="-10,0,10,20,30,40")
    parser.add_argument("--drive", type=float, default=90.0)
    parser.add_argument("--out", default="results/simulations/network")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fwd_cfg = {
        "kind": "fsaverage_bem",
        "subjects_dir": str(Path(args.subjects_dir).resolve()),
        "spacing": "ico4",
        "parcellation": "aparc",
        "patch_normalization": "l2",
        "cache_dir": str(Path(args.cache_dir).resolve()),
    }
    jr = JansenRit()
    x_star = jr.equilibria(args.drive)[0]
    a0, h0 = jr.jacobian(x_star), jr.output_vector()
    b_raw = jr.input_vector()
    b0 = b_raw / np.sqrt(h0 @ V.stationary_covariance(a0, b_raw) @ h0)
    snrs = [float(s) for s in args.snr.split(",")]
    record = {"drive": args.drive, "state_labels": list(jr.state_labels), "montages": {}}
    for montage in args.montages.split(","):
        start = time.time()
        fwd = load_or_build_forward(fwd_cfg, Path("."), montage)
        lead = fwd.lead_field / np.median(np.linalg.norm(fwd.lead_field, axis=0))
        positions = fwd.positions_m
        n_src = lead.shape[1]
        priors = {
            "independent": np.eye(n_src),
            "smooth_40mm": spatial_covariance(positions, {"kind": "rbf", "length_scale_m": 0.04, "nugget": 0.05}),
        }
        m_rec = {"channels": int(fwd.metadata["physical_channel_count"]), "sources": fwd.source_names, "priors": {}}
        for prior_name, s_cov in priors.items():
            p_rec = {"by_snr": {}}
            for snr in snrs:
                psd = snr_to_psd(lead, s_cov, snr)
                sep = V.separate_network(a0, b0, h0, lead, psd * np.eye(lead.shape[0]), s_cov)
                r2 = sep.source_r2()
                hsel = np.array([0.0, 1.0, -1.0, 0, 0, 0])
                # output R^2 per source
                weights = sep.mixing**2
                post_out = np.einsum("ni,i->n", weights, np.einsum("j,ijk,k->i", hsel, sep.local_posteriors, hsel))
                prior_out = weights.sum(1) * float(hsel @ sep.local_prior @ hsel)
                p_rec["by_snr"][f"{snr:g}"] = {
                    "state_r2": r2.tolist(),
                    "output_r2": (1 - post_out / prior_out).tolist(),
                    "spatial_snr": sep.spatial_snr.tolist(),
                }
            p_rec["ceiling_output_r2"] = sep.ceiling_r2().tolist()
            m_rec["priors"][prior_name] = p_rec
        # exactness check against the full network Riccati (identical, uncoupled)
        checks = {}
        if montage in ("19", "64"):
            s_cov = priors["independent"]
            for snr in (0.0, 20.0):
                psd = snr_to_psd(lead, s_cov, snr)
                a_n, b_n, c_n = V.network_matrices(a0, b0, h0, lead, s_cov)
                full = V.stationary_recoverability(a_n, b_n, c_n, psd * np.eye(lead.shape[0]))
                full_r2 = full.physical_r2("filtered").reshape(n_src, 6)
                sep = V.separate_network(a0, b0, h0, lead, psd * np.eye(lead.shape[0]), s_cov)
                checks[f"identical_{snr:g}dB_max_abs_diff"] = float(np.max(np.abs(full_r2 - sep.source_r2())))
            # heterogeneous, coupled network (matched filter and nominal-filter mismatch)
            rng = np.random.default_rng(20261003)
            for label, coupling, drive_sd in (("coupled", 6.0, 0.0), ("heterogeneous_coupled", 6.0, 0.10)):
                a_true, _, _, meta = build_network_linearization(
                    positions,
                    {"name": label, "p0_per_second": args.drive, "equilibrium_branch": "lowest",
                     "coupling_gain": coupling, "drive_relative_sd": drive_sd},
                    {"nearest_neighbors": 6, "length_scale_m": 0.04},
                    rng,
                    JRParameters(),
                )
                b_true = np.kron(np.eye(n_src), b0.reshape(6, 1))
                c_true = np.kron(lead, h0.reshape(1, 6))
                entry = {"spectral_abscissa": meta["spectral_abscissa_per_second"]}
                for snr in (0.0, 20.0, 40.0):
                    psd = snr_to_psd(lead, np.eye(n_src), snr)
                    matched = V.stationary_recoverability(a_true, b_true, c_true, psd * np.eye(lead.shape[0]))
                    sep = V.separate_network(a0, b0, h0, lead, psd * np.eye(lead.shape[0]), np.eye(n_src))
                    m_r2 = matched.physical_r2("filtered").reshape(n_src, 6)
                    entry[f"{snr:g}dB"] = {
                        "matched_r2_median_by_state": np.median(m_r2, 0).tolist(),
                        "separation_prediction_median_by_state": np.median(sep.source_r2(), 0).tolist(),
                        "max_abs_diff_matched_vs_separation": float(np.max(np.abs(m_r2 - sep.source_r2()))),
                        "median_abs_diff_matched_vs_separation": float(np.median(np.abs(m_r2 - sep.source_r2()))),
                    }
                checks[label] = entry
        m_rec["checks"] = checks
        record["montages"][montage] = m_rec
        with open(out / "network_results.json", "w") as handle:
            json.dump(record, handle)
        print(f"montage {montage}: {n_src} sources, {time.time() - start:.0f} s, checks {json.dumps(checks)[:400]}", flush=True)


if __name__ == "__main__":
    main()
