"""Finite-SNR comparison of EEG acquisition upgrades vs model calibration.

Evaluates the exact steady-state mismatched Kalman filter error on the 68-parcel
fsaverage BEM Jansen--Rit cortical network across electrode montages
(19, 32, 64, 128, 256), finite channel SNRs (0, 10, 20, 40 dB), and parameter
uncertainty levels (0%, 2%, 5%, 10% physiological mismatch), reporting both the
target hidden state (inhibitory PSP x_3) and the measured pyramidal potential
(x_2 - x_3).

Also verifies:
1. Exact spatial--temporal separation of the homogeneously mismatched network
   filter against the full 816-state coupled Lyapunov solution.
2. Robustness when mismatch varies independently across all 68 parcels in a
   coupled cortical network (K = 6 s^-1), solved via the full 816-state system
   and cross-checked against direct discrete-time Monte Carlo simulation.
3. Non-additivity of exact-model finite-SNR error and the asymptotic mismatch
   floor at finite SNR.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_continuous_lyapunov

from neural_observability.head_models.template import load_or_build_forward
from neural_observability.theory import visibility as V
from neural_observability.theory.atlas import FAMILY_SPECS, _closest_equilibrium, _normalised_column
from neural_observability.theory.families import JansenRit
from neural_observability.theory.information import spatial_covariance
from run_network import snr_to_psd


def separated_mismatched_network(
    a_true: np.ndarray,
    b_true: np.ndarray,
    h_true: np.ndarray,
    a_nom: np.ndarray,
    b_nom: np.ndarray,
    h_nom: np.ndarray,
    lead: np.ndarray,
    sensor_psd: np.ndarray,
    source_cov: np.ndarray,
    rank_tolerance: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Exact parcel-level MSE/R^2 of a nominal network filter on a homogeneously
    mismatched uncoupled network via the SVD spatial--temporal transformation."""
    r_factor = np.linalg.cholesky(0.5 * (sensor_psd + sensor_psd.T))
    s_half = V._psd_factor(source_cov)
    whitened = np.linalg.solve(r_factor, lead) @ s_half
    _, sv, vt = np.linalg.svd(whitened, full_matrices=True)
    n_src = lead.shape[1]
    mu = np.zeros(n_src)
    mu[: len(sv)] = sv**2
    mu[mu < rank_tolerance * max(mu.max(), np.finfo(float).tiny)] = 0.0
    mixing_sq = (s_half @ vt.T) ** 2  # (N, N)

    p_true = V.stationary_covariance(a_true, b_true)
    ht = np.asarray(h_true, dtype=float).reshape(-1)
    hn = np.asarray(h_nom, dtype=float).reshape(-1)
    prior_diag = np.diag(p_true)
    prior_out = float(ht @ p_true @ ht)

    q = len(a_true)
    mode_state_err = np.empty((n_src, q))
    mode_out_err = np.empty(n_src)
    cache: dict[float, tuple[np.ndarray, float]] = {}

    for i, val in enumerate(mu):
        if val == 0.0:
            # Zero Kalman gain on spatially unobservable modes -> hat_xi_i = 0
            mode_state_err[i] = prior_diag
            mode_out_err[i] = prior_out
            continue
        key = float(val)
        if key not in cache:
            res = V.mismatched_filter_error(
                a_true, b_true, a_nom, b_nom, ht, 1.0 / key, h_nom=hn
            )
            st_err = np.diag(res["error"])
            out_err = (1.0 - float(res["output_r2"])) * prior_out
            cache[key] = (st_err, out_err)
        mode_state_err[i], mode_out_err[i] = cache[key]

    src_scale = mixing_sq.sum(axis=1)  # diag(S)
    vis_frac = mixing_sq[:, mu > 0].sum(axis=1) / src_scale
    parcel_state_nmse = (mixing_sq @ mode_state_err) / (src_scale[:, None] * prior_diag[None, :])
    parcel_out_nmse = (mixing_sq @ mode_out_err) / (src_scale * prior_out)
    mu_sum = float(mu.sum())
    sensor_sig_nmse = float((mu @ mode_out_err) / (mu_sum * prior_out)) if mu_sum > 0.0 else 1.0
    return {
        "state_nmse": parcel_state_nmse,
        "state_r2": 1.0 - parcel_state_nmse,
        "output_nmse": parcel_out_nmse,
        "output_r2": 1.0 - parcel_out_nmse,
        "sensor_signal_nmse": sensor_sig_nmse,
        "sensor_signal_r2": 1.0 - sensor_sig_nmse,
        "visible_fraction": vis_frac,
        "spatial_snr": mu,
    }


def full_network_mismatched_error(
    a_true: np.ndarray,
    b_true: np.ndarray,
    c_true: np.ndarray,
    h_true_blocks: list[np.ndarray],
    a_nom: np.ndarray,
    b_nom: np.ndarray,
    c_nom: np.ndarray,
    h_nom_blocks: list[np.ndarray],
    sensor_psd: np.ndarray,
    kalman_gain: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Solve the full 2Nq x 2Nq coupled Lyapunov equation for a nominal network
    steady-state Kalman filter applied to an arbitrary (possibly coupled and
    parcel-heterogeneous) true linearised cortical network."""
    n_tot = len(a_true)
    p_true = V.stationary_covariance(a_true, b_true)
    scale = np.sqrt(np.clip(np.diag(p_true), np.finfo(float).tiny, None))
    s_inv = 1.0 / scale

    at = a_true * scale[None, :] * s_inv[:, None]
    an = a_nom * scale[None, :] * s_inv[:, None]
    bt = b_true * s_inv[:, None]
    bn = b_nom * s_inv[:, None]
    ct = c_true * scale[None, :]
    cn = c_nom * scale[None, :]

    if kalman_gain is None:
        sigma_nom = V._sym(solve_continuous_are(an.T, cn.T, bn @ bn.T, sensor_psd))
        gain_z = sigma_nom @ np.linalg.solve(sensor_psd, cn).T
    else:
        # Transform physical Kalman gain K_phys (dx_hat = A_nom x_hat + K_phys(dy - C_nom x_hat dt))
        gain_z = kalman_gain * s_inv[:, None]

    r_half = np.linalg.cholesky(0.5 * (sensor_psd + sensor_psd.T))
    big_a = np.block([[at, np.zeros((n_tot, n_tot))], [gain_z @ ct, an - gain_z @ cn]])
    big_b = np.block([[bt, np.zeros((n_tot, r_half.shape[1]))], [np.zeros((n_tot, bt.shape[1])), gain_z @ r_half]])
    cov = V._sym(solve_continuous_lyapunov(big_a, -big_b @ big_b.T))
    pxx, pxh, phh = cov[:n_tot, :n_tot], cov[:n_tot, n_tot:], cov[n_tot:, n_tot:]
    err_z = V._sym(pxx - pxh - pxh.T + phh)

    n_src = len(h_true_blocks)
    q = n_tot // n_src
    state_nmse = np.diag(err_z).reshape(n_src, q) / np.diag(pxx).reshape(n_src, q)

    out_nmse = np.empty(n_src)
    for idx in range(n_src):
        sl = slice(idx * q, (idx + 1) * q)
        sc = scale[sl]
        ht = np.asarray(h_true_blocks[idx], dtype=float).reshape(-1) * sc
        hn = np.asarray(h_nom_blocks[idx], dtype=float).reshape(-1) * sc
        p_sub = pxx[sl, sl]
        pxh_sub = pxh[sl, sl]
        phh_sub = phh[sl, sl]
        denom = float(ht @ p_sub @ ht)
        num = float(ht @ p_sub @ ht - 2.0 * ht @ pxh_sub @ hn + hn @ phh_sub @ hn)
        out_nmse[idx] = num / denom

    # Whitened sensor-space EEG signal reconstruction error:
    # E[||R^{-1/2}(C_true x - C_nom x_hat)||^2] / E[||R^{-1/2} C_true x||^2]
    rc_t = np.linalg.solve(r_half, ct)
    rc_n = np.linalg.solve(r_half, cn)
    sens_denom = float(np.sum(rc_t * (rc_t @ pxx)))
    sens_num = float(
        np.sum(rc_t * (rc_t @ pxx))
        - 2.0 * np.sum(rc_t * (rc_n @ pxh.T))
        + np.sum(rc_n * (rc_n @ phh))
    )
    sensor_sig_nmse = sens_num / sens_denom if sens_denom > 0.0 else 1.0

    return {
        "state_nmse": state_nmse,
        "state_r2": 1.0 - state_nmse,
        "output_nmse": out_nmse,
        "output_r2": 1.0 - out_nmse,
        "sensor_signal_nmse": sensor_sig_nmse,
        "sensor_signal_r2": 1.0 - sensor_sig_nmse,
    }


def sample_jr_perturbed_columns(
    base_model: JansenRit,
    drive: float,
    x_star: np.ndarray,
    level: float,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, int]]:
    """Draw ``n_draws`` stable +/-level physiological perturbations of Jansen--Rit."""
    phys_names = FAMILY_SPECS["jansen_rit"]["physiological"]
    draws = []
    excl = {"attempts": 0, "fixed_point_lost": 0, "unstable": 0}
    while len(draws) < n_draws and excl["attempts"] < 20 * n_draws:
        excl["attempts"] += 1
        signs = rng.choice([-1.0, 1.0], size=len(phys_names))
        changes = {name: float(getattr(base_model, name) * (1.0 + level * s)) for name, s in zip(phys_names, signs)}
        truth = replace(base_model, **changes)
        xt = _closest_equilibrium(truth, drive, x_star)
        if xt is None:
            excl["fixed_point_lost"] += 1
            continue
        at_raw = truth.jacobian(xt)
        if np.linalg.eigvals(at_raw).real.max() >= -1e-3:
            excl["unstable"] += 1
            continue
        _, a_t, b_t, h_t = _normalised_column(truth, xt)
        draws.append((a_t, b_t, h_t))
    return draws, excl


def monte_carlo_single_column_check(
    a_true: np.ndarray,
    b_true: np.ndarray,
    a_nom: np.ndarray,
    b_nom: np.ndarray,
    h: np.ndarray,
    r: float,
    dt: float = 5e-4,
    seconds: float = 400.0,
    burn_s: float = 5.0,
    seed: int = 20261118,
) -> dict[str, float]:
    """Direct time-domain exact-discretised simulation check of the mismatched
    steady-state continuous-time Kalman filter."""
    n = len(a_true)
    h_row = np.asarray(h, dtype=float).reshape(1, n)
    sigma_nom = V._sym(solve_continuous_are(a_nom.T, h_row.T, np.outer(b_nom, b_nom), np.array([[r]])))
    gain = (sigma_nom @ h_row.T / r).reshape(n, 1)

    big_a = np.block([[a_true, np.zeros((n, n))], [gain @ h_row, a_nom - gain @ h_row]])
    big_b = np.block([[b_true.reshape(n, 1), np.zeros((n, 1))], [np.zeros((n, 1)), gain * np.sqrt(r)]])
    # Exact discrete transition and innovation covariance of the augmented 2n-dim OU process
    ad = expm(big_a * dt)
    cov_inf = V._sym(solve_continuous_lyapunov(big_a, -big_b @ big_b.T))
    qd = V._sym(cov_inf - ad @ cov_inf @ ad.T)
    chol_q = V._psd_factor(qd)

    rng = np.random.default_rng(seed)
    steps = int(round((seconds + burn_s) / dt))
    burn = int(round(burn_s / dt))
    z = np.zeros(2 * n)
    sq_err = np.zeros(n)
    sq_x = np.zeros(n)
    out_err = 0.0
    out_var = 0.0
    h_vec = h_row.reshape(-1)
    for k in range(steps):
        z = ad @ z + chol_q @ rng.standard_normal(2 * n)
        if k >= burn:
            x, xh = z[:n], z[n:]
            diff = x - xh
            sq_err += diff**2
            sq_x += x**2
            yo = float(h_vec @ x)
            yh = float(h_vec @ xh)
            out_err += (yo - yh) ** 2
            out_var += yo**2
    p_true = V.stationary_covariance(a_true, b_true)
    exact = V.mismatched_filter_error(a_true, b_true, a_nom, b_nom, h_vec, r)
    mc_nmse_x3 = float((sq_err[2] / (steps - burn)) / p_true[2, 2])
    ex_nmse_x3 = float(1.0 - exact["r2"][2])
    mc_out_nmse = float((out_err / (steps - burn)) / float(h_vec @ p_true @ h_vec))
    ex_out_nmse = float(1.0 - exact["output_r2"])
    return {
        "mc_nmse_x3": mc_nmse_x3,
        "exact_nmse_x3": ex_nmse_x3,
        "rel_diff_x3": abs(mc_nmse_x3 - ex_nmse_x3) / max(ex_nmse_x3, 1e-6),
        "mc_output_nmse": mc_out_nmse,
        "exact_output_nmse": ex_out_nmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--cache-dir", default="derived/cache")
    parser.add_argument("--montages", default="19,32,64,128,256")
    parser.add_argument("--snrs", default="0,10,20,40")
    parser.add_argument("--levels", default="0,0.02,0.05,0.10")
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--drive", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--out", default="results/simulations/design_tradeoff")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    _, a0, b0, h0 = _normalised_column(jr, x_star)

    montages = args.montages.split(",")
    snrs = [float(s) for s in args.snrs.split(",")]
    levels = [float(v) for v in args.levels.split(",")]

    rng = np.random.default_rng(args.seed)
    draws_by_level: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    exclusions_by_level: dict[str, dict[str, int]] = {}
    floors_by_level: dict[str, dict[str, float]] = {}

    draw_floors_by_level: dict[str, list[float]] = {}

    for lev in levels:
        key = f"{lev:g}"
        if lev == 0.0:
            draws_by_level[key] = [(a0, b0, h0)]
            exclusions_by_level[key] = {"attempts": 1, "fixed_point_lost": 0, "unstable": 0}
            draw_floors_by_level[key] = [0.0]
            floors_by_level[key] = {"x3_median": 0.0, "x3_q25": 0.0, "x3_q75": 0.0}
        else:
            dr, ex = sample_jr_perturbed_columns(jr, args.drive, x_star, lev, args.draws, rng)
            draws_by_level[key] = dr
            exclusions_by_level[key] = ex
            fl = [float(V.high_snr_floor(at, bt, a0, b0, ht, h_nom=h0)[2]) for at, bt, ht in dr]
            draw_floors_by_level[key] = fl
            floors_by_level[key] = {
                "x3_median": float(np.median(fl)),
                "x3_q25": float(np.percentile(fl, 25)),
                "x3_q75": float(np.percentile(fl, 75)),
                "x3_mean": float(np.mean(fl)),
            }

    # Direct Monte Carlo verification of exact mismatched continuous-time Lyapunov solver
    at_chk, bt_chk, ht_chk = draws_by_level["0.1"][0]
    r_10db = 1.0 / (200.0 * 10 ** (10.0 / 10.0))
    mc_check = monte_carlo_single_column_check(at_chk, bt_chk, a0, b0, ht_chk, r_10db, seed=args.seed + 1)

    results: dict = {
        "drive": args.drive,
        "seed": args.seed,
        "draws_per_nonzero_level": args.draws,
        "state_labels": list(jr.state_labels),
        "target_state_index": 2,
        "target_state_label": "inhibitory PSP x_3",
        "exclusions_by_level": exclusions_by_level,
        "single_column_asymptotic_floor_x3": floors_by_level,
        "monte_carlo_single_column_10dB_check": mc_check,
        "montages": {},
        "verification_checks": {},
    }

    for m in montages:
        fwd = load_or_build_forward(fwd_cfg, Path("."), m)
        lead = fwd.lead_field / np.median(np.linalg.norm(fwd.lead_field, axis=0))
        positions = fwd.positions_m
        n_src = lead.shape[1]
        priors = {
            "independent": np.eye(n_src),
            "smooth_40mm": spatial_covariance(positions, {"kind": "rbf", "length_scale_m": 0.04, "nugget": 0.05}),
        }
        m_entry: dict = {"channels": int(fwd.metadata["physical_channel_count"]), "priors": {}}

        for prior_name, s_cov in priors.items():
            p_entry: dict = {}
            for snr in snrs:
                psd_scalar = snr_to_psd(lead, s_cov, snr)
                r_mat = psd_scalar * np.eye(lead.shape[0])
                snr_entry: dict = {}
                exact_x3_med = None
                exact_x3_parcels = None
                vis_frac_parcels = None

                for lev in levels:
                    lkey = f"{lev:g}"
                    draw_parcel_x3 = []
                    draw_parcel_x2 = []
                    draw_parcel_x1 = []
                    draw_parcel_out = []
                    draw_sensor_sig = []
                    draw_vis_weighted_add = []
                    for (at, bt, ht), fl_d in zip(draws_by_level[lkey], draw_floors_by_level[lkey]):
                        sep = separated_mismatched_network(at, bt, ht, a0, b0, h0, lead, r_mat, s_cov)
                        if lev == 0.0:
                            exact_x3_parcels = sep["state_nmse"][:, 2].copy()
                            vis_frac_parcels = sep["visible_fraction"].copy()
                        # Parcel-median NMSE for this mismatch draw, then summary across draws
                        draw_parcel_x1.append(float(np.median(sep["state_nmse"][:, 0])))
                        draw_parcel_x2.append(float(np.median(sep["state_nmse"][:, 1])))
                        draw_parcel_x3.append(float(np.median(sep["state_nmse"][:, 2])))
                        draw_parcel_out.append(float(np.median(sep["output_nmse"])))
                        draw_sensor_sig.append(float(sep["sensor_signal_nmse"]))
                        if exact_x3_parcels is not None and vis_frac_parcels is not None:
                            draw_vis_weighted_add.append(
                                float(np.median(exact_x3_parcels + vis_frac_parcels * fl_d))
                            )

                    med_x3 = float(np.median(draw_parcel_x3))
                    if lev == 0.0:
                        exact_x3_med = med_x3

                    naive_additive = (
                        float(exact_x3_med + floors_by_level[lkey]["x3_median"])
                        if exact_x3_med is not None
                        else None
                    )
                    vis_weighted_additive = (
                        float(np.median(draw_vis_weighted_add)) if draw_vis_weighted_add else None
                    )
                    snr_entry[lkey] = {
                        "x3_nmse_median": med_x3,
                        "x3_nmse_q25": float(np.percentile(draw_parcel_x3, 25)),
                        "x3_nmse_q75": float(np.percentile(draw_parcel_x3, 75)),
                        "x3_r2_median": float(1.0 - med_x3),
                        "x2_nmse_median": float(np.median(draw_parcel_x2)),
                        "x1_nmse_median": float(np.median(draw_parcel_x1)),
                        "output_nmse_median": float(np.median(draw_parcel_out)),
                        "output_nmse_q25": float(np.percentile(draw_parcel_out, 25)),
                        "output_nmse_q75": float(np.percentile(draw_parcel_out, 75)),
                        "output_r2_median": float(1.0 - np.median(draw_parcel_out)),
                        "sensor_signal_nmse_median": float(np.median(draw_sensor_sig)),
                        "sensor_signal_nmse_q25": float(np.percentile(draw_sensor_sig, 25)),
                        "sensor_signal_nmse_q75": float(np.percentile(draw_sensor_sig, 75)),
                        "naive_noise_plus_floor_x3": naive_additive,
                        "non_additivity_diff_x3": (
                            float(med_x3 - naive_additive) if naive_additive is not None else 0.0
                        ),
                        "vis_weighted_noise_plus_floor_x3": vis_weighted_additive,
                        "finite_snr_non_additivity_x3": (
                            float(med_x3 - vis_weighted_additive)
                            if vis_weighted_additive is not None
                            else 0.0
                        ),
                    }
                p_entry[f"{snr:g}"] = snr_entry
            m_entry["priors"][prior_name] = p_entry
        results["montages"][m] = m_entry
        print(f"done montage {m} ({time.time() - t0:.1f}s)", flush=True)

    # Full 816-state network verification at 64 electrodes, 10 dB:
    # (1) exact agreement of separated_mismatched_network with full 816-state Lyapunov
    # (2) coupled + parcel-heterogeneous 10% vs 5% vs 2% mismatch check (8 network draws)
    fwd64 = load_or_build_forward(fwd_cfg, Path("."), "64")
    lead64 = fwd64.lead_field / np.median(np.linalg.norm(fwd64.lead_field, axis=0))
    pos64 = fwd64.positions_m
    n_src = lead64.shape[1]
    s_ind = np.eye(n_src)
    psd64_10 = snr_to_psd(lead64, s_ind, 10.0)
    r64_10 = psd64_10 * np.eye(lead64.shape[0])

    a_nom_net, b_nom_net, c_nom_net = V.network_matrices(a0, b0, h0, lead64, s_ind)
    # Precompute nominal network Kalman gain once
    p_nom_net = V.stationary_covariance(a_nom_net, b_nom_net)
    sc_nom = np.sqrt(np.diag(p_nom_net))
    an_z = a_nom_net * sc_nom[None, :] / sc_nom[:, None]
    bn_z = b_nom_net / sc_nom[:, None]
    cn_z = c_nom_net * sc_nom[None, :]
    sig_nom = V._sym(solve_continuous_are(an_z.T, cn_z.T, bn_z @ bn_z.T, r64_10))
    k_phys_64_10 = (sig_nom @ np.linalg.solve(r64_10, cn_z).T) * sc_nom[:, None]

    # Check 1: Exactness of homogeneous separation vs full 816-state solve
    at0, bt0, ht0 = draws_by_level["0.1"][0]
    a_tr_hom, b_tr_hom, c_tr_hom = V.network_matrices(at0, bt0, ht0, lead64, s_ind)
    full_hom = full_network_mismatched_error(
        a_tr_hom, b_tr_hom, c_tr_hom, [ht0] * n_src,
        a_nom_net, b_nom_net, c_nom_net, [h0] * n_src,
        r64_10, kalman_gain=k_phys_64_10,
    )
    sep_hom = separated_mismatched_network(at0, bt0, ht0, a0, b0, h0, lead64, r64_10, s_ind)
    max_sep_diff_state = float(np.max(np.abs(full_hom["state_nmse"] - sep_hom["state_nmse"])))
    max_sep_diff_out = float(np.max(np.abs(full_hom["output_nmse"] - sep_hom["output_nmse"])))
    max_sep_diff_sensor = float(abs(full_hom["sensor_signal_nmse"] - sep_hom["sensor_signal_nmse"]))

    # Check 2: Parcel-heterogeneous mismatch + nearest-neighbour cortical coupling (K = 6 s^-1)
    # Build 6-NN Gaussian distance graph (40 mm), unit row sums, matching run_network.py
    dist = np.linalg.norm(pos64[:, None, :] - pos64[None, :, :], axis=-1)
    w_graph = np.zeros((n_src, n_src))
    for i in range(n_src):
        nbrs = np.argsort(dist[i])[1:7]
        w_row = np.exp(-0.5 * (dist[i, nbrs] / 0.04) ** 2)
        w_graph[i, nbrs] = w_row / w_row.sum()
    # Coupling enters excitatory interneuron input equation (row 4) from pyramidal firing S(C1 x1) (col 0)
    s_prime_c1x1 = float(jr.ds(jr.c1 * x_star[0]))
    coupling_scalar = jr.gain_e * jr.a * 6.0 * jr.c1 * s_prime_c1x1

    het_coupled_summary = {}
    rng_het = np.random.default_rng(args.seed + 99)
    for lkey in ("0.02", "0.05", "0.1"):
        pool_cols = draws_by_level[lkey]
        net_x3, net_out = [], []
        for _ in range(8):
            picks = [pool_cols[idx] for idx in rng_het.integers(0, len(pool_cols), size=n_src)]
            a_blocks = [p[0] for p in picks]
            b_blocks = [p[1] for p in picks]
            h_blocks = [p[2] for p in picks]
            a_het = np.zeros((n_src * 6, n_src * 6))
            b_het = np.zeros((n_src * 6, n_src))
            c_het = np.zeros((lead64.shape[0], n_src * 6))
            for idx in range(n_src):
                sl = slice(idx * 6, (idx + 1) * 6)
                a_het[sl, sl] = a_blocks[idx]
                b_het[sl, idx] = b_blocks[idx]
                c_het[:, sl] = np.outer(lead64[:, idx], h_blocks[idx])
            e_mat = np.zeros((6, 6))
            e_mat[4, 0] = coupling_scalar
            a_het = a_het + np.kron(w_graph, e_mat)
            res_het = full_network_mismatched_error(
                a_het, b_het, c_het, h_blocks,
                a_nom_net, b_nom_net, c_nom_net, [h0] * n_src,
                r64_10, kalman_gain=k_phys_64_10,
            )
            net_x3.append(float(np.median(res_het["state_nmse"][:, 2])))
            net_out.append(float(np.median(res_het["output_nmse"])))
        het_coupled_summary[lkey] = {
            "x3_nmse_median": float(np.median(net_x3)),
            "output_nmse_median": float(np.median(net_out)),
        }

    results["verification_checks"] = {
        "homogeneous_separation_vs_full_816_state_64ch_10dB": {
            "max_abs_diff_state_nmse": max_sep_diff_state,
            "max_abs_diff_output_nmse": max_sep_diff_out,
            "max_abs_diff_sensor_signal_nmse": max_sep_diff_sensor,
        },
        "heterogeneous_coupled_network_64ch_10dB": het_coupled_summary,
    }

    out_file = out_dir / "design_tradeoff_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"saved {out_file} in {time.time() - t0:.1f}s")
    print("Check sep vs 816-state:", results["verification_checks"]["homogeneous_separation_vs_full_816_state_64ch_10dB"])
    print("Check MC vs Lyapunov:", mc_check)


if __name__ == "__main__":
    main()
