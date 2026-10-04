"""Family-level visibility atlas.

For each neural-mass family, sample stable operating points across
physiological parameter ranges and compute, per state:

* steady-state recoverability (filter, smoother, future-only) versus SNR,
* Hankel spectra and transmission zeros of the minimal realisation,
* the share of each readout state in the output variance,
* the high-SNR error floor under physiological parameter mismatch, with a
  check against the exact mismatched filter and a quadrature check,
* recoverability when a second, independent drive reaches the inhibitory
  population,
* the expected hidden-state loss after T seconds of data, and
* the rank of the map from physiological to linear parameters.

Every exclusion (failed fixed-point search, unstable or lost operating point,
rejected draw) is counted so that the inclusion record can be reported.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from neural_observability.theory import visibility as V
from neural_observability.theory.families import JansenRit, NextGenerationEI, Wendling
from neural_observability.theory.linear_families import linear_column, verification_analysis

WENDLING_2002 = dict(gain_e=4.5, gain_s=10.0, gain_f=25.0, tau_a=0.010, tau_b=0.020, tau_g=0.002,
                     c1=135.0, c2=108.0, c3=33.75, c4=33.75, c5=40.5, c6=13.5, c7=108.0)


def wendling_2002() -> Wendling:
    """Wendling et al. (2002) connectivity (C5 = 0.3C, C6 = 0.1C, C7 = 0.8C) with
    reduced dendritic inhibition, a regime in which fast somatic inhibition is active."""
    return Wendling(**WENDLING_2002)


FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "jansen_rit": {
        "factory": JansenRit,
        "drive_range": (0.0, 350.0),
        "physiological": ("gain_e", "gain_i", "a", "b", "c1", "c2", "c3", "c4"),
    },
    "wendling": {
        "factory": Wendling,
        "drive_range": (0.0, 300.0),
        "physiological": (
            "gain_e", "gain_s", "gain_f", "tau_a", "tau_b", "tau_g",
            "c1", "c2", "c3", "c4", "c5", "c6", "c7",
        ),
    },
    "wendling_2002": {
        "factory": wendling_2002,
        "drive_range": (30.0, 150.0),
        "physiological": (
            "gain_e", "gain_s", "gain_f", "tau_a", "tau_b", "tau_g",
            "c1", "c2", "c3", "c4", "c5", "c6", "c7",
        ),
    },
    "next_generation": {
        "factory": NextGenerationEI,
        "drive_range": (-8.0, 6.0),
        "physiological": (
            "tau_e", "tau_i", "tau_se", "tau_si", "delta_e", "delta_i",
            "j_ee", "j_ei", "j_ie", "j_ii", "eta_i",
        ),
    },
}


def second_drive(model) -> np.ndarray:
    """Input direction of an independent drive to the inhibitory population(s)."""
    if model.name == "Jansen-Rit":
        vec = np.zeros(6)
        vec[5] = model.gain_i * model.b
    elif model.name == "Wendling-Chauvel":
        # slow inhibitory interneurons project to pyramidal cells (y2, gain C4)
        # and to fast interneurons (y4), so their input enters both equations
        vec = np.zeros(10)
        vec[7] = model.gain_s / model.tau_b * model.c4
        vec[9] = model.gain_s / model.tau_b
    else:
        vec = np.zeros(6)
        vec[3] = 1.0 / model.tau_i
    return vec


def minimal_realisation(model, a, b, h):
    if model.name == "Wendling-Chauvel":
        from neural_observability.theory.linear_families import reduce_wendling

        return reduce_wendling(a, b, h)
    return a, b, h, np.eye(len(a))


def _normalised_column(model, x_star):
    """Linear column with the drive scaled to unit output variance."""
    col = linear_column(model, x_star)
    a, b, h = col.matrices()
    p = V.stationary_covariance(a, b)
    return col, a, b / np.sqrt(h @ p @ h), h


def sample_operating_points(family: str, count: int, seed: int, jitter_sd: float, min_damping: float,
                            record: dict | None = None):
    """Draw jittered parameter sets and drives; keep stable fixed points.

    ``record`` (if given) receives counts of attempts and exclusions.
    """
    spec = FAMILY_SPECS[family]
    base = spec["factory"]()
    rng = np.random.default_rng(seed)
    lo, hi = spec["drive_range"]
    out = []
    counts = {"attempts": 0, "root_finding_failed": 0, "no_fixed_point": 0, "no_stable_fixed_point": 0}
    while len(out) < count and counts["attempts"] < 50 * count:
        counts["attempts"] += 1
        changes = {}
        for name in spec["physiological"]:
            value = getattr(base, name)
            factor = np.exp(jitter_sd * rng.standard_normal())
            if name == "eta_i":  # additive jitter for a signed excitability
                changes[name] = float(value + 5.0 * jitter_sd * rng.standard_normal())
            else:
                changes[name] = float(value * factor)
        model = replace(base, **changes)
        drive = float(rng.uniform(lo, hi))
        try:
            equilibria = model.equilibria(drive)
        except Exception:  # noqa: BLE001 - root finding failures are counted and skipped
            counts["root_finding_failed"] += 1
            continue
        if not equilibria:
            counts["no_fixed_point"] += 1
            continue
        for index, x_star in enumerate(equilibria):
            a = model.jacobian(x_star)
            eig = np.linalg.eigvals(a)
            if eig.real.max() < -min_damping:
                out.append({"model": model, "drive": drive, "branch": index, "x_star": x_star})
                break
        else:
            counts["no_stable_fixed_point"] += 1
    counts["accepted"] = len(out)
    if record is not None:
        record.update(counts)
    return out


def _closest_equilibrium(model, drive, reference):
    try:
        eqs = model.equilibria(drive)
    except Exception:  # noqa: BLE001
        return None
    if not eqs:
        return None
    scale = np.abs(reference) + 1e-9
    distances = [np.max(np.abs(e - reference) / scale) for e in eqs]
    return eqs[int(np.argmin(distances))]


def _snr_r(snr_db: float) -> float:
    return 1.0 / (200.0 * 10 ** (snr_db / 10.0))


def two_drive_recoverability(model, x_star, fractions, snrs_db) -> dict[str, Any]:
    """Filtering recoverability with an independent inhibitory drive.

    The second drive is scaled so that it contributes ``fraction`` of the
    output variance; the total output variance is one.
    """
    a = model.jacobian(x_star)
    b1, b2, h = model.input_vector(), second_drive(model), model.output_vector()
    v1 = h @ V.stationary_covariance(a, b1) @ h
    v2 = h @ V.stationary_covariance(a, b2) @ h
    out: dict[str, Any] = {}
    for frac in fractions:
        if frac == 0.0:
            bb = (b1 / np.sqrt(v1))[:, None]
        else:
            bb = np.column_stack([b1 * np.sqrt((1 - frac) / v1), b2 * np.sqrt(frac / v2)])
        rows = []
        for snr in snrs_db:
            rows.append(_safe_filtered_r2(a, bb, h, snr))
        out[f"{frac:g}"] = rows
    return out


def _safe_filtered_r2(a, b, h, snr_db):
    """Filtering recoverability, or None where the Riccati solver fails (very high SNR)."""
    try:
        return V.stationary_recoverability(a, b, h, _snr_r(snr_db)).physical_r2("filtered").tolist()
    except (ValueError, np.linalg.LinAlgError):
        return None


def analyse_point(point: dict[str, Any], config: dict[str, Any], seed: int, family: str) -> dict[str, Any]:
    model, drive, x_star = point["model"], point["drive"], point["x_star"]
    rng = np.random.default_rng(seed)
    col, a, b, h = _normalised_column(model, x_star)
    n = len(a)
    eig = np.linalg.eigvals(a)
    am, bm, hm, _ = minimal_realisation(model, a, b, h)
    zeros = V.transmission_zeros(am, bm, hm)
    hsv_fwd = V.hankel_singular_values(am, bm, hm)
    p_full = V.stationary_covariance(a, b)
    p_min = V.stationary_covariance(am, bm)
    sd = np.sqrt(np.diag(p_min))
    record: dict[str, Any] = {
        "family": family,
        "drive": drive,
        "branch": point["branch"],
        "spectral_abscissa": float(eig.real.max()),
        "dominant_frequency_hz": float(np.abs(eig.imag[np.argmax(eig.real)]) / (2 * np.pi)),
        "max_zero_real_part": float(zeros.real.max()) if len(zeros) else float("nan"),
        "minimum_phase": bool(len(zeros) == 0 or zeros.real.max() < 0),
        "hsv_forward_rel": (hsv_fwd / hsv_fwd[0]).tolist(),
        "state_dimension": n,
        "minimal_dimension": len(am),
        "minimal_correlation_floor": float(np.linalg.eigvalsh(p_min / np.outer(sd, sd)).min()),
        "output_share": [float(h[j] ** 2 * p_full[j, j]) for j in range(n)],
        "state_variance": np.diag(p_full).tolist(),
    }
    # recoverability curves
    curves = {"filtered": [], "smoothed": [], "future_only": [], "output_filtered": [], "output_smoothed": []}
    for snr_db in config["snr_db"]:
        rec = V.stationary_recoverability(a, b, h, _snr_r(snr_db))
        for key in ("filtered", "smoothed", "future_only"):
            curves[key].append(rec.physical_r2(key).tolist())
        curves["output_filtered"].append(V.output_r2(rec, h, "filtered"))
        curves["output_smoothed"].append(V.output_r2(rec, h, "smoothed"))
    record["curves"] = curves
    # high-SNR convergence of exact-model filtering (for theorem 2)
    record["filtered_high_snr"] = {f"{snr:g}": _safe_filtered_r2(a, b, h, snr) for snr in config["high_snr_db"]}
    # an independent second drive to the inhibitory population
    record["two_drive"] = two_drive_recoverability(model, x_star, config["second_drive_fractions"],
                                                   config["second_drive_snr_db"])
    # physiological mismatch floors
    spec = FAMILY_SPECS[family]
    floors: dict[str, list[list[float]]] = {}
    exclusions: dict[str, dict[str, int]] = {}
    checks: dict[str, Any] = {}
    for level in config["mismatch_levels"]:
        rows = []
        excl = {"fixed_point_lost": 0, "unstable": 0, "numerical": 0}
        truth_min_phase = []
        for _ in range(config["mismatch_draws"]):
            signs = rng.choice([-1.0, 1.0], size=len(spec["physiological"]))
            changes = {}
            for name, sign in zip(spec["physiological"], signs):
                value = getattr(model, name)
                changes[name] = float(value * (1.0 + level * sign)) if name != "eta_i" else float(value + level * sign * abs(value))
            truth = replace(model, **changes)
            xt = _closest_equilibrium(truth, drive, x_star)
            if xt is None:
                excl["fixed_point_lost"] += 1
                continue
            at = truth.jacobian(xt)
            if np.linalg.eigvals(at).real.max() >= -1e-3:
                excl["unstable"] += 1
                continue
            try:
                _, a_t, b_t, h_t = _normalised_column(truth, xt)
                floor = V.high_snr_floor(a_t, b_t, a, b, h_t, h_nom=h)
            except (ValueError, np.linalg.LinAlgError):
                excl["numerical"] += 1
                continue
            if not np.all(np.isfinite(floor)):
                excl["numerical"] += 1
                continue
            amt, bmt, hmt, _ = minimal_realisation(truth, a_t, b_t, h_t)
            zt = V.transmission_zeros(amt, bmt, hmt)
            truth_min_phase.append(bool(len(zt) == 0 or zt.real.max() < 0))
            rows.append(floor.tolist())
            if len(rows) == 1:
                # exact mismatched filter at increasing SNR and a quadrature check
                chk: dict[str, Any] = {"formula_loss": floor.tolist()}
                for snr_db in config["floor_check_snr_db"]:
                    try:
                        res = V.mismatched_filter_error(a_t, b_t, a, b, h_t, _snr_r(snr_db), h_nom=h)
                        chk[f"{snr_db:g}dB"] = {"loss": (1 - res["r2"]).tolist(), "output_r2": res["output_r2"]}
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        chk[f"{snr_db:g}dB"] = {"error": repr(exc)}
                chk["quadrature"] = {
                    "wmax_x4": V.high_snr_floor(a_t, b_t, a, b, h_t, h_nom=h, omega_max=2 * np.pi * 8000.0,
                                                points=16000).tolist(),
                    "wmax_x16": V.high_snr_floor(a_t, b_t, a, b, h_t, h_nom=h, omega_max=2 * np.pi * 32000.0,
                                                 points=32000).tolist(),
                }
                checks[f"{level:g}"] = chk
        floors[f"{level:g}"] = rows
        exclusions[f"{level:g}"] = excl
        record.setdefault("truth_minimum_phase_fraction", {})[f"{level:g}"] = (
            float(np.mean(truth_min_phase)) if truth_min_phase else None)
    record["mismatch_floors"] = floors
    record["mismatch_exclusions"] = exclusions
    record["floor_checks"] = checks
    # Bayesian verification loss
    va = verification_analysis(col, snr_db=config["verification_snr_db"])
    fisher = va["fisher_per_second"]
    theta0 = col.log_theta
    prior_precision = np.eye(len(theta0)) / config["prior_log_sd"] ** 2
    losses = {}
    for t_sec in config["record_seconds"]:
        cov = np.linalg.inv(prior_precision + t_sec * fisher)
        cov = 0.5 * (cov + cov.T)
        chol = np.linalg.cholesky(cov + 1e-15 * np.eye(len(cov)))
        draws, rejected = [], {"unstable": 0, "numerical": 0}
        for _ in range(config["verification_draws"]):
            delta = chol @ rng.standard_normal(len(theta0))
            at, bt, ht = col.matrices(theta0 + delta)
            if np.linalg.eigvals(at).real.max() >= -1e-3:
                rejected["unstable"] += 1
                continue
            pt = V.stationary_covariance(at, bt)
            out_var = float(ht @ pt @ ht)
            if not np.isfinite(out_var) or out_var <= 0:
                rejected["numerical"] += 1
                continue
            loss = V.high_snr_floor(at, bt / np.sqrt(out_var), a, b, ht, h_nom=h)
            if np.all(np.isfinite(loss)):
                draws.append(loss)
            else:
                rejected["numerical"] += 1
        losses[f"{t_sec:g}"] = {
            "mean_loss": np.mean(draws, axis=0).tolist() if draws else None,
            "valid_draws": len(draws),
            "rejected": rejected,
        }
    record["verification"] = {
        "fisher_eigenvalues_per_s": va["fisher_eigenvalues"].tolist(),
        "structural_null_leak": va["null_leak"].tolist(),
        "tau_s": [float(x) for x in va["verification_time_s"]],
        "loss_by_record_seconds": losses,
        "parameter_labels": list(col.labels),
    }
    # physiological realisability of every linear-parameter direction
    try:
        from neural_observability.theory.symmetry import physiological_jacobian

        jp, _ = physiological_jacobian(model, drive, x_star, linear_column)
        sv = np.linalg.svd(jp, compute_uv=False)
        record["physiological_rank"] = int(np.sum(sv > 1e-8 * sv[0]))
        record["linear_parameter_count"] = int(jp.shape[0])
    except Exception as exc:  # noqa: BLE001
        record["physiological_rank_error"] = repr(exc)
    return record
