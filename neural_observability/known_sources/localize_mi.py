"""Absolute visibility of known intracerebral current dipoles (Localize-MI)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from neural_observability.head_models.referencing import average_reference_basis


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_session(description: str) -> tuple[str, str, float]:
    """'Stimulation of channel K13-14 1mA' -> ('K13', 'K14', 0.001)."""
    match = re.search(r"channel\s+([A-Za-z]+'?)(\d+)-(\d+)\s+([\d.]+)\s*mA", description)
    if not match:
        raise ValueError(f"Cannot parse session description: {description!r}")
    shaft, first, second, current = match.groups()
    return f"{shaft}{first}", f"{shaft}{second}", float(current) * 1e-3


def oas_covariance(samples: np.ndarray) -> np.ndarray:
    """Oracle-approximating shrinkage covariance of rows-as-observations data."""
    x = samples - samples.mean(0, keepdims=True)
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    alpha = np.mean(s**2)
    num = alpha + mu**2
    den = (n + 1.0) * (alpha - mu**2 / p)
    shrink = 1.0 if den == 0 else min(1.0, num / den)
    return (1.0 - shrink) * s + shrink * mu * np.eye(p)


def write_freesurfer_bem(surface_dir: Path, subject: str, subjects_dir: Path) -> None:
    import nibabel as nib

    bem_dir = subjects_dir / subject / "bem"
    bem_dir.mkdir(parents=True, exist_ok=True)
    for name in ("inner_skull", "outer_skull", "outer_skin"):
        target = bem_dir / f"{name}.surf"
        if target.exists():
            continue
        gii = nib.load(str(surface_dir / f"{subject}_{name}.surf.gii"))
        coords = np.asarray(gii.darrays[0].data, dtype=float)
        tris = np.asarray(gii.darrays[1].data, dtype=np.int32)
        nib.freesurfer.write_geometry(str(target), coords, tris)


def session_field(epochs: np.ndarray, good: np.ndarray, sfreq: float, zero_time: float,
                  peak_window: tuple[float, float], baseline: tuple[float, float]):
    """Return the average-referenced peak field, its noise covariance and metadata."""
    data = epochs[:, good, :]
    basis = average_reference_basis(int(good.sum()))
    times = np.arange(data.shape[-1]) / sfreq - zero_time
    base_mask = (times >= baseline[0]) & (times < baseline[1])
    peak_mask = (times >= peak_window[0]) & (times <= peak_window[1])
    average = data.mean(0)
    projected = basis.T @ average
    energy = np.sum(projected[:, peak_mask] ** 2, axis=0)
    peak_index = np.flatnonzero(peak_mask)[int(np.argmax(energy))]
    y_peak = projected[:, peak_index]
    baseline_samples = np.einsum("ck,tcs->tsk", basis, data[:, :, base_mask]).reshape(-1, basis.shape[1])
    sigma_sample = oas_covariance(baseline_samples)
    sigma_mean = sigma_sample / data.shape[0]
    return y_peak, sigma_mean, basis, float(times[peak_index]), int(data.shape[0])


def whitened_energy(vector: np.ndarray, covariance: np.ndarray) -> float:
    return float(vector @ np.linalg.solve(covariance, vector))


def squared_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))
