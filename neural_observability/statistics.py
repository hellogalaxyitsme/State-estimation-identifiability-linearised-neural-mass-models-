"""Subject-level inference shared by the empirical analyses."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Sequence

import numpy as np


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_subject_sign_flip_test(differences: Sequence[float]) -> dict[str, float | int]:
    """Exact one-sided sign-flip test for a positive equal-subject mean."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a nonempty finite vector")
    if len(values) > 20:
        raise ValueError("Exact enumeration is limited to 20 subjects")
    observed = float(np.mean(values))
    extreme = 0
    total = 2 ** len(values)
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = float(np.mean(values * np.asarray(signs)))
        if permuted >= observed - tolerance:
            extreme += 1
    return {
        "subject_count": int(len(values)),
        "observed_mean": observed,
        "extreme_sign_patterns": int(extreme),
        "total_sign_patterns": int(total),
        "one_sided_exact_p": float(extreme / total),
    }


def subject_bootstrap_mean_interval(
    values: Sequence[float], resamples: int, seed: int
) -> list[float]:
    """Return a fixed-seed subject bootstrap percentile interval for a mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("At least two finite subject values are required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    batch_size = 4096
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]
