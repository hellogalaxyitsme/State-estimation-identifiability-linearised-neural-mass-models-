"""Dependency-light deterministic EEG reference-coordinate utilities."""

from __future__ import annotations

import numpy as np


def average_reference_basis(channel_count: int) -> np.ndarray:
    """Return a deterministic orthonormal basis for zero-sum sensor vectors."""
    if channel_count < 2:
        raise ValueError("Average reference requires at least two channels")
    basis = np.zeros((channel_count, channel_count - 1), dtype=float)
    for column in range(channel_count - 1):
        k = column + 1
        denominator = np.sqrt(k * (k + 1.0))
        basis[:k, column] = 1.0 / denominator
        basis[k, column] = -k / denominator
    if not np.allclose(basis.T @ basis, np.eye(channel_count - 1), atol=1e-12):
        raise RuntimeError("Internal error constructing average-reference basis")
    if not np.allclose(np.ones(channel_count) @ basis, 0.0, atol=1e-12):
        raise RuntimeError("Average-reference basis does not have zero column sums")
    return basis

