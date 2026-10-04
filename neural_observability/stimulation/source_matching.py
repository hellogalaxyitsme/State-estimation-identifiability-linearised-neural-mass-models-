"""Source characterisation, matching, scoring and inference for the matched-source analysis."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "source_depth_mm",
    "log_lead_field_column_norm",
    "absolute_dipole_radiality",
)


def characterize_sources(
    lead_field: np.ndarray,
    source_labels: Sequence[str],
    source_positions_mm: np.ndarray,
    source_orientations: np.ndarray,
    source_depth_mm: np.ndarray,
    scalp_center_mm: np.ndarray,
    eligible_labels: set[str],
) -> dict[str, np.ndarray | list[str]]:
    """Return raw and within-subject standardized source features."""
    lead = np.asarray(lead_field, dtype=float)
    labels = [str(label) for label in source_labels]
    positions = np.asarray(source_positions_mm, dtype=float)
    orientations = np.asarray(source_orientations, dtype=float)
    depths = np.asarray(source_depth_mm, dtype=float)
    center = np.asarray(scalp_center_mm, dtype=float)
    source_count = len(labels)
    if len(set(labels)) != source_count:
        raise ValueError("Source labels must be unique")
    if lead.ndim != 2 or lead.shape[1] != source_count:
        raise ValueError("Lead field must have one column per source")
    if positions.shape != (source_count, 3) or orientations.shape != (source_count, 3):
        raise ValueError("Source positions and orientations must be source x 3")
    if depths.shape != (source_count,) or center.shape != (3,):
        raise ValueError("Invalid depth or scalp-centre dimensions")
    if not all(
        np.all(np.isfinite(value))
        for value in (lead, positions, orientations, depths, center)
    ):
        raise ValueError("Source characterization inputs must be finite")
    indices = np.asarray(
        [index for index, label in enumerate(labels) if label in eligible_labels],
        dtype=int,
    )
    if len(indices) < 11:
        raise ValueError("At least eleven eligible sources are required")
    selected_labels = [labels[index] for index in indices]
    missing = set(eligible_labels) - set(selected_labels)
    if missing:
        raise ValueError(f"Eligible labels are absent from the lead field: {sorted(missing)}")

    columns = lead[:, indices]
    column_norms = np.linalg.norm(columns, axis=0)
    radial_vectors = positions[indices] - center[None, :]
    radial_norms = np.linalg.norm(radial_vectors, axis=1)
    orientation_norms = np.linalg.norm(orientations[indices], axis=1)
    tiny = np.finfo(float).tiny
    if np.any(column_norms <= tiny):
        raise ValueError("Every eligible lead-field column must be nonzero")
    if np.any(radial_norms <= tiny) or np.any(orientation_norms <= tiny):
        raise ValueError("Radial vectors and source orientations must be nonzero")
    radiality = np.abs(
        np.sum(orientations[indices] * radial_vectors, axis=1)
        / (orientation_norms * radial_norms)
    )
    radiality = np.clip(radiality, 0.0, 1.0)
    raw = np.column_stack(
        [depths[indices], np.log(column_norms), radiality]
    )
    means = raw.mean(axis=0)
    standard_deviations = raw.std(axis=0, ddof=0)
    if np.any(standard_deviations <= np.finfo(float).eps):
        raise ValueError("Every within-subject matching feature must vary")
    standardized = (raw - means[None, :]) / standard_deviations[None, :]
    return {
        "labels": selected_labels,
        "leadfield_indices": indices,
        "positions_mm": positions[indices],
        "column_norms": column_norms,
        "raw_features": raw,
        "feature_means": means,
        "feature_standard_deviations": standard_deviations,
        "standardized_features": standardized,
    }


def select_feature_matches(
    source_labels: Sequence[str],
    standardized_features: np.ndarray,
    target_labels: Sequence[str],
    match_count: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Select deterministic nearest wrong sources in standardized feature space."""
    labels = [str(label) for label in source_labels]
    features = np.asarray(standardized_features, dtype=float)
    if features.shape != (len(labels), 3) or not np.all(np.isfinite(features)):
        raise ValueError("Standardized features must be finite source x 3 values")
    if len(set(labels)) != len(labels) or match_count < 1:
        raise ValueError("Source labels must be unique and match_count positive")
    if len(labels) <= match_count:
        raise ValueError("The source pool is too small for the requested match count")
    lookup = {label: index for index, label in enumerate(labels)}
    output: dict[str, list[tuple[str, float]]] = {}
    for target in target_labels:
        target = str(target)
        if target not in lookup:
            raise ValueError(f"Target source is absent from the eligible pool: {target}")
        index = lookup[target]
        distances = np.linalg.norm(features - features[index], axis=1)
        ranked = sorted(
            (
                (labels[candidate], float(distances[candidate]))
                for candidate in range(len(labels))
                if candidate != index
            ),
            key=lambda item: (item[1], item[0]),
        )
        selected = ranked[:match_count]
        if len(selected) != match_count or len({label for label, _ in selected}) != match_count:
            raise RuntimeError("Feature matching did not produce unique controls")
        output[target] = selected
    return output


def select_spatial_neighbours(
    source_labels: Sequence[str],
    source_positions_mm: np.ndarray,
    stimulation_labels: Sequence[str],
) -> dict[str, tuple[str, float]]:
    """Select each stimulation site's nearest other observed stimulation site."""
    labels = [str(label) for label in source_labels]
    positions = np.asarray(source_positions_mm, dtype=float)
    stimulation = sorted({str(label) for label in stimulation_labels})
    if positions.shape != (len(labels), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("Source positions must be finite source x 3 values")
    if len(set(labels)) != len(labels) or len(stimulation) < 2:
        raise ValueError("Unique sources and at least two stimulation sites are required")
    lookup = {label: index for index, label in enumerate(labels)}
    if not set(stimulation) <= set(lookup):
        raise ValueError("A stimulation site is absent from the eligible source pool")
    output: dict[str, tuple[str, float]] = {}
    for target in stimulation:
        target_position = positions[lookup[target]]
        ranked = sorted(
            (
                (
                    candidate,
                    float(np.linalg.norm(positions[lookup[candidate]] - target_position)),
                )
                for candidate in stimulation
                if candidate != target
            ),
            key=lambda item: (item[1], item[0]),
        )
        output[target] = ranked[0]
    return output


def score_constant_topographies(
    lead_field: np.ndarray, observation: np.ndarray
) -> np.ndarray:
    """Vectorized mean average-reference squared-cosine source scores."""
    lead = np.asarray(lead_field, dtype=float)
    observed = np.asarray(observation, dtype=float)
    if lead.ndim != 2 or observed.ndim != 2 or lead.shape[0] != observed.shape[0]:
        raise ValueError("Lead field and observation must share the sensor dimension")
    if not np.all(np.isfinite(lead)) or not np.all(np.isfinite(observed)):
        raise ValueError("Topography scoring inputs must be finite")
    centered_lead = lead - lead.mean(axis=0, keepdims=True)
    centered_observed = observed - observed.mean(axis=0, keepdims=True)
    lead_energy = np.sum(centered_lead**2, axis=0)
    observed_energy = np.sum(centered_observed**2, axis=0)
    valid_time = observed_energy > np.finfo(float).tiny
    if not np.any(valid_time) or np.any(lead_energy <= np.finfo(float).tiny):
        raise ValueError("No nondegenerate spatial source or time sample")
    inner = centered_lead.T @ centered_observed[:, valid_time]
    scores = np.mean(
        inner**2 / (lead_energy[:, None] * observed_energy[valid_time][None, :]),
        axis=1,
    )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("A source topographic score is non-finite")
    return scores


def subject_bootstrap_mean_interval(
    subject_effects: Sequence[float],
    resamples: int,
    seed: int,
    quantiles: tuple[float, float] = (0.025, 0.975),
    batch_size: int = 2000,
) -> list[float]:
    """Percentile interval for the equal-subject mean."""
    values = np.asarray(subject_effects, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("At least two finite subject effects are required")
    if resamples < 1 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=float)
    for start in range(0, int(resamples), batch_size):
        stop = min(int(resamples), start + batch_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, quantiles)]


def hierarchical_lookup_permutation_tests(
    lookup_by_control: Mapping[str, Sequence[np.ndarray]],
    observed_label_indices: Sequence[np.ndarray],
    permutations: int,
    seed: int,
    batch_size: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Joint within-subject label permutations for one or more fixed controls.

    Each lookup matrix is runs x candidate stimulation labels. Entry ``(r, j)`` is
    the effect for run ``r`` if label ``j`` were assigned as its source. The same
    shuffled label assignments are used for every control.
    """
    names = list(lookup_by_control)
    if not names or not observed_label_indices or permutations < 1:
        raise ValueError("Nonempty controls, subjects, and permutations are required")
    subject_count = len(observed_label_indices)
    matrices: dict[str, list[np.ndarray]] = {}
    for name, source in lookup_by_control.items():
        if len(source) != subject_count:
            raise ValueError("Every control must contain every subject")
        matrices[name] = [np.asarray(value, dtype=float) for value in source]
    labels = [np.asarray(value, dtype=int) for value in observed_label_indices]
    observed_subject: dict[str, np.ndarray] = {
        name: np.empty(subject_count, dtype=float) for name in names
    }
    for subject_index, label_vector in enumerate(labels):
        if label_vector.ndim != 1 or len(label_vector) < 1:
            raise ValueError("Each subject must contain at least one run label")
        for name in names:
            matrix = matrices[name][subject_index]
            if matrix.ndim != 2 or matrix.shape[0] != len(label_vector):
                raise ValueError("Lookup matrices must be runs x candidate labels")
            if (
                np.any(label_vector < 0)
                or np.any(label_vector >= matrix.shape[1])
                or not np.all(np.isfinite(matrix))
            ):
                raise ValueError("Invalid lookup values or observed labels")
            observed_subject[name][subject_index] = float(
                np.mean(matrix[np.arange(len(label_vector)), label_vector])
            )

    observed = {name: float(values.mean()) for name, values in observed_subject.items()}
    exceedances = {name: 0 for name in names}
    null_sum = {name: 0.0 for name in names}
    null_sum_squares = {name: 0.0 for name in names}
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < int(permutations):
        count = min(batch_size, int(permutations) - completed)
        batch_statistics = {name: np.zeros(count, dtype=float) for name in names}
        for subject_index, label_vector in enumerate(labels):
            order = np.argsort(
                rng.random((count, len(label_vector))), axis=1, kind="stable"
            )
            permuted_labels = label_vector[order]
            row_indices = np.arange(len(label_vector))[None, :]
            for name in names:
                selected = matrices[name][subject_index][row_indices, permuted_labels]
                batch_statistics[name] += selected.mean(axis=1) / subject_count
        for name in names:
            statistics = batch_statistics[name]
            exceedances[name] += int(np.sum(statistics >= observed[name]))
            null_sum[name] += float(np.sum(statistics))
            null_sum_squares[name] += float(np.sum(statistics**2))
        completed += count

    output: dict[str, dict[str, Any]] = {}
    for name in names:
        null_mean = null_sum[name] / permutations
        null_variance = max(
            0.0, null_sum_squares[name] / permutations - null_mean**2
        )
        effects = observed_subject[name]
        output[name] = {
            "subject_count": int(subject_count),
            "observed_mean_subject_effect": observed[name],
            "median_subject_effect": float(np.median(effects)),
            "positive_subject_fraction": float(np.mean(effects > 0)),
            "null_mean": float(null_mean),
            "null_sd": float(np.sqrt(null_variance)),
            "permutation_count": int(permutations),
            "one_sided_p_value": float(
                (exceedances[name] + 1) / (int(permutations) + 1)
            ),
            "subject_effects": effects,
        }
    return output
