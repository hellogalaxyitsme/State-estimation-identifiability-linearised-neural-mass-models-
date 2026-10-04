from __future__ import annotations

import numpy as np

from neural_observability.stimulation.cohort import mean_spatial_squared_cosine
from neural_observability.stimulation.source_matching import (
    characterize_sources,
    hierarchical_lookup_permutation_tests,
    score_constant_topographies,
    select_feature_matches,
    select_spatial_neighbours,
    subject_bootstrap_mean_interval,
)


def _source_fixture(source_count: int = 14):
    rng = np.random.default_rng(9)
    labels = [f"S{index:02d}" for index in range(source_count)]
    lead = rng.normal(size=(19, source_count))
    positions = rng.normal(size=(source_count, 3)) * 20
    positions[:, 2] += 50
    orientations = rng.normal(size=(source_count, 3))
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    depths = np.linspace(3.0, 30.0, source_count)
    return labels, lead, positions, orientations, depths


def test_characterization_uses_absolute_radiality_and_population_z_scores() -> None:
    labels, lead, positions, orientations, depths = _source_fixture()
    result = characterize_sources(
        lead,
        labels,
        positions,
        orientations,
        depths,
        np.zeros(3),
        set(labels),
    )
    standardized = np.asarray(result["standardized_features"])
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(standardized.std(axis=0, ddof=0), 1.0, atol=1e-14)
    radiality = np.asarray(result["raw_features"])[:, 2]
    assert np.all((radiality >= 0) & (radiality <= 1))

    reversed_result = characterize_sources(
        lead,
        labels,
        positions,
        -orientations,
        depths,
        np.zeros(3),
        set(labels),
    )
    np.testing.assert_allclose(
        result["raw_features"], reversed_result["raw_features"], rtol=0, atol=0
    )


def test_feature_matches_are_exactly_ten_unique_wrong_sources_and_deterministic() -> None:
    labels, lead, positions, orientations, depths = _source_fixture()
    features = characterize_sources(
        lead, labels, positions, orientations, depths, np.zeros(3), set(labels)
    )
    first = select_feature_matches(
        labels, np.asarray(features["standardized_features"]), labels[:3], 10
    )
    second = select_feature_matches(
        labels, np.asarray(features["standardized_features"]), labels[:3], 10
    )
    assert first == second
    for target, matches in first.items():
        assert len(matches) == 10
        assert target not in {label for label, _ in matches}
        assert len({label for label, _ in matches}) == 10
        assert [value for _, value in matches] == sorted(value for _, value in matches)


def test_spatial_neighbour_uses_only_other_observed_stimulation_sites() -> None:
    labels = ["A", "B", "C", "D"]
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [3, 0, 0], [0.1, 0, 0]], dtype=float)
    neighbours = select_spatial_neighbours(labels, positions, ["A", "B", "C"])
    assert neighbours["A"] == ("B", 1.0)
    assert neighbours["B"] == ("A", 1.0)
    assert neighbours["C"] == ("B", 2.0)
    assert all(value[0] != "D" for value in neighbours.values())


def test_vectorized_scores_exactly_reproduce_the_scalar_definition() -> None:
    rng = np.random.default_rng(17)
    lead = rng.normal(size=(31, 12))
    observation = rng.normal(size=(31, 15))
    vectorized = score_constant_topographies(lead, observation)
    scalar = np.asarray(
        [
            mean_spatial_squared_cosine(
                np.repeat(lead[:, index, None], observation.shape[1], axis=1),
                observation,
            )
            for index in range(lead.shape[1])
        ]
    )
    np.testing.assert_allclose(vectorized, scalar, rtol=2e-15, atol=2e-15)


def test_joint_within_subject_label_permutation_and_bootstrap_are_deterministic() -> None:
    labels = [np.asarray([0, 1, 0, 1]), np.asarray([0, 1, 0, 1])]
    strong = np.asarray([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
    weak = strong * 0.25
    lookup = {
        "strong": [strong, strong + 0.1],
        "weak": [weak, weak + 0.05],
    }
    first = hierarchical_lookup_permutation_tests(lookup, labels, 500, 31, batch_size=37)
    second = hierarchical_lookup_permutation_tests(lookup, labels, 500, 31, batch_size=37)
    for name in lookup:
        np.testing.assert_allclose(first[name]["subject_effects"], second[name]["subject_effects"])
        assert first[name]["one_sided_p_value"] == second[name]["one_sided_p_value"]
        assert first[name]["observed_mean_subject_effect"] > 0
        assert first[name]["one_sided_p_value"] < 0.1
    interval1 = subject_bootstrap_mean_interval([0.1, 0.2, 0.3], 100, 7)
    interval2 = subject_bootstrap_mean_interval([0.1, 0.2, 0.3], 100, 7)
    assert interval1 == interval2

