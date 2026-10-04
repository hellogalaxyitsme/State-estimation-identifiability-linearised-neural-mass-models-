from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm

from neural_observability.theory.information import (
    expected_standardized_mse,
    information_eigenvalues,
    sampled_dynamic_map,
    symmetric_psd_sqrt,
)


class InformationTests(unittest.TestCase):
    def test_sampled_dynamic_map_matches_direct_exponential(self) -> None:
        a = np.array([[-1.0, 0.3], [0.0, -2.0]])
        c = np.array([[1.0, -0.5]])
        times = np.linspace(0.0, 0.2, 6)
        expected = np.vstack([c @ expm(a * time) for time in times])
        actual = sampled_dynamic_map(a, c, times)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

    def test_invariant_eigenvalues_survive_coordinate_change(self) -> None:
        rng = np.random.default_rng(7)
        dmap = rng.normal(size=(9, 4))
        p = np.diag([1.0, 2.0, 0.4, 3.0])
        p_sqrt = symmetric_psd_sqrt(p)
        transform = np.diag([0.01, 0.2, 5.0, 100.0])
        inverse = np.linalg.inv(transform)
        p_z = inverse @ p @ inverse.T
        values_x = information_eigenvalues(dmap, p_sqrt)
        values_z = information_eigenvalues(
            dmap @ transform, symmetric_psd_sqrt(p_z)
        )
        np.testing.assert_allclose(values_x, values_z, rtol=1e-10, atol=1e-10)

    def test_matched_mse_equals_posterior_variance(self) -> None:
        rng = np.random.default_rng(9)
        dmap = rng.normal(size=(12, 5))
        p_sqrt = np.diag([1.0, 0.7, 2.0, 1.2, 0.5])
        result = expected_standardized_mse(dmap, dmap, p_sqrt)
        self.assertAlmostEqual(
            result["expected_standardized_mse"],
            result["assumed_normalized_posterior_variance"],
            places=11,
        )
        self.assertAlmostEqual(result["mismatch_calibration_ratio"], 1.0, places=11)


if __name__ == "__main__":
    unittest.main()
