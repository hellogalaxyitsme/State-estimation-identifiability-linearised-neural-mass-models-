from __future__ import annotations

import unittest

import numpy as np

from neural_observability.theory.network import (
    JRParameters,
    build_network_linearization,
    local_jacobian,
    select_equilibrium,
    single_column_equilibria,
)


class ModelTests(unittest.TestCase):
    def test_standard_column_has_three_equilibria_at_low_drive(self) -> None:
        equilibria = single_column_equilibria(50.0)
        self.assertEqual(len(equilibria), 3)
        low = select_equilibrium(50.0, "lowest")
        high = select_equilibrium(50.0, "highest")
        self.assertLess(low[0], high[0])
        self.assertLess(np.max(np.linalg.eigvals(local_jacobian(low)).real), 0.0)

    def test_coupled_network_equilibrium_and_dimensions(self) -> None:
        positions = np.array(
            [[0.0, 0.0, 0.06], [0.04, 0.0, 0.04], [-0.04, 0.0, 0.04], [0.0, 0.04, 0.04]]
        )
        scenario = {
            "name": "test",
            "p0_per_second": 50.0,
            "equilibrium_branch": "lowest",
            "coupling_gain": 5.0,
            "drive_relative_sd": 0.05,
        }
        graph = {"nearest_neighbors": 2, "length_scale_m": 0.04}
        a, h, equilibrium, metadata = build_network_linearization(
            positions, scenario, graph, np.random.default_rng(11), JRParameters()
        )
        self.assertEqual(a.shape, (24, 24))
        self.assertEqual(h.shape, (4, 24))
        self.assertEqual(equilibrium.shape, (4, 3))
        self.assertLess(metadata["equilibrium_residual_inf"], 1e-8)
        self.assertTrue(np.isfinite(metadata["spectral_abscissa_per_second"]))


if __name__ == "__main__":
    unittest.main()
