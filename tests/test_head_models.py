from __future__ import annotations

import unittest

import numpy as np

from neural_observability.head_models.template import average_reference_basis


class ForwardTests(unittest.TestCase):
    def test_average_reference_basis(self) -> None:
        for channel_count in (2, 19, 64):
            basis = average_reference_basis(channel_count)
            self.assertEqual(basis.shape, (channel_count, channel_count - 1))
            np.testing.assert_allclose(
                basis.T @ basis, np.eye(channel_count - 1), atol=1e-12
            )
            np.testing.assert_allclose(np.ones(channel_count) @ basis, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
