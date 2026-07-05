from __future__ import annotations

import numpy as np

from neureptrace.decoding.label_proportions import adjust_probabilities_to_label_proportions


def test_label_proportion_calibration_handles_rows_on_zero_prior_class() -> None:
    probabilities = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    result = adjust_probabilities_to_label_proportions(
        probabilities,
        [1.0, 0.0],
        max_iter=5,
        tol=1e-12,
    )

    assert result.converged
    np.testing.assert_allclose(result.probabilities, np.array([[1.0, 0.0], [1.0, 0.0]]), atol=1e-10)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(result.target_proportions, (1.0, 0.0))
