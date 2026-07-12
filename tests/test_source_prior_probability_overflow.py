from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior


def test_source_prior_normalizes_large_finite_probability_rows_without_overflow() -> None:
    probabilities = np.asarray([[1e308, 1e308], [1e308, 5e307]], dtype=float)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = adjust_probabilities_to_source_prior(
            probabilities,
            source_labels=[0, 1],
            classes=[0, 1],
            config={"target_prior": "source"},
        )

    assert np.all(np.isfinite(result.probabilities))
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.probabilities[0], np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(result.probabilities[1], np.asarray([2.0 / 3.0, 1.0 / 3.0]))
