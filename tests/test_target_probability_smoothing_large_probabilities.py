from __future__ import annotations

import numpy as np

from neureptrace.decoding.target_probability_smoothing import row_normalize, smooth_target_probabilities


def test_row_normalize_handles_overflowing_finite_row_sums() -> None:
    matrix = np.asarray([[1e308, 1e308], [1e308, 0.0]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        normalized = row_normalize(matrix)

    np.testing.assert_allclose(normalized, np.asarray([[0.5, 0.5], [1.0, 0.0]]))
    np.testing.assert_allclose(normalized.sum(axis=1), np.ones(2))


def test_smoothing_normalizes_large_finite_probability_rows() -> None:
    probabilities = np.asarray([[1e308, 1e308], [1e308, 2e307]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = smooth_target_probabilities(
            [[0.0], [1.0]],
            probabilities,
            config={"alpha": 0.0, "standardize": False},
        )

    expected = np.asarray([[0.5, 0.5], [5.0 / 6.0, 1.0 / 6.0]], dtype=np.float32)
    np.testing.assert_allclose(result.initial_probabilities, expected)
    np.testing.assert_allclose(result.probabilities, expected)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(2))
