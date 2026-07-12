from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence_filter import confidence_filter, probability_entropy


def test_confidence_filter_normalizes_large_finite_rows_without_overflow() -> None:
    probabilities = np.asarray([[1.0e308, 1.0e308], [1.0e308, 5.0e307]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = confidence_filter(probabilities, min_confidence=0.6)

    np.testing.assert_allclose(result.confidence, [0.5, 2.0 / 3.0])
    np.testing.assert_allclose(result.margin, [0.0, 1.0 / 3.0])
    assert result.predicted_index.tolist() == [0, 0]
    assert result.accepted_mask.tolist() == [False, True]
    assert np.all(np.isfinite(result.confidence))
    assert np.all(np.isfinite(result.margin))


def test_probability_entropy_preserves_large_finite_score_ratios() -> None:
    probabilities = np.asarray([[1.0e308, 1.0e308], [1.0e308, 5.0e307]], dtype=float)
    scaled = np.asarray([[1.0, 1.0], [1.0, 0.5]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = probability_entropy(probabilities)

    expected = probability_entropy(scaled)
    np.testing.assert_allclose(actual, expected)
    assert np.all(np.isfinite(actual))
