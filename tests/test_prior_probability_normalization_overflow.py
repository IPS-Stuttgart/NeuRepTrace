from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior, estimate_source_class_prior
from neureptrace.decoding.target_prior_adjustment import adjust_target_probabilities_to_prior


def test_source_prior_adjustment_normalizes_overflowing_finite_rows() -> None:
    probabilities = np.asarray([[1e308, 1e308], [1e308, 2e307]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = adjust_probabilities_to_source_prior(
            probabilities,
            source_labels=[0, 1],
            classes=[0, 1],
            config={"target_prior": "source"},
        )

    expected = np.asarray([[0.5, 0.5], [5.0 / 6.0, 1.0 / 6.0]], dtype=np.float32)
    np.testing.assert_allclose(result.probabilities, expected)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(2))


def test_source_prior_normalizes_large_finite_smoothing() -> None:
    with np.errstate(over="raise", invalid="raise"):
        prior, classes = estimate_source_class_prior(
            ["a", "b"],
            classes=["a", "b"],
            smoothing=1e308,
        )

    assert classes.tolist() == ["a", "b"]
    np.testing.assert_allclose(prior, [0.5, 0.5])


def test_target_prior_adjustment_normalizes_large_rows_and_explicit_prior() -> None:
    probabilities = np.asarray([[1e308, 1e308], [1e308, 2e307]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = adjust_target_probabilities_to_prior(
            probabilities,
            config={
                "estimator": "mean",
                "source_prior": [1e308, 1e308],
                "strength": 0.0,
            },
        )

    expected = np.asarray([[0.5, 0.5], [5.0 / 6.0, 1.0 / 6.0]], dtype=np.float32)
    np.testing.assert_allclose(result.original_probabilities, expected)
    np.testing.assert_allclose(result.probabilities, expected)
    np.testing.assert_allclose(result.source_prior, [0.5, 0.5])
