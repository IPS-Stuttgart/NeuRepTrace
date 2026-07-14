from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior, estimate_source_class_prior


def test_source_prior_normalizes_extreme_finite_rows_without_overflow() -> None:
    limit = np.finfo(np.float64).max
    probabilities = np.asarray(
        [
            [limit, limit],
            [limit, limit / 9.0],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = adjust_probabilities_to_source_prior(
            probabilities,
            source_labels=[0, 1],
            classes=[0, 1],
            config={"target_prior": "source"},
        )

    np.testing.assert_allclose(result.probabilities, [[0.5, 0.5], [0.9, 0.1]])


def test_source_prior_normalizes_extreme_finite_smoothing_without_overflow() -> None:
    limit = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        prior, classes = estimate_source_class_prior(
            ["left", "right"],
            classes=["left", "right"],
            smoothing=limit,
        )

    assert classes.tolist() == ["left", "right"]
    np.testing.assert_allclose(prior, [0.5, 0.5])
