from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.progressive_sequence_finetune import (
    permutation_constrained_decode,
    sinkhorn_trial_probabilities,
)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (
            np.asarray([[[True, False], [False, True]]]),
            "Boolean",
        ),
        (
            np.asarray([[[0.5 + 0.1j, 0.5], [0.4, 0.6]]]),
            "complex",
        ),
        (
            np.asarray([[[0.5, 0.5], [-0.1, 1.1]]]),
            "non-negative",
        ),
        (
            np.asarray([[[0.5, 0.5], [True, 0.0]]], dtype=object),
            "Boolean",
        ),
        (
            np.asarray([[[0.5, 0.5], [0.4 + 0.2j, 0.6]]], dtype=object),
            "complex",
        ),
    ],
)
def test_sinkhorn_rejects_lossy_probability_inputs(probabilities: np.ndarray, message: str):
    with pytest.raises(ValueError, match=message):
        sinkhorn_trial_probabilities(probabilities)


def test_permutation_decode_rejects_negative_probabilities():
    probabilities = np.asarray([[[0.8, 0.2], [1.1, -0.1]]])

    with pytest.raises(ValueError, match="non-negative"):
        permutation_constrained_decode(probabilities)


def test_sinkhorn_normalizes_extreme_finite_rows_without_overflow():
    maximum = np.finfo(float).max
    probabilities = np.asarray(
        [
            [
                [maximum, maximum],
                [maximum, maximum / 2.0],
            ]
        ]
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = sinkhorn_trial_probabilities(probabilities, iterations=10)

    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(normalized.sum(axis=2), 1.0)


def test_sinkhorn_preserves_valid_nonnegative_inputs():
    probabilities = np.asarray([[[0.8, 0.2], [0.3, 0.7]]])

    normalized = sinkhorn_trial_probabilities(probabilities, iterations=20)

    assert normalized.shape == probabilities.shape
    assert np.all(normalized >= 0.0)
    np.testing.assert_allclose(normalized.sum(axis=2), 1.0)
