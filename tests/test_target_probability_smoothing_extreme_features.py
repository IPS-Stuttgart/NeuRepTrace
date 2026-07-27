from __future__ import annotations

import numpy as np

from neureptrace.decoding.target_probability_smoothing import rbf_affinity, smooth_target_probabilities


def test_rbf_affinity_handles_extreme_finite_features() -> None:
    maximum = np.finfo(float).max
    features = np.asarray([[-maximum], [0.0], [maximum]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        affinity, gamma = rbf_affinity(features, gamma="auto")

    assert np.isfinite(gamma)
    assert gamma > 0.0
    assert np.all(np.isfinite(affinity))
    np.testing.assert_allclose(affinity, affinity.T)
    np.testing.assert_array_equal(np.diag(affinity), np.zeros(3))


def test_smoothing_standardizes_extreme_finite_features_without_overflow() -> None:
    maximum = np.finfo(float).max
    features = np.asarray([[-maximum], [0.0], [maximum]], dtype=float)
    probabilities = np.asarray([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = smooth_target_probabilities(features, probabilities)

    assert np.all(np.isfinite(result.affinity))
    assert np.all(np.isfinite(result.probabilities))
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(3))
