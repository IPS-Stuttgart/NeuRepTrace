from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.random_subspace import _aligned_probabilities as random_subspace_aligned_probabilities
from neureptrace.decoding.source_bagging import _aligned_probabilities as source_bagging_aligned_probabilities


class BinaryMarginEstimator:
    classes_ = np.asarray([0, 1], dtype=int)

    def decision_function(self, features):
        return np.full(np.asarray(features).shape[0], 2.0, dtype=float)


@pytest.mark.parametrize(
    "aligned_probabilities",
    [source_bagging_aligned_probabilities, random_subspace_aligned_probabilities],
)
def test_binary_decision_margins_are_not_doubled(aligned_probabilities):
    features = np.zeros((3, 2), dtype=float)

    probabilities = aligned_probabilities(
        BinaryMarginEstimator(),
        features,
        classes=np.asarray([0, 1], dtype=int),
        epsilon=1e-12,
    )

    expected_positive = 1.0 / (1.0 + np.exp(-2.0))
    assert probabilities.shape == (3, 2)
    assert probabilities[:, 1] == pytest.approx(expected_positive)
    assert probabilities[:, 0] == pytest.approx(1.0 - expected_positive)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(3))
