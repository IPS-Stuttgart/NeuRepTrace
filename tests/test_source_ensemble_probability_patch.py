from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401 - installs runtime robustness patches
from neureptrace.decoding import source_ensemble


class BinaryDecisionModel:
    classes_ = np.asarray([0, 1])

    def decision_function(self, features):
        return np.asarray([-2.0, 0.0, 2.0], dtype=float)


def test_source_ensemble_binary_decision_fallback_accepts_list_features():
    raw_scores, model_classes = source_ensemble._decision_probabilities(
        BinaryDecisionModel(),
        [[-1.0], [0.0], [1.0]],
        np.asarray([0, 1]),
    )

    probabilities = source_ensemble._normalize_probability_rows(raw_scores, epsilon=1e-12)

    assert model_classes.tolist() == [0, 1]
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(probabilities[:, 1], 1.0 / (1.0 + np.exp(-np.asarray([-2.0, 0.0, 2.0]))))
