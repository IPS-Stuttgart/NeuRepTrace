from __future__ import annotations

import numpy as np

from neureptrace.decoding.class_scores import as_class_score_matrix, class_score_matrix


def test_as_class_score_matrix_rejects_non_finite_scores() -> None:
    assert as_class_score_matrix([0.0, np.nan], np.array([0, 1]), n_samples=2) is None
    assert (
        as_class_score_matrix(
            [[0.2, 0.8], [np.inf, 0.0]],
            np.array([0, 1]),
            n_samples=2,
        )
        is None
    )


def test_class_score_matrix_falls_back_after_non_finite_decision_function() -> None:
    class Model:
        classes_ = np.array([0, 1])

        def decision_function(self, features):
            return np.array([np.nan, np.inf])

        def predict_proba(self, features):
            return np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float)

    scores, classes = class_score_matrix(Model(), np.zeros((2, 2)))

    np.testing.assert_array_equal(classes, np.array([0, 1]))
    np.testing.assert_allclose(scores, np.array([[0.7, 0.3], [0.2, 0.8]]))


def test_class_score_matrix_can_fall_back_to_predictions_after_invalid_scores() -> None:
    class Model:
        classes_ = np.array(["a", "b"])

        def decision_function(self, features):
            return np.array([np.nan, np.inf])

        def predict(self, features):
            return np.array(["b", "a"])

    scores, classes = class_score_matrix(Model(), np.zeros((2, 2)), predict_fallback=True)

    np.testing.assert_array_equal(classes, np.array(["a", "b"]))
    np.testing.assert_allclose(scores, np.array([[0.0, 1.0], [1.0, 0.0]]))
