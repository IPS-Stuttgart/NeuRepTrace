from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from neureptrace.decoding.source_bagging import fit_source_bagging_decoder


class _BinaryDecisionEstimator(BaseEstimator):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_BinaryDecisionEstimator":
        self.classes_ = np.asarray([0, 1], dtype=int)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float)[:, 0]


class _ReorderedMulticlassDecisionEstimator(BaseEstimator):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_ReorderedMulticlassDecisionEstimator":
        self.classes_ = np.asarray([2, 0, 1], dtype=int)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        scores = np.asarray(
            [
                [8.0, 1.0, 0.0],
                [0.0, 8.0, 1.0],
                [1.0, 0.0, 8.0],
            ],
            dtype=float,
        )
        return scores[: np.asarray(features).shape[0]]


def test_source_bagging_binary_decision_margin_is_not_double_scaled() -> None:
    source_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-2.0], [0.0], [2.0]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        estimator=_BinaryDecisionEstimator(),
        config={"n_estimators": 1, "random_state": 0},
    )

    positive = 1.0 / (1.0 + np.exp(-test_features[:, 0]))
    expected = np.column_stack([1.0 - positive, positive])
    np.testing.assert_allclose(result.probabilities, expected, rtol=1e-6, atol=1e-7)


def test_source_bagging_aligns_multiclass_decision_columns() -> None:
    source_features = np.asarray([[-3.0], [-2.0], [0.0], [1.0], [3.0], [4.0]], dtype=float)
    source_labels = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    test_features = np.asarray([[-1.0], [0.0], [1.0]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        estimator=_ReorderedMulticlassDecisionEstimator(),
        config={"n_estimators": 1, "random_state": 0},
    )

    model_order_scores = np.asarray(
        [
            [8.0, 1.0, 0.0],
            [0.0, 8.0, 1.0],
            [1.0, 0.0, 8.0],
        ],
        dtype=float,
    )
    global_order_scores = model_order_scores[:, [1, 2, 0]]
    shifted = global_order_scores - np.max(global_order_scores, axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected /= np.sum(expected, axis=1, keepdims=True)

    np.testing.assert_allclose(result.probabilities, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.predictions, np.asarray(["c", "a", "b"], dtype=object))
