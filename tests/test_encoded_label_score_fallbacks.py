from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401  # importing the package installs runtime patches
from neureptrace._mekt_vector_validation_patch import _EncodedLabelEstimator
from neureptrace.decoding.classifiers import DecodedLabelClassifier


class _PredictionOnlyModel:
    def __init__(self, predictions: np.ndarray | list[int]):
        self._predictions = np.asarray(predictions, dtype=int)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._predictions[: np.asarray(features).shape[0]]


def _features(n_rows: int) -> np.ndarray:
    return np.zeros((n_rows, 1), dtype=float)


def test_decoded_label_classifier_prediction_fallback_returns_one_hot_scores() -> None:
    classifier = DecodedLabelClassifier(_PredictionOnlyModel([1, 0]), classes=np.asarray(["left", "right"]))

    scores = classifier.decision_function(_features(2))

    np.testing.assert_array_equal(scores, np.asarray([[0.0, 1.0], [1.0, 0.0]]))


def test_decoded_label_classifier_prediction_fallback_rejects_unknown_ids() -> None:
    classifier = DecodedLabelClassifier(_PredictionOnlyModel([0, 2]), classes=np.asarray(["left", "right"]))

    with pytest.raises(ValueError, match="outside the fitted class range"):
        classifier.decision_function(_features(2))


def test_mekt_prediction_fallback_returns_one_hot_scores() -> None:
    classifier = _EncodedLabelEstimator(_PredictionOnlyModel([0, 1]))
    classifier.classes_ = np.asarray(["left", "right"], dtype=object)
    classifier.estimator_ = _PredictionOnlyModel([0, 1])

    scores = classifier.decision_function(_features(2))

    np.testing.assert_array_equal(scores, np.asarray([[1.0, 0.0], [0.0, 1.0]]))


def test_mekt_prediction_fallback_rejects_unknown_ids() -> None:
    classifier = _EncodedLabelEstimator(_PredictionOnlyModel([0, 1]))
    classifier.classes_ = np.asarray(["left", "right"], dtype=object)
    classifier.estimator_ = _PredictionOnlyModel([-1, 1])

    with pytest.raises(ValueError, match="outside the fitted class range"):
        classifier.decision_function(_features(2))
