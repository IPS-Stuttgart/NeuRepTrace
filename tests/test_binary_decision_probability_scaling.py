from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401  # importing the package installs runtime patches
from neureptrace.decoding import source_ensemble
from neureptrace.decoding import source_free
from neureptrace.decoding import subspace_alignment
from neureptrace.decoding import transfer_component_analysis
from neureptrace.decoding import transfer_components
from neureptrace.decoding.class_scores import as_class_score_matrix
from neureptrace.decoding.classifiers import DecodedLabelClassifier


class DecisionOnlyBinaryClassifier:
    classes_ = np.asarray([0, 1])

    def __init__(self, scores: np.ndarray):
        self._scores = np.asarray(scores, dtype=float)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return self._scores[: np.asarray(features).shape[0]]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[(self.decision_function(features) > 0.0).astype(int)]


class _EncodedDecisionModel:
    def __init__(self, scores: np.ndarray):
        self._scores = np.asarray(scores, dtype=float)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return self._scores[: np.asarray(features).shape[0]]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.decision_function(features) > 0.0).astype(int)


def _features(n_rows: int) -> np.ndarray:
    return np.arange(float(n_rows), dtype=float).reshape(n_rows, 1)


def _expected_binary_logits(scores: np.ndarray) -> np.ndarray:
    half_scores = 0.5 * np.asarray(scores, dtype=float)
    return np.column_stack([-half_scores, half_scores])


def _expected_binary_probabilities(scores: np.ndarray) -> np.ndarray:
    positive = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))
    return np.column_stack([1.0 - positive, positive])


def test_source_free_binary_decision_scores_are_not_double_scaled() -> None:
    scores = np.asarray([-2.0, 0.0, 2.0])
    model = DecisionOnlyBinaryClassifier(scores)

    probabilities = source_free._predict_source_probabilities(model, _features(3), np.asarray([0, 1]))

    np.testing.assert_allclose(probabilities, _expected_binary_probabilities(scores), rtol=1e-12, atol=1e-12)


def test_source_ensemble_binary_decision_scores_are_not_double_scaled() -> None:
    scores = np.asarray([-2.0, 0.0, 2.0])
    model = DecisionOnlyBinaryClassifier(scores)

    probabilities = source_ensemble._aligned_probabilities(model, _features(3), classes=np.asarray([0, 1]), epsilon=1e-12)

    np.testing.assert_allclose(probabilities, _expected_binary_probabilities(scores), rtol=1e-12, atol=1e-12)


def test_aligned_decoders_binary_decision_scores_are_not_double_scaled() -> None:
    scores = np.asarray([-2.0, 0.0, 2.0])
    model = DecisionOnlyBinaryClassifier(scores)
    expected = _expected_binary_probabilities(scores)

    helpers = (
        subspace_alignment._probabilities_or_none,
        transfer_components._predict_probabilities_or_none,
        transfer_component_analysis._predict_probabilities_or_none,
    )
    for helper in helpers:
        probabilities = helper(model, _features(3))
        np.testing.assert_allclose(probabilities, expected, rtol=1e-12, atol=1e-12)


def test_class_score_matrix_binary_decision_scores_are_not_double_scaled() -> None:
    scores = np.asarray([-2.0, 0.0, 2.0])

    logits = as_class_score_matrix(scores, classes=np.asarray([0, 1]), n_samples=3)
    column_logits = as_class_score_matrix(scores.reshape(-1, 1), classes=np.asarray([0, 1]), n_samples=3)

    np.testing.assert_allclose(logits, _expected_binary_logits(scores), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(column_logits, _expected_binary_logits(scores), rtol=1e-12, atol=1e-12)


def test_decoded_label_classifier_binary_decision_scores_are_not_double_scaled() -> None:
    scores = np.asarray([-2.0, 0.0, 2.0])
    classifier = DecodedLabelClassifier(_EncodedDecisionModel(scores), classes=np.asarray(["left", "right"]))

    logits = classifier.decision_function(_features(3))

    np.testing.assert_allclose(logits, _expected_binary_logits(scores), rtol=1e-12, atol=1e-12)
