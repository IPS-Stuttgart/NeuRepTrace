from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.class_scores import class_score_matrix
from neureptrace.decoding.windowed import fit_window_model, predict_window_model, score_windowed_decoding


class _ConstantClassifier:
    def fit(self, features, labels):
        self.label_ = np.asarray(labels)[0]
        return self

    def predict(self, features):
        return np.full(np.asarray(features).shape[0], self.label_)

    def decision_function(self, features):
        return np.zeros(np.asarray(features).shape[0])


class _ScoreModel:
    classes_ = np.asarray([0, 1])

    def decision_function(self, features):
        return np.asarray(features)[:, 0]


def _fit_constant_classifier(features, labels):
    return _ConstantClassifier().fit(features, labels)


def test_fit_window_model_rejects_complex_train_features() -> None:
    train_features = np.asarray([[0.0 + 0.0j], [1.0 + 2.0j]])

    with pytest.raises(ValueError, match="train_features must contain real-valued features"):
        fit_window_model(
            train_features,
            np.asarray([0, 1]),
            fit_model=_fit_constant_classifier,
            components_pca=float("inf"),
        )


def test_score_windowed_decoding_rejects_complex_generator_features() -> None:
    validation_features = (row for row in ([0.0], [1.0 + 2.0j]))

    with pytest.raises(ValueError, match="validation_features must contain real-valued features"):
        score_windowed_decoding(
            train_features=np.asarray([[0.0], [1.0]]),
            train_labels=np.asarray([0, 1]),
            validation_features=validation_features,
            validation_labels=np.asarray([0, 1]),
            fit_model=_fit_constant_classifier,
            components_pca=float("inf"),
        )


def test_predict_window_model_rejects_complex_object_features() -> None:
    bundle = fit_window_model(
        np.asarray([[0.0], [1.0]]),
        np.asarray([0, 1]),
        fit_model=_fit_constant_classifier,
        components_pca=float("inf"),
    )
    features = np.asarray([[0.0], [1.0 + 2.0j]], dtype=object)

    with pytest.raises(ValueError, match="features must contain real-valued features"):
        predict_window_model(bundle, features)


def test_class_score_matrix_rejects_complex_features() -> None:
    features = np.asarray([[0.0 + 0.0j], [1.0 + 2.0j]])

    with pytest.raises(ValueError, match="features must contain real-valued features"):
        class_score_matrix(_ScoreModel(), features)
