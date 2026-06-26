from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from neureptrace.decoding.source_free import SOURCE_FREE_ADAPTATION_PROTOCOL, SourceFreeSubjectAdapter, fit_source_free_predict_proba


class _CompositeLabelSourceModel:
    def __init__(self):
        self.classes_ = _label_vector(("left", 1), ("right", 2))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.80, 0.20]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.25, 0.75], dtype=float)
        return probabilities


def _label_vector(*labels: object) -> np.ndarray:
    values = np.empty(len(labels), dtype=object)
    values[:] = labels
    return values


def _source_target_fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    source_features = np.vstack(
        [
            rng.normal(-1.0, 0.4, size=(40, 4)),
            rng.normal(1.0, 0.4, size=(40, 4)),
        ]
    )
    source_labels = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
    target_features = np.vstack(
        [
            rng.normal(-0.6, 0.5, size=(18, 4)),
            rng.normal(1.4, 0.5, size=(18, 4)),
        ]
    )
    return source_features, source_labels, target_features


def test_source_free_adaptation_uses_target_features_but_no_target_labels():
    source_features, source_labels, target_features = _source_target_fixture()
    source_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=13))
    source_model.fit(source_features, source_labels)

    result = fit_source_free_predict_proba(
        source_model=source_model,
        target_features=target_features,
        confidence_threshold=0.55,
        max_iterations=3,
        prototype_weight=0.5,
    )

    assert result.probabilities.shape == (target_features.shape[0], 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.adapter.predict_proba(target_features), result.probabilities)
    assert result.metadata["source_free_protocol"] == SOURCE_FREE_ADAPTATION_PROTOCOL
    assert result.metadata["source_free_uses_source_features_during_adaptation"] is False
    assert result.metadata["source_free_uses_source_labels_during_adaptation"] is False
    assert result.metadata["source_free_uses_target_features"] is True
    assert result.metadata["source_free_uses_target_labels"] is False
    assert result.metadata["source_free_valid_for_benchmark"] is True
    assert result.metadata["source_free_target_rows"] == target_features.shape[0]


def test_source_free_adapter_does_not_accept_target_labels_keyword():
    source_features, source_labels, target_features = _source_target_fixture()
    source_model = LogisticRegression(max_iter=500, random_state=13).fit(source_features, source_labels)
    adapter = SourceFreeSubjectAdapter(source_model=source_model)

    with pytest.raises(TypeError):
        adapter.fit(target_features, target_labels=np.zeros(target_features.shape[0], dtype=int))


def test_source_free_adaptation_supports_decision_function_models():
    source_features, source_labels, target_features = _source_target_fixture(seed=1)
    source_model = make_pipeline(StandardScaler(), LinearSVC(random_state=13, max_iter=5000))
    source_model.fit(source_features, source_labels)

    adapter = SourceFreeSubjectAdapter(
        source_model=source_model,
        confidence_threshold=0.50,
        max_iterations=2,
        feature_space="model_preprocessor",
    ).fit(target_features)

    probabilities = adapter.predict_proba(target_features)
    assert probabilities.shape == (target_features.shape[0], 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert adapter.metadata()["source_free_feature_space"] == "model_preprocessor"


def test_source_free_adaptation_accepts_explicit_tuple_class_order():
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)
    source_model = _CompositeLabelSourceModel()

    result = fit_source_free_predict_proba(
        source_model=source_model,
        target_features=target_features,
        classes=[("right", 2), ("left", 1)],
        max_iterations=0,
    )

    assert result.adapter.classes_.tolist() == [("right", 2), ("left", 1)]
    assert np.allclose(result.probabilities, source_model.predict_proba(target_features)[:, [1, 0]])
    assert result.adapter.predict(target_features).tolist() == [("left", 1), ("right", 2)]
