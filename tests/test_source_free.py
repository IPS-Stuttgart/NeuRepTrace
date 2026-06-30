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


class _ImbalancedPseudoLabelSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.94, 0.06]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.42, 0.58], dtype=float)
        return probabilities


class _CollapsedPseudoLabelSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.55, 0.45]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.51, 0.49], dtype=float)
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
    assert result.metadata["source_free_prototype_estimator"] == "hard"


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


def test_balanced_topk_selection_keeps_minority_pseudo_class_active():
    target_features = np.vstack([np.full((10, 2), -1.0), np.full((4, 2), 1.0)])

    confidence_adapter = SourceFreeSubjectAdapter(
        source_model=_ImbalancedPseudoLabelSourceModel(),
        confidence_threshold=0.80,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        pseudo_label_selection="confidence",
    ).fit(target_features)
    assert confidence_adapter.metadata()["source_free_active_classes"] == 1
    assert confidence_adapter.metadata()["source_free_stop_reason"] == "insufficient_active_classes"

    balanced_adapter = SourceFreeSubjectAdapter(
        source_model=_ImbalancedPseudoLabelSourceModel(),
        confidence_threshold=0.80,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        pseudo_label_selection="balanced_topk",
        balanced_topk_per_class=2,
    ).fit(target_features)

    metadata = balanced_adapter.metadata()
    assert metadata["source_free_pseudo_label_selection"] == "balanced_topk"
    assert metadata["source_free_balanced_topk_per_class"] == 2
    assert metadata["source_free_active_classes"] == 2
    assert balanced_adapter.prototype_class_counts_.tolist() == [2, 2]
    assert balanced_adapter.selected_.sum() == 4


def test_soft_all_prototypes_keep_classes_active_when_argmax_collapses():
    target_features = np.vstack([np.full((8, 2), -1.0), np.full((8, 2), 1.0)])

    hard_adapter = SourceFreeSubjectAdapter(
        source_model=_CollapsedPseudoLabelSourceModel(),
        confidence_threshold=0.90,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        pseudo_label_selection="confidence",
        prototype_estimator="hard",
    ).fit(target_features)
    assert hard_adapter.metadata()["source_free_stop_reason"] == "none_selected"
    assert hard_adapter.metadata()["source_free_active_classes"] == 0
    assert hard_adapter.metadata()["source_free_n_selected"] == 0

    soft_adapter = SourceFreeSubjectAdapter(
        source_model=_CollapsedPseudoLabelSourceModel(),
        confidence_threshold=0.90,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        prototype_temperature=0.1,
        pseudo_label_selection="confidence",
        prototype_estimator="soft_all",
    ).fit(target_features)

    metadata = soft_adapter.metadata()
    assert metadata["source_free_prototype_estimator"] == "soft_all"
    assert metadata["source_free_uses_target_labels"] is False
    assert metadata["source_free_valid_for_benchmark"] is True
    assert metadata["source_free_active_classes"] == 2
    assert metadata["source_free_n_selected"] == target_features.shape[0]
    assert np.all(soft_adapter.selected_)
    assert metadata["source_free_stop_reason"] != "selection_repeated"
    assert metadata["source_free_iterations"] >= 2
    assert np.all(np.isfinite(soft_adapter.prototypes_))
    assert np.allclose(soft_adapter.predict_proba(target_features).sum(axis=1), 1.0)


def test_fit_source_free_predict_proba_forwards_soft_prototype_estimator():
    target_features = np.vstack([np.full((4, 2), -1.0), np.full((4, 2), 1.0)])

    result = fit_source_free_predict_proba(
        source_model=_CollapsedPseudoLabelSourceModel(),
        target_features=target_features,
        confidence_threshold=0.90,
        max_iterations=1,
        min_class_count=2,
        min_active_classes=2,
        prototype_estimator="soft_all",
    )

    assert result.metadata["source_free_prototype_estimator"] == "soft_all"
    assert result.metadata["source_free_active_classes"] == 2
    assert result.probabilities.shape == (8, 2)


def test_source_free_rejects_unknown_prototype_estimator():
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="prototype_estimator"):
        SourceFreeSubjectAdapter(
            source_model=_CompositeLabelSourceModel(),
            prototype_estimator="target_labels",
        ).fit(target_features)


def test_source_free_string_boolean_is_parsed_for_metadata():
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)

    result = fit_source_free_predict_proba(
        source_model=_CompositeLabelSourceModel(),
        target_features=target_features,
        max_iterations=0,
        standardize_target="false",
    )

    assert result.metadata["source_free_standardize_target"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_threshold", [0.5]),
        ("max_iterations", [1]),
        ("min_class_count", (1,)),
        ("min_active_classes", {"value": 2}),
        ("prototype_weight", [0.5]),
        ("prototype_temperature", (1.0,)),
        ("balanced_topk_per_class", [1]),
    ],
)
def test_source_free_rejects_array_like_numeric_scalar_controls(field: str, value: object):
    target_features = np.array([[-1.0, 0.0], [2.0, 0.0]], dtype=float)
    kwargs = {field: value}
    if field == "balanced_topk_per_class":
        kwargs["pseudo_label_selection"] = "balanced_topk"

    with pytest.raises(ValueError, match=field):
        SourceFreeSubjectAdapter(
            source_model=_CompositeLabelSourceModel(),
            **kwargs,
        ).fit(target_features)


def test_legacy_singular_soft_prototype_patch_does_not_shadow_current_runtime():
    import neureptrace._source_free_soft_prototype_patch as legacy_patch
    import neureptrace.decoding.source_free as source_free

    before_fit = source_free.SourceFreeSubjectAdapter.fit
    before_predict_proba = source_free.fit_source_free_predict_proba

    legacy_patch.install()

    assert source_free.SourceFreeSubjectAdapter.fit is before_fit
    assert source_free.fit_source_free_predict_proba is before_predict_proba
