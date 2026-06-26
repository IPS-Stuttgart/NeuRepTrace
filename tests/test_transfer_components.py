from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from neureptrace.decoding.transfer_components import (
    TRANSFER_COMPONENT_CATEGORY,
    TransferComponentConfig,
    fit_transfer_component_classifier,
    fit_transfer_component_features,
    normalize_standardize_scope,
    normalize_transfer_component_kernel,
    transfer_component_config,
)


def _toy_data():
    source = np.asarray([[-2.0, 0.0], [-1.5, 0.2], [-2.2, -0.1], [2.0, 0.0], [1.7, 0.2], [2.1, -0.1]])
    labels = np.asarray(["left", "left", "left", "right", "right", "right"], dtype=object)
    target = source + np.asarray([0.7, 1.5])
    return source, labels, target


def test_linear_tca_returns_category2_latent_features() -> None:
    source, _labels, target = _toy_data()

    result = fit_transfer_component_features(source_features=source, target_features=target, config={"n_components": 1})

    assert result.source_features.shape == (6, 1)
    assert result.target_features.shape == (6, 1)
    assert result.projection.shape == (2, 1)
    assert result.metadata["transfer_component_protocol_category"] == TRANSFER_COMPONENT_CATEGORY
    assert result.metadata["transfer_component_uses_target_features"] is True
    assert result.metadata["transfer_component_uses_target_labels"] is False
    assert np.all(np.isfinite(result.source_features))


def test_rbf_tca_returns_kernel_latent_features() -> None:
    source, _labels, target = _toy_data()

    result = fit_transfer_component_features(source_features=source, target_features=target, config={"n_components": 2, "kernel": "rbf", "gamma": "median", "standardize_scope": "source-target"})

    assert result.source_features.shape == (6, 2)
    assert result.target_features.shape == (6, 2)
    assert result.projection.shape == (12, 2)
    assert result.metadata["transfer_component_kernel"] == "rbf"


def test_transfer_component_classifier_trains_on_source_labels() -> None:
    source, labels, target = _toy_data()

    result = fit_transfer_component_classifier(source_features=source, source_labels=labels, target_features=target, config={"n_components": "all", "kernel": "linear"})

    assert result.predictions.shape == (6,)
    assert result.probabilities is not None
    assert result.probabilities.shape == (6, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["transfer_component_classifier_uses_source_labels"] is True
    assert result.metadata["transfer_component_classifier_uses_target_labels"] is False


def test_transfer_component_classifier_supports_decision_function_classifier() -> None:
    source, labels, target = _toy_data()

    result = fit_transfer_component_classifier(source_features=source, source_labels=labels, target_features=target, config={"n_components": 1}, classifier=LinearSVC(random_state=0))

    assert result.probabilities is not None
    assert result.probabilities.shape == (6, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_transfer_component_config_and_aliases() -> None:
    config = transfer_component_config(n_components="all", kernel="gaussian", standardize_scope="source+target", regularization="0.01")

    assert config.n_components == "all"
    assert config.kernel == "rbf"
    assert config.standardize_scope == "source_target"
    assert np.isclose(config.regularization, 0.01)
    assert normalize_transfer_component_kernel("lin") == "linear"
    assert normalize_standardize_scope("off") == "none"


def test_transfer_components_support_composite_source_labels() -> None:
    source, _labels, target = _toy_data()
    labels = [("left", 1), ("left", 1), ("left", 1), ("right", 2), ("right", 2), ("right", 2)]

    result = fit_transfer_component_classifier(source_features=source, source_labels=labels, target_features=target, config={"n_components": 1})

    assert result.predictions.shape == (6,)
    assert result.metadata["transfer_component_classifier_uses_target_labels"] is False
    assert result.metadata["transfer_component_classifier_label_encoding"] == "atomic_integer"
    assert set(result.classes.tolist()) == {("left", 1), ("right", 2)}
    assert all(isinstance(prediction, tuple) for prediction in result.predictions)


def test_transfer_components_support_numpy_object_matrix_composite_labels() -> None:
    source, _labels, target = _toy_data()
    labels = np.asarray([("left", 1), ("left", 1), ("left", 1), ("right", 2), ("right", 2), ("right", 2)], dtype=object)

    result = fit_transfer_component_classifier(source_features=source, source_labels=labels, target_features=target, config={"n_components": 1})

    assert result.classes.shape == (2,)
    assert set(result.classes.tolist()) == {("left", 1), ("right", 2)}
    assert all(isinstance(prediction, tuple) for prediction in result.predictions)


def test_transfer_component_config_parses_string_boolean_center_kernel() -> None:
    assert transfer_component_config(center_kernel="false").center_kernel is False
    assert transfer_component_config(center_kernel="off").center_kernel is False
    assert transfer_component_config(center_kernel="enabled").center_kernel is True


def test_transfer_components_normalize_direct_config_center_kernel_string() -> None:
    source, _labels, target = _toy_data()

    result = fit_transfer_component_features(source_features=source, target_features=target, config=TransferComponentConfig(n_components=1, kernel="rbf", center_kernel="false"))

    assert result.config.center_kernel is False


def test_transfer_components_reject_boolean_gamma() -> None:
    source, _labels, target = _toy_data()

    with pytest.raises(ValueError, match="gamma"):
        transfer_component_config(gamma=True)
    with pytest.raises(ValueError, match="gamma"):
        fit_transfer_component_features(source_features=source, target_features=target, config=TransferComponentConfig(n_components=1, kernel="rbf", gamma=np.bool_(False)))


def test_transfer_component_classifier_rejects_matrix_sample_weight() -> None:
    source, labels, target = _toy_data()

    with pytest.raises(ValueError, match="sample_weight"):
        fit_transfer_component_classifier(source_features=source, source_labels=labels, target_features=target, config={"n_components": 1}, sample_weight=np.ones((2, 3)))


def test_transfer_component_classifier_rejects_boolean_sample_weight() -> None:
    source, labels, target = _toy_data()

    with pytest.raises(ValueError, match="sample_weight"):
        fit_transfer_component_classifier(source_features=source, source_labels=labels, target_features=target, config={"n_components": 1}, sample_weight=[True, False, True, False, True, False])


def test_transfer_component_classifier_accepts_column_sample_weight() -> None:
    source, labels, target = _toy_data()

    result = fit_transfer_component_classifier(source_features=source, source_labels=labels, target_features=target, config={"n_components": 1}, sample_weight=np.ones((6, 1)))

    assert result.predictions.shape == (6,)
