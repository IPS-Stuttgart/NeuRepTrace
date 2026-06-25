from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from neureptrace.decoding.transfer_component_analysis import (
    TCA_CATEGORY,
    fit_tca_transfer_classifier,
    normalize_tca_kernel,
    transfer_component_analysis_features,
    transform_with_tca_model,
)


def test_tca_linear_shapes_and_metadata() -> None:
    source = np.asarray(
        [
            [-2.0, 0.0, 0.2],
            [-1.5, 0.1, 0.0],
            [1.5, -0.1, 0.0],
            [2.0, 0.0, -0.2],
        ],
        dtype=float,
    )
    target = np.asarray([[-1.8, 0.2, 0.1], [1.8, -0.2, -0.1], [0.0, 0.0, 0.0]], dtype=float)

    result = transfer_component_analysis_features(source, target, n_components=2, kernel="linear")

    assert result.source_features.shape == (4, 2)
    assert result.target_features.shape == (3, 2)
    assert result.model.projection.shape == (7, 2)
    assert result.metadata["tca_protocol_category"] == TCA_CATEGORY
    assert result.metadata["tca_uses_target_features"] is True
    assert result.metadata["tca_uses_target_labels"] is False
    assert result.metadata["tca_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["tca_valid_for_strict_source_only"] is False
    assert np.all(np.isfinite(result.source_features))
    assert np.all(np.isfinite(result.target_features))
    assert np.allclose(transform_with_tca_model(result.model, target), result.target_features)


def test_tca_rbf_kernel_and_component_cap() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    target = np.asarray([[0.2, 0.1], [0.8, 0.2]], dtype=float)

    result = transfer_component_analysis_features(source, target, n_components="all", kernel="rbf", gamma="median")

    assert result.source_features.shape == (3, 5)
    assert result.target_features.shape == (2, 5)
    assert result.model.kernel == "rbf"
    assert result.model.gamma is not None
    assert result.metadata["tca_kernel"] == "rbf"
    assert result.metadata["tca_n_components"] == 5


def test_tca_transfer_classifier_uses_source_labels_only() -> None:
    source = np.asarray(
        [
            [-2.0, 0.0],
            [-1.4, 0.1],
            [1.4, -0.1],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    target = np.asarray([[-1.8, 0.0], [1.8, 0.0]], dtype=float)

    result = fit_tca_transfer_classifier(
        source_features=source,
        source_labels=source_labels,
        target_features=target,
        n_components=2,
    )

    assert result.predictions.shape == (2,)
    assert set(result.classes.tolist()) == {"left", "right"}
    assert result.probabilities is not None
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["tca_classifier_uses_source_labels"] is True
    assert result.metadata["tca_classifier_uses_target_labels"] is False


def test_tca_transfer_classifier_supports_decision_function_probability_fallback() -> None:
    source = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    target = np.asarray([[-1.5], [1.5]], dtype=float)

    result = fit_tca_transfer_classifier(
        source_features=source,
        source_labels=[0, 0, 1, 1],
        target_features=target,
        n_components=1,
        classifier=LinearSVC(random_state=0),
    )

    assert result.predictions.shape == (2,)
    assert result.probabilities is not None
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_tca_transform_rejects_wrong_feature_width() -> None:
    result = transfer_component_analysis_features([[0.0, 0.0], [1.0, 0.0]], [[0.2, 0.1]], n_components=1)

    with pytest.raises(ValueError, match="width"):
        transform_with_tca_model(result.model, [[1.0, 2.0, 3.0]])


def test_tca_rejects_bad_kernel_and_normalizes_aliases() -> None:
    assert normalize_tca_kernel("gaussian") == "rbf"

    with pytest.raises(ValueError, match="Unknown TCA kernel"):
        normalize_tca_kernel("cosine")


def test_tca_classifier_rejects_invalid_sample_weight() -> None:
    with pytest.raises(ValueError, match="sample_weight"):
        fit_tca_transfer_classifier(
            source_features=[[0.0], [1.0], [2.0], [3.0]],
            source_labels=[0, 0, 1, 1],
            target_features=[[0.5], [2.5]],
            sample_weight=[1.0, -1.0, 1.0, 1.0],
        )


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_tca_transfer_classifier(
            source_features=[[0.0], [1.0], [2.0], [3.0]],
            source_labels=[0, 0, 1, 1],
            target_features=[[0.5], [2.5]],
            target_labels=[0, 1],  # type: ignore[call-arg]
        )

    with pytest.raises(TypeError):
        transfer_component_analysis_features(
            [[0.0], [1.0]],
            [[0.5]],
            target_labels=[0],  # type: ignore[call-arg]
        )
