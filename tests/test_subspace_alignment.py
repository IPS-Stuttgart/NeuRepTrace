from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from neureptrace.decoding.subspace_alignment import (
    SUBSPACE_ALIGNMENT_CATEGORY,
    fit_subspace_aligned_classifier,
    fit_subspace_alignment,
    normalize_standardization_scope,
)


def _toy_domains():
    rng = np.random.default_rng(13)
    source = np.vstack(
        [
            rng.normal(loc=(-2.0, 0.0, 0.5), scale=0.2, size=(8, 3)),
            rng.normal(loc=(2.0, 0.0, -0.5), scale=0.2, size=(8, 3)),
        ]
    )
    rotation = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    target = source @ rotation + np.asarray([0.5, -1.0, 2.0])
    labels = np.asarray(["left"] * 8 + ["right"] * 8, dtype=object)
    return source, target, labels


def test_subspace_alignment_returns_category2_features() -> None:
    source, target, _labels = _toy_domains()

    result = fit_subspace_alignment(source, target, n_components=2)

    assert result.source_features.shape == (16, 2)
    assert result.target_features.shape == (16, 2)
    assert result.model.source_basis.shape == (3, 2)
    assert result.model.target_basis.shape == (3, 2)
    assert result.model.alignment_matrix.shape == (2, 2)
    assert result.metadata["subspace_alignment_protocol_category"] == SUBSPACE_ALIGNMENT_CATEGORY
    assert result.metadata["subspace_alignment_uses_target_features"] is True
    assert result.metadata["subspace_alignment_uses_target_labels"] is False
    assert result.metadata["subspace_alignment_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["subspace_alignment_valid_for_strict_source_only"] is False
    assert np.all(np.isfinite(result.source_features))
    assert np.all(np.isfinite(result.target_features))


def test_subspace_alignment_component_cap_and_scope_alias() -> None:
    source, target, _labels = _toy_domains()

    result = fit_subspace_alignment(source[:3], target[:4], n_components="all", standardization_scope="source-plus-target")

    assert result.source_features.shape[1] == 2
    assert result.target_features.shape[1] == 2
    assert result.metadata["subspace_alignment_standardization_scope"] == "source_target"
    assert normalize_standardization_scope("off") == "none"


def test_subspace_alignment_model_transforms_new_rows() -> None:
    source, target, _labels = _toy_domains()
    result = fit_subspace_alignment(source, target, n_components=2)

    new_source = result.model.transform_source(source[:2])
    new_target = result.model.transform_target(target[:2])

    assert new_source.shape == (2, 2)
    assert new_target.shape == (2, 2)
    assert np.all(np.isfinite(new_source))
    assert np.all(np.isfinite(new_target))


def test_subspace_aligned_classifier_predicts_target_rows() -> None:
    source, target, labels = _toy_domains()

    result = fit_subspace_aligned_classifier(
        source_features=source,
        source_labels=labels,
        target_features=target,
        n_components=2,
    )

    assert result.predictions.shape == (16,)
    assert set(result.predictions.tolist()) <= {"left", "right"}
    assert result.probabilities is not None
    assert result.probabilities.shape == (16, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["subspace_alignment_uses_source_labels"] is True
    assert result.metadata["subspace_alignment_uses_target_labels"] is False


def test_subspace_aligned_classifier_supports_decision_function_fallback() -> None:
    source, target, labels = _toy_domains()

    result = fit_subspace_aligned_classifier(
        source_features=source,
        source_labels=labels,
        target_features=target,
        n_components=2,
        classifier=LinearSVC(random_state=0),
    )

    assert result.predictions.shape == (16,)
    assert result.probabilities is not None
    assert result.probabilities.shape == (16, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_subspace_aligned_classifier_preserves_tuple_labels() -> None:
    source, target, _labels = _toy_domains()
    labels = [("left", 1)] * 8 + [("right", 2)] * 8

    result = fit_subspace_aligned_classifier(source_features=source, source_labels=labels, target_features=target, n_components=2)

    assert set(result.classes.tolist()) == {("left", 1), ("right", 2)}
    assert set(result.predictions.tolist()) <= {("left", 1), ("right", 2)}


def test_subspace_alignment_rejects_mismatched_feature_width() -> None:
    source, target, _labels = _toy_domains()

    with pytest.raises(ValueError, match="same width"):
        fit_subspace_alignment(source, target[:, :2])


def test_subspace_aligned_classifier_rejects_target_labels_argument() -> None:
    source, target, labels = _toy_domains()

    with pytest.raises(TypeError):
        fit_subspace_aligned_classifier(
            source_features=source,
            source_labels=labels,
            target_features=target,
            target_labels=labels,  # type: ignore[call-arg]
        )


def test_subspace_aligned_classifier_validates_sample_weight() -> None:
    source, target, labels = _toy_domains()

    with pytest.raises(ValueError, match="sample_weight"):
        fit_subspace_aligned_classifier(
            source_features=source,
            source_labels=labels,
            target_features=target,
            sample_weight=[1.0, 2.0],
        )
