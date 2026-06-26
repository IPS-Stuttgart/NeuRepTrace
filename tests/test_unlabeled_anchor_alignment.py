from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.unlabeled_anchor_alignment import (
    UNLABELED_ANCHOR_ALIGNMENT_CATEGORY,
    anchor_template,
    fit_anchor_projection,
    fit_unlabeled_anchor_alignment,
    transform_with_anchor_projection,
)


def _source_and_target_features():
    anchors = np.asarray(["movie-0", "movie-1", "movie-2", "movie-3"], dtype=object)
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    source_a = base.copy()
    source_b = base @ np.asarray([[0.7, 0.2, 0.0], [-0.1, 0.9, 0.1], [0.2, 0.1, 0.8]]) + 1.5
    target_calibration = base @ np.asarray([[1.1, 0.1, 0.0], [0.0, 0.8, -0.2], [0.1, 0.2, 1.0]]) - 0.75
    target_test = np.vstack([target_calibration, target_calibration.mean(axis=0, keepdims=True)])
    return anchors, base, source_a, source_b, target_calibration, target_test


def test_unlabeled_anchor_alignment_transforms_source_and_target_without_labels() -> None:
    anchors, _base, source_a, source_b, target_calibration, target_test = _source_and_target_features()
    source_features = np.vstack([source_a, source_b])
    source_domains = np.asarray(["sub-a"] * len(anchors) + ["sub-b"] * len(anchors), dtype=object)
    source_anchor_values = np.concatenate([anchors, anchors])

    result = fit_unlabeled_anchor_alignment(
        source_features=source_features,
        source_domains=source_domains,
        source_anchor_values=source_anchor_values,
        target_calibration_features=target_calibration,
        target_calibration_anchor_values=anchors,
        target_test_features=target_test,
        n_components=2,
    )

    assert result.train_features.shape == (8, 2)
    assert result.test_features.shape == (5, 2)
    assert result.template.shape == (4, 2)
    assert result.common_anchors == tuple(anchors.tolist())
    assert set(result.source_projections) == {"sub-a", "sub-b"}
    assert result.metadata["unlabeled_anchor_alignment_category"] == UNLABELED_ANCHOR_ALIGNMENT_CATEGORY
    assert result.metadata["unlabeled_anchor_alignment_uses_target_calibration_features"] is True
    assert result.metadata["unlabeled_anchor_alignment_uses_target_labels"] is False
    assert result.metadata["unlabeled_anchor_alignment_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["unlabeled_anchor_alignment_valid_for_strict_source_only"] is False
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))


def test_target_projection_maps_calibration_anchor_rows_near_template() -> None:
    anchors, _base, source_a, source_b, target_calibration, _target_test = _source_and_target_features()
    source_features = np.vstack([source_a, source_b])
    source_domains = np.asarray(["sub-a"] * len(anchors) + ["sub-b"] * len(anchors), dtype=object)
    source_anchor_values = np.concatenate([anchors, anchors])

    result = fit_unlabeled_anchor_alignment(
        source_features=source_features,
        source_domains=source_domains,
        source_anchor_values=source_anchor_values,
        target_calibration_features=target_calibration,
        target_calibration_anchor_values=anchors,
        target_test_features=target_calibration,
        n_components="all",
        regularization=0.0,
    )

    assert result.test_features.shape == result.template.shape
    assert np.allclose(result.test_features, result.template, atol=1e-6)


def test_unlabeled_anchor_alignment_ignores_missing_anchor_values() -> None:
    source_features = np.asarray([[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [0.0, 1.0], [1.0, 1.0], [8.0, 8.0]])
    source_domains = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    source_anchor_values = np.asarray(["u", "v", "", "u", "v", None], dtype=object)
    target_calibration_features = np.asarray([[0.2, 0.0], [1.2, 0.0], [7.0, 7.0]])
    target_anchor_values = np.asarray(["u", "v", np.nan], dtype=object)
    target_test_features = np.asarray([[0.2, 0.0], [1.2, 0.0]])

    result = fit_unlabeled_anchor_alignment(
        source_features=source_features,
        source_domains=source_domains,
        source_anchor_values=source_anchor_values,
        target_calibration_features=target_calibration_features,
        target_calibration_anchor_values=target_anchor_values,
        target_test_features=target_test_features,
        n_components=1,
    )

    assert result.common_anchors == ("u", "v")
    assert result.train_features.shape == (6, 1)
    assert result.test_features.shape == (2, 1)


def test_unlabeled_anchor_alignment_requires_common_anchors() -> None:
    source_features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    source_domains = np.asarray(["a", "a", "b", "b"], dtype=object)
    source_anchor_values = np.asarray(["u", "v", "u", "w"], dtype=object)
    target_calibration_features = np.asarray([[0.1], [1.1]])
    target_anchor_values = np.asarray(["u", "v"], dtype=object)

    with pytest.raises(ValueError, match="at least 2 common anchors"):
        fit_unlabeled_anchor_alignment(
            source_features=source_features,
            source_domains=source_domains,
            source_anchor_values=source_anchor_values,
            target_calibration_features=target_calibration_features,
            target_calibration_anchor_values=target_anchor_values,
            target_test_features=target_calibration_features,
            min_common_anchors=2,
        )


def test_anchor_template_caps_components() -> None:
    template = anchor_template(4, n_components="all", feature_dim=10)
    assert template.shape == (4, 3)
    assert np.allclose(template.mean(axis=0), 0.0)


def test_projection_transform_rejects_wrong_feature_width() -> None:
    template = anchor_template(3, n_components=2, feature_dim=2)
    projection = fit_anchor_projection(np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), template)

    with pytest.raises(ValueError, match="column count"):
        transform_with_anchor_projection(np.asarray([[0.0, 0.0, 0.0]]), projection)


def test_target_labels_are_not_part_of_public_api() -> None:
    anchors, _base, source_a, _source_b, target_calibration, target_test = _source_and_target_features()

    with pytest.raises(TypeError):
        fit_unlabeled_anchor_alignment(
            source_features=source_a,
            source_domains=["sub-a"] * len(anchors),
            source_anchor_values=anchors,
            target_calibration_features=target_calibration,
            target_calibration_anchor_values=anchors,
            target_test_features=target_test,
            target_labels=[0, 1, 0, 1],  # type: ignore[call-arg]
        )
