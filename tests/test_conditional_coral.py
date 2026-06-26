from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.conditional_coral import (
    CONDITIONAL_CORAL_CATEGORY,
    conditional_coral_config,
    coral_align_features,
    feature_stats,
    fit_pseudo_label_conditional_coral,
    normalize_conditional_coral_fallback,
)


def test_conditional_coral_aligns_source_class_means_to_pseudo_target_classes() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.2, 0.1],
            [5.0, 5.0],
            [5.2, 4.9],
            [4.8, 5.1],
        ],
        dtype=float,
    )
    source_labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    target_features = np.asarray(
        [
            [1.0, 2.0],
            [1.2, 1.9],
            [0.8, 2.1],
            [8.0, 9.0],
            [8.2, 8.9],
            [7.8, 9.1],
        ],
        dtype=float,
    )
    pseudo_labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)

    result = fit_pseudo_label_conditional_coral(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_pseudo_labels=pseudo_labels,
        config={"regularization": 1e-6, "min_target_rows_per_class": 2},
    )

    assert result.train_features.shape == source_features.shape
    assert result.test_features.shape == target_features.shape
    assert result.metadata["conditional_coral_protocol_category"] == CONDITIONAL_CORAL_CATEGORY
    assert result.metadata["conditional_coral_uses_target_labels"] is False
    assert result.metadata["conditional_coral_uses_target_pseudo_labels"] is True
    assert result.used_fallback_classes == ()
    for class_label in result.classes.tolist():
        aligned_mean = result.train_features[source_labels == class_label].mean(axis=0)
        target_mean = target_features[pseudo_labels == class_label].mean(axis=0)
        assert np.allclose(aligned_mean, target_mean, atol=1e-5)


def test_target_probabilities_define_pseudo_labels_and_confidence_fallback() -> None:
    source_features = np.asarray([[0.0], [0.2], [5.0], [5.2]], dtype=float)
    source_labels = np.asarray([0, 0, 1, 1], dtype=object)
    target_features = np.asarray([[1.0], [1.2], [2.0], [2.2]], dtype=float)
    target_probabilities = np.asarray(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.70, 0.30],
            [0.65, 0.35],
        ],
        dtype=float,
    )

    result = fit_pseudo_label_conditional_coral(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_probabilities=target_probabilities,
        config={"confidence_threshold": 0.8, "min_target_rows_per_class": 2, "fallback": "global"},
    )

    assert result.pseudo_labels.tolist() == [0, 0, 0, 0]
    assert result.used_fallback_classes == (1,)
    assert result.metadata["conditional_coral_pseudo_label_source"] == "target_probabilities"
    assert result.metadata["conditional_coral_confident_target_rows"] == 2
    assert result.metadata["conditional_coral_fallback_classes"] == "1"


def test_default_source_classifier_pseudo_labels_target_rows() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [2.0], [1.5]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    target_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_pseudo_label_conditional_coral(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        config={"min_target_rows_per_class": 1},
    )

    assert result.pseudo_labels.tolist() == ["left", "right"]
    assert result.metadata["conditional_coral_pseudo_label_source"] == "source_classifier"
    assert result.train_features.shape == source_features.shape


def test_conditional_coral_error_fallback_rejects_under_supported_pseudo_class() -> None:
    source_features = np.asarray([[0.0], [0.1], [3.0], [3.1]], dtype=float)
    source_labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    target_features = np.asarray([[1.0], [1.1], [2.0]], dtype=float)
    pseudo_labels = np.asarray(["a", "a", "b"], dtype=object)

    with pytest.raises(ValueError, match="below min_target_rows_per_class"):
        fit_pseudo_label_conditional_coral(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_pseudo_labels=pseudo_labels,
            config={"min_target_rows_per_class": 2, "fallback": "error"},
        )


def test_unknown_target_pseudo_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent from source classes"):
        fit_pseudo_label_conditional_coral(
            source_features=[[0.0], [1.0]],
            source_labels=["a", "b"],
            target_features=[[0.5]],
            target_pseudo_labels=["missing"],
        )


def test_coral_align_features_can_skip_target_mean_recentering() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    target = source + np.asarray([10.0, 20.0])
    source_stats = feature_stats(source)
    target_stats = feature_stats(target)

    centered = coral_align_features(source, source_stats=source_stats, target_stats=target_stats, center=False)
    recentered = coral_align_features(source, source_stats=source_stats, target_stats=target_stats, center=True)

    assert np.allclose(centered.mean(axis=0), source.mean(axis=0), atol=1e-5)
    assert np.allclose(recentered.mean(axis=0), target.mean(axis=0), atol=1e-5)


def test_config_aliases_and_validation() -> None:
    assert normalize_conditional_coral_fallback("strict") == "error"
    cfg = conditional_coral_config(confidence_threshold="0.25", fallback="global")
    assert cfg.confidence_threshold == 0.25

    with pytest.raises(ValueError, match="fallback"):
        normalize_conditional_coral_fallback("unknown")


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_pseudo_label_conditional_coral(
            source_features=[[0.0], [1.0]],
            source_labels=[0, 1],
            target_features=[[0.5], [1.5]],
            target_labels=[0, 1],  # type: ignore[call-arg]
        )
