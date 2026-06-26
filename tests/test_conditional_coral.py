from __future__ import annotations

import inspect

import numpy as np
import pytest

from neureptrace.decoding.conditional_coral import (
    CONDITIONAL_CORAL_CATEGORY,
    conditional_coral_config,
    coral_transform,
    fit_pseudo_conditional_coral,
)


def _toy_data():
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [-0.1, 0.1],
            [5.0, 5.0],
            [5.2, 5.1],
            [4.9, 4.8],
        ],
        dtype=float,
    )
    source_labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    target_features = np.asarray(
        [
            [1.0, 1.0],
            [1.2, 1.1],
            [0.9, 1.2],
            [8.0, 8.0],
            [8.3, 7.9],
            [7.8, 8.2],
        ],
        dtype=float,
    )
    pseudo = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    return source_features, source_labels, target_features, pseudo


def test_conditional_coral_aligns_source_classes_to_pseudo_target_means() -> None:
    source_features, source_labels, target_features, pseudo = _toy_data()

    result = fit_pseudo_conditional_coral(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        pseudo_labels=pseudo,
        config={"min_target_per_class": 2, "shrinkage": 0.1},
    )

    assert result.train_features.shape == source_features.shape
    assert result.test_features.shape == target_features.shape
    assert result.metadata["conditional_coral_protocol_category"] == CONDITIONAL_CORAL_CATEGORY
    assert result.metadata["conditional_coral_uses_target_features"] is True
    assert result.metadata["conditional_coral_uses_target_y"] is False
    assert result.metadata["conditional_coral_uses_pseudo_classes"] is True
    assert result.metadata["conditional_coral_valid_for_strict_source_only"] is False
    assert np.allclose(result.test_features, target_features)
    assert np.linalg.norm(result.train_features[source_labels == "a"].mean(axis=0) - target_features[pseudo == "a"].mean(axis=0)) < 0.35
    assert np.linalg.norm(result.train_features[source_labels == "b"].mean(axis=0) - target_features[pseudo == "b"].mean(axis=0)) < 0.35


def test_conditional_coral_accepts_probabilities_and_confidence_threshold() -> None:
    source_features, source_labels, target_features, _pseudo = _toy_data()
    probabilities = np.asarray(
        [
            [0.95, 0.05],
            [0.92, 0.08],
            [0.55, 0.45],
            [0.10, 0.90],
            [0.05, 0.95],
            [0.40, 0.60],
        ]
    )

    result = fit_pseudo_conditional_coral(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        pseudo_probabilities=probabilities,
        classes=["a", "b"],
        config={"confidence_threshold": 0.8, "min_target_per_class": 2},
    )

    assert result.pseudo_labels.tolist() == ["a", "a", "a", "b", "b", "b"]
    assert result.class_counts == {"a": 2, "b": 2}
    assert result.metadata["conditional_coral_n_target_supported_classes"] == 2


def test_conditional_coral_global_fallback_for_missing_pseudo_class() -> None:
    source_features, source_labels, target_features, _pseudo = _toy_data()
    pseudo = np.asarray(["a", "a", "a", "a", "a", "a"], dtype=object)

    result = fit_pseudo_conditional_coral(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        pseudo_labels=pseudo,
        config={"min_target_per_class": 2, "fallback_to_global": True},
    )

    assert result.class_counts == {"a": 6, "b": 0}
    assert result.metadata["conditional_coral_n_target_supported_classes"] == 1
    assert np.all(np.isfinite(result.train_features))


def test_coral_transform_moves_source_mean_to_target_mean() -> None:
    source_features, source_labels, target_features, pseudo = _toy_data()
    result = fit_pseudo_conditional_coral(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        pseudo_labels=pseudo,
    )
    source_stats = result.global_source_stats
    target_stats = result.global_target_stats
    transformed = coral_transform(source_features, source_stats, target_stats)

    assert np.linalg.norm(transformed.mean(axis=0) - target_features.mean(axis=0)) < 1e-5


def test_conditional_coral_rejects_unknown_pseudo_class() -> None:
    source_features, source_labels, target_features, _pseudo = _toy_data()

    with pytest.raises(ValueError, match="unknown"):
        fit_pseudo_conditional_coral(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            pseudo_labels=["a", "a", "a", "b", "b", "z"],
        )


def test_conditional_coral_rejects_missing_pseudo_inputs() -> None:
    source_features, source_labels, target_features, _pseudo = _toy_data()

    with pytest.raises(ValueError, match="pseudo"):
        fit_pseudo_conditional_coral(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
        )


def test_conditional_coral_config_validation() -> None:
    cfg = conditional_coral_config(confidence_threshold="0.5", min_target_per_class="2")
    assert cfg.confidence_threshold == 0.5
    assert cfg.min_target_per_class == 2

    with pytest.raises(ValueError, match="confidence_threshold"):
        conditional_coral_config(confidence_threshold=1.5)


def test_public_api_has_no_scored_target_label_argument() -> None:
    signature = inspect.signature(fit_pseudo_conditional_coral)
    assert "target_labels" not in signature.parameters
    assert "target_y" not in signature.parameters
