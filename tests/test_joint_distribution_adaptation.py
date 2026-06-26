from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.joint_distribution_adaptation import (
    JDA_CATEGORY,
    fit_joint_distribution_adaptation,
    joint_distribution_adaptation_config,
    normalize_jda_method,
    transform_joint_distribution_features,
)


def _two_class_data():
    source = np.asarray(
        [
            [-2.2, -0.2],
            [-2.0, 0.1],
            [-1.8, 0.0],
            [1.8, 0.0],
            [2.0, -0.1],
            [2.2, 0.2],
        ],
        dtype=float,
    )
    labels = np.asarray(["left", "left", "left", "right", "right", "right"], dtype=object)
    target = np.asarray([[-1.7, 0.2], [-1.5, -0.1], [2.3, 0.1], [2.5, -0.2]], dtype=float)
    return source, labels, target


def test_jda_projects_rows_and_records_category_two_metadata() -> None:
    source, labels, target = _two_class_data()

    result = fit_joint_distribution_adaptation(
        source,
        labels,
        target,
        n_components=2,
        max_iterations=5,
    )

    assert result.source_features.shape == (6, 2)
    assert result.target_features.shape == (4, 2)
    assert result.projection.shape == (2, 2)
    assert result.target_probabilities.shape == (4, 2)
    assert np.allclose(result.target_probabilities.sum(axis=1), 1.0)
    assert result.classes == ("left", "right")
    assert result.metadata["jda_protocol_category"] == JDA_CATEGORY
    assert result.metadata["jda_uses_source_labels"] is True
    assert result.metadata["jda_uses_target_features"] is True
    assert result.metadata["jda_uses_target_labels"] is False
    assert result.metadata["jda_valid_for_strict_source_only"] is False
    assert result.target_pseudo_labels[:2].tolist() == ["left", "left"]
    assert result.target_pseudo_labels[2:].tolist() == ["right", "right"]


def test_soft_jda_accepts_source_model_target_probabilities() -> None:
    source, labels, target = _two_class_data()
    probabilities = np.asarray(
        [
            [0.95, 0.05],
            [0.85, 0.15],
            [0.10, 0.90],
            [0.05, 0.95],
        ]
    )

    result = fit_joint_distribution_adaptation(
        source,
        labels,
        target,
        method="soft-jda",
        target_probabilities=probabilities,
        n_components=1,
        max_iterations=3,
    )

    assert result.metadata["jda_method"] == "soft_jda"
    assert result.metadata["jda_uses_target_probabilities"] is True
    assert result.target_probabilities.shape == probabilities.shape
    assert np.all(result.target_probabilities >= 0.0)
    assert np.allclose(result.target_probabilities.sum(axis=1), 1.0)


def test_transform_reuses_fitted_projection() -> None:
    source, labels, target = _two_class_data()
    result = fit_joint_distribution_adaptation(source, labels, target, n_components=1)

    transformed = transform_joint_distribution_features(target[:2], result)

    assert transformed.shape == (2, 1)
    assert np.allclose(transformed, result.target_features[:2])


def test_composite_source_labels_are_preserved() -> None:
    source, _labels, target = _two_class_data()
    labels = [("left", 1)] * 3 + [("right", 2)] * 3

    result = fit_joint_distribution_adaptation(source, labels, target, n_components=1)

    assert result.classes == (("left", 1), ("right", 2))
    assert result.target_pseudo_labels[0] == ("left", 1)
    assert result.target_pseudo_labels[-1] == ("right", 2)


def test_jda_is_deterministic() -> None:
    source, labels, target = _two_class_data()

    first = fit_joint_distribution_adaptation(source, labels, target, method="soft_jda", n_components=2)
    second = fit_joint_distribution_adaptation(source, labels, target, method="soft_jda", n_components=2)

    assert np.allclose(first.source_features, second.source_features)
    assert np.allclose(first.target_features, second.target_features)
    assert np.allclose(first.projection, second.projection)
    assert first.target_pseudo_labels.tolist() == second.target_pseudo_labels.tolist()


def test_jda_config_and_aliases() -> None:
    assert normalize_jda_method("joint-distribution-adaptation") == "jda"
    assert normalize_jda_method("probabilistic-jda") == "soft_jda"

    config = joint_distribution_adaptation_config(
        method="soft",
        n_components="all",
        max_iterations="4",
        conditional_weight="0.5",
    )
    assert config.method == "soft_jda"
    assert config.n_components == "all"
    assert config.max_iterations == 4
    assert np.isclose(config.conditional_weight, 0.5)


def test_jda_guardrails_and_target_label_rejection() -> None:
    source, labels, target = _two_class_data()

    with pytest.raises(ValueError, match="same feature width"):
        fit_joint_distribution_adaptation(source, labels, target[:, :1])
    with pytest.raises(ValueError, match="shape"):
        fit_joint_distribution_adaptation(source, labels, target, target_probabilities=[[0.5, 0.5]])
    with pytest.raises(ValueError, match="at least two source classes"):
        fit_joint_distribution_adaptation(source, ["one"] * source.shape[0], target)
    with pytest.raises(ValueError, match="omit source label"):
        fit_joint_distribution_adaptation(source, labels, target, classes=["left"])
    with pytest.raises(TypeError):
        fit_joint_distribution_adaptation(
            source,
            labels,
            target,
            target_labels=["left", "left", "right", "right"],  # type: ignore[call-arg]
        )
