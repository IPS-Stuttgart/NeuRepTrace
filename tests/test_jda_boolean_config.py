from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.joint_distribution_adaptation import (
    fit_joint_distribution_adaptation,
    joint_distribution_adaptation_config,
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


def test_jda_config_string_boolean_values_are_normalized() -> None:
    false_config = joint_distribution_adaptation_config(standardize="false", normalize_latent="off")
    assert false_config.standardize is False
    assert false_config.normalize_latent is False

    true_config = joint_distribution_adaptation_config(standardize="yes", normalize_latent="1")
    assert true_config.standardize is True
    assert true_config.normalize_latent is True


def test_jda_fit_honors_string_false_standardize_from_config() -> None:
    source, labels, target = _two_class_data()

    result = fit_joint_distribution_adaptation(
        source,
        labels,
        target,
        config={
            "n_components": 1,
            "max_iterations": 1,
            "standardize": "false",
            "normalize_latent": "false",
        },
    )

    assert np.allclose(result.feature_mean, np.zeros(source.shape[1]))
    assert np.allclose(result.feature_scale, np.ones(source.shape[1]))
    assert result.source_features.shape == (source.shape[0], 1)
    assert result.target_features.shape == (target.shape[0], 1)


def test_jda_fit_honors_string_false_standardize_argument() -> None:
    source, labels, target = _two_class_data()

    result = fit_joint_distribution_adaptation(
        source,
        labels,
        target,
        n_components=1,
        max_iterations=1,
        standardize="off",  # type: ignore[arg-type]
    )

    assert np.allclose(result.feature_mean, np.zeros(source.shape[1]))
    assert np.allclose(result.feature_scale, np.ones(source.shape[1]))


def test_jda_boolean_options_reject_ambiguous_strings() -> None:
    source, labels, target = _two_class_data()

    with pytest.raises(ValueError, match="standardize"):
        joint_distribution_adaptation_config(standardize="maybe")
    with pytest.raises(ValueError, match="normalize_latent"):
        fit_joint_distribution_adaptation(
            source,
            labels,
            target,
            n_components=1,
            max_iterations=1,
            normalize_latent="maybe",  # type: ignore[arg-type]
        )
