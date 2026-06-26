from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.joint_distribution_adaptation import fit_joint_distribution_adaptation, joint_distribution_adaptation_config


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("false", False),
        ("0", False),
        ("off", False),
        (0, False),
        (0.0, False),
        ("true", True),
        ("1", True),
        ("on", True),
        (1, True),
        (1.0, True),
    ],
)
def test_jda_config_normalizes_boolean_flags(value, expected) -> None:
    config = joint_distribution_adaptation_config(standardize=value, normalize_latent=value)

    assert config.standardize is expected
    assert config.normalize_latent is expected


def test_jda_string_false_standardize_disables_standardization() -> None:
    source, labels, target = _two_class_data()

    result = fit_joint_distribution_adaptation(
        source,
        labels,
        target,
        standardize="false",
        normalize_latent="false",
        n_components=1,
        max_iterations=1,
    )

    assert np.allclose(result.feature_mean, np.zeros(source.shape[1]))
    assert np.allclose(result.feature_scale, np.ones(source.shape[1]))


@pytest.mark.parametrize("field", ["standardize", "normalize_latent"])
def test_jda_config_rejects_ambiguous_boolean_strings(field: str) -> None:
    with pytest.raises(ValueError, match="boolean"):
        joint_distribution_adaptation_config(**{field: "maybe"})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("n_components", "positive integer"),
        ("max_iterations", "positive integer"),
        ("conditional_weight", "finite and non-negative"),
        ("regularization", "finite and non-negative"),
        ("eigen_ridge", "positive and finite"),
        ("temperature", "positive and finite"),
    ],
)
def test_jda_config_rejects_boolean_numeric_fields(field: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        joint_distribution_adaptation_config(**{field: True})
