from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.kernel_mean_matching import (
    KMM_PROTOCOL_CATEGORY,
    kernel_mean_matching_weights,
    kernel_mean_matching_weights_from_config,
    kmm_config,
    normalize_kmm_epsilon,
    normalize_kmm_kernel,
    resolve_kmm_gamma,
)


def test_kmm_upweights_source_rows_near_unlabeled_target_distribution() -> None:
    source = np.asarray(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [4.0, 4.0],
            [4.2, 3.8],
            [3.8, 4.1],
        ],
        dtype=float,
    )
    target = np.asarray([[0.0, 0.1], [0.2, -0.1], [-0.2, 0.0]], dtype=float)

    result = kernel_mean_matching_weights(
        source,
        target,
        gamma=1.0,
        max_weight=8.0,
        epsilon=None,
        regularization=1e-6,
    )

    near_weight = float(np.mean(result.weights[:3]))
    far_weight = float(np.mean(result.weights[3:]))
    assert near_weight > far_weight
    assert np.isclose(np.mean(result.weights), 1.0)
    assert result.metadata["kmm_protocol_category"] == KMM_PROTOCOL_CATEGORY
    assert result.metadata["kmm_uses_target_features"] is True
    assert result.metadata["kmm_uses_target_labels"] is False
    assert result.metadata["kmm_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["kmm_valid_for_strict_source_only"] is False


def test_kmm_linear_kernel_returns_finite_mean_one_weights() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    target = np.asarray([[2.0], [2.5], [3.0]], dtype=float)

    result = kernel_mean_matching_weights(source, target, kernel="linear", epsilon="auto", max_weight=5.0)

    assert result.weights.shape == (4,)
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0.0)
    assert np.isclose(np.mean(result.weights), 1.0)
    assert result.metadata["kmm_kernel"] == "linear"
    assert result.metadata["kmm_gamma"] == ""


def test_kmm_class_balance_equalizes_source_class_weight_mass() -> None:
    source = np.asarray([[0.0], [0.1], [0.2], [0.3], [3.0], [3.1]], dtype=float)
    target = np.asarray([[0.0], [0.15], [3.0]], dtype=float)
    labels = np.asarray(["major", "major", "major", "major", "minor", "minor"], dtype=object)

    result = kernel_mean_matching_weights(
        source,
        target,
        gamma=2.0,
        epsilon=None,
        source_labels=labels,
        class_balance=True,
    )

    major_mass = float(np.sum(result.weights[labels == "major"]))
    minor_mass = float(np.sum(result.weights[labels == "minor"]))
    assert np.isclose(major_mass, minor_mass)
    assert np.isclose(np.mean(result.weights), 1.0)
    assert result.metadata["kmm_uses_source_labels"] is True


def test_kmm_class_balance_preserves_composite_source_labels() -> None:
    source = np.asarray([[0.0], [0.1], [0.2], [0.3], [3.0], [3.1]], dtype=float)
    target = np.asarray([[0.0], [0.15], [3.0]], dtype=float)
    labels = [("major", "left"), ("major", "left"), ("major", "left"), ("major", "left"), ("minor", "right"), ("minor", "right")]

    result = kernel_mean_matching_weights(
        source,
        target,
        gamma=2.0,
        epsilon=None,
        source_labels=labels,
        class_balance=True,
    )

    major_mask = np.asarray([label == ("major", "left") for label in labels], dtype=bool)
    minor_mask = np.asarray([label == ("minor", "right") for label in labels], dtype=bool)
    assert np.isclose(float(np.sum(result.weights[major_mask])), float(np.sum(result.weights[minor_mask])))
    assert np.isclose(np.mean(result.weights), 1.0)
    assert result.metadata["kmm_uses_source_labels"] is True


def test_kmm_from_config_mapping_and_aliases() -> None:
    source = np.asarray([[0.0], [1.0], [3.0]], dtype=float)
    target = np.asarray([[0.1], [0.2]], dtype=float)

    result = kernel_mean_matching_weights_from_config(
        source,
        target,
        {"kernel": "gaussian", "gamma": "scale", "epsilon": "off", "max_weight": "4", "max_iter": 200},
    )

    assert result.weights.shape == (3,)
    assert result.metadata["kmm_kernel"] == "rbf"
    assert np.isclose(result.metadata["kmm_gamma"], 1.0)
    assert result.metadata["kmm_epsilon"] == ""


def test_kmm_config_normalizes_values() -> None:
    config = kmm_config(kernel="linear-kernel", max_weight="3", epsilon="auto", regularization="0.01", max_iter="25")

    assert config.kernel == "linear"
    assert np.isclose(config.max_weight, 3.0)
    assert np.isclose(config.regularization, 0.01)
    assert config.max_iter == 25
    assert normalize_kmm_kernel("dot") == "linear"
    assert normalize_kmm_epsilon("auto", n_source=4) == pytest.approx(0.5)


def test_kmm_gamma_median_and_scale() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    target = np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=float)

    gamma_median = resolve_kmm_gamma("median", source, target)
    gamma_scale = resolve_kmm_gamma("scale", source, target)

    assert gamma_median > 0.0
    assert gamma_scale == pytest.approx(0.5)


def test_kmm_rejects_mismatched_feature_width() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        kernel_mean_matching_weights([[0.0, 1.0]], [[0.0]])


def test_kmm_requires_source_labels_for_class_balance() -> None:
    with pytest.raises(ValueError, match="source_labels"):
        kernel_mean_matching_weights([[0.0], [1.0]], [[0.0]], class_balance=True)


def test_kmm_public_api_rejects_target_labels_argument() -> None:
    with pytest.raises(TypeError):
        kernel_mean_matching_weights(
            [[0.0], [1.0]],
            [[0.0]],
            target_labels=[0],  # type: ignore[call-arg]
        )
