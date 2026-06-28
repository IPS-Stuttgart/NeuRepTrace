from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scaling import (
    SOURCE_SCALING_CATEGORY,
    augment_source_with_feature_scaling,
    normalize_scaling_distribution,
    normalize_scaling_mode,
    sample_scaling_factors,
    source_feature_scaling_config,
)


def test_source_feature_scaling_appends_rows_and_metadata() -> None:
    features = np.asarray([[1.0, 2.0], [2.0, 4.0], [10.0, 20.0], [12.0, 24.0]], dtype=float)
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)

    result = augment_source_with_feature_scaling(
        features,
        labels,
        config={"synthetic_per_class": 2, "scale_std": 0.2, "random_state": 7},
    )

    assert result.features.shape == (8, 2)
    assert result.labels.shape == (8,)
    assert result.synthetic_mask.tolist() == [False, False, False, False, True, True, True, True]
    assert result.n_synthetic == 4
    assert result.scale_factors.shape == (4, 2)
    assert result.metadata["source_feature_scaling_protocol_category"] == SOURCE_SCALING_CATEGORY
    assert result.metadata["source_feature_scaling_valid_for_strict_source_only"] is True


def test_source_feature_scaling_preserves_composite_labels_and_domains() -> None:
    features = np.asarray([[1.0, 2.0], [2.0, 4.0], [10.0, 20.0], [12.0, 24.0]], dtype=float)
    labels = [("face", "left"), ("face", "left"), ("scene", "right"), ("scene", "right")]
    domains = [("subject-1", "run-1"), ("subject-1", "run-1"), ("subject-2", "run-1"), ("subject-2", "run-1")]

    result = augment_source_with_feature_scaling(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "scale_std": 0.1, "random_state": 11},
    )

    assert result.labels.shape == (6,)
    assert result.labels.tolist() == labels + [("face", "left"), ("scene", "right")]
    assert result.metadata["source_feature_scaling_n_classes"] == 2
    assert result.metadata["source_feature_scaling_n_source_domains"] == 2


def test_row_scaling_uses_same_factor_for_all_features() -> None:
    rng = np.random.default_rng(13)
    factors = sample_scaling_factors(5, scale_std=0.1, scaling_mode="row", rng=rng)

    assert factors.shape == (5,)
    assert np.allclose(factors, factors[0])
    assert np.all(factors > 0.0)


def test_feature_scaling_can_use_different_factors() -> None:
    rng = np.random.default_rng(13)
    factors = sample_scaling_factors(8, scale_std=0.2, scaling_mode="feature", distribution="uniform", rng=rng)

    assert factors.shape == (8,)
    assert np.all(factors > 0.0)
    assert not np.allclose(factors, factors[0])


def test_preserve_original_false_returns_only_generated_rows() -> None:
    features = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    labels = np.asarray(["x", "x"], dtype=object)

    result = augment_source_with_feature_scaling(
        features,
        labels,
        config={"synthetic_per_class": 3, "preserve_original": False, "random_state": 3},
    )

    assert result.features.shape == (3, 2)
    assert result.synthetic_mask.tolist() == [True, True, True]
    assert result.labels.tolist() == ["x", "x", "x"]


def test_disabled_scaling_returns_original_rows_only() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=float)
    labels = np.asarray([0, 1])

    result = augment_source_with_feature_scaling(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == [0, 1]
    assert not np.any(result.synthetic_mask)
    assert result.n_synthetic == 0


def test_scaling_is_reproducible_with_fixed_seed() -> None:
    features = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    config = source_feature_scaling_config(synthetic_per_class=3, random_state=42)

    first = augment_source_with_feature_scaling(features, labels, config=config)
    second = augment_source_with_feature_scaling(features, labels, config=config)

    assert np.allclose(first.features, second.features)
    assert first.labels.tolist() == second.labels.tolist()
    assert first.content_indices.tolist() == second.content_indices.tolist()
    assert np.allclose(first.scale_factors, second.scale_factors)


def test_aliases_and_invalid_options() -> None:
    assert normalize_scaling_mode("global") == "row"
    assert normalize_scaling_mode("column") == "feature"
    assert normalize_scaling_distribution("gaussian") == "normal"

    with pytest.raises(ValueError):
        source_feature_scaling_config(scale_std=-0.1)

    with pytest.raises(ValueError):
        normalize_scaling_mode("bad")

    with pytest.raises(ValueError):
        normalize_scaling_distribution("bad")
