from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_jitter import (
    SOURCE_JITTER_CATEGORY,
    augment_source_with_feature_jitter,
    normalize_jitter_scale_mode,
    source_feature_jitter_config,
)


def test_source_feature_jitter_appends_synthetic_rows() -> None:
    features = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s1", "s2"], dtype=object)

    result = augment_source_with_feature_jitter(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 2, "noise_scale": 0.1, "random_state": 7},
    )

    assert result.features.shape == (8, 3)
    assert result.labels.shape == (8,)
    assert result.synthetic_mask.tolist() == [False, False, False, False, True, True, True, True]
    assert result.noise.shape == (4, 3)
    assert result.n_synthetic == 4
    assert result.metadata["source_feature_jitter_protocol_category"] == SOURCE_JITTER_CATEGORY
    assert result.metadata["source_feature_jitter_uses_heldout_features"] is False
    assert result.metadata["source_feature_jitter_uses_heldout_labels"] is False
    assert result.metadata["source_feature_jitter_valid_for_strict_source_only"] is True


def test_unit_scale_jitter_matches_recorded_noise() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    labels = np.asarray(["x", "x"], dtype=object)

    result = augment_source_with_feature_jitter(
        features,
        labels,
        config={"synthetic_per_class": 1, "noise_scale": 0.25, "scale_mode": "unit", "random_state": 1},
    )

    original_row = features[result.content_indices[0]]
    assert np.allclose(result.features[-1], original_row + result.noise[0])


def test_jitter_can_return_only_synthetic_rows() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=object)

    result = augment_source_with_feature_jitter(
        features,
        labels,
        config={"synthetic_per_class": 3, "preserve_original": False, "random_state": 3},
    )

    assert result.features.shape == (6, 1)
    assert result.synthetic_mask.tolist() == [True] * 6
    assert result.labels.tolist().count(0) == 3
    assert result.labels.tolist().count(1) == 3


def test_string_false_preserve_original_returns_only_synthetic_rows() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=object)

    result = augment_source_with_feature_jitter(
        features,
        labels,
        config={"synthetic_per_class": 2, "preserve_original": "false", "random_state": 3},
    )

    assert result.features.shape == (4, 1)
    assert result.synthetic_mask.tolist() == [True] * 4
    assert result.metadata["source_feature_jitter_preserve_original"] is False
    assert result.metadata["source_feature_jitter_n_output_rows"] == 4


def test_composite_labels_and_domains_are_preserved_as_row_values() -> None:
    features = np.asarray(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [10.0, 10.1],
            [10.2, 10.3],
        ],
        dtype=float,
    )
    labels = np.asarray([("visual", 1), ("visual", 1), ("auditory", 2), ("auditory", 2)], dtype=object)
    domains = np.asarray([("subject", 1), ("subject", 1), ("subject", 2), ("subject", 2)], dtype=object)

    result = augment_source_with_feature_jitter(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "scale_mode": "class", "random_state": 5},
    )

    assert result.features.shape == (6, 2)
    assert result.labels.shape == (6,)
    assert result.metadata["source_feature_jitter_n_classes"] == 2
    assert result.metadata["source_feature_jitter_n_source_domains"] == 2
    assert all(isinstance(label, tuple) for label in result.labels.tolist())
    assert result.labels.tolist().count(("visual", 1)) == 3
    assert result.labels.tolist().count(("auditory", 2)) == 3


def test_disabled_jitter_returns_original_rows_only() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    labels = np.asarray([0, 1])

    result = augment_source_with_feature_jitter(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == [0, 1]
    assert result.n_synthetic == 0
    assert result.noise.shape == (0, 2)


def test_jitter_is_reproducible_with_fixed_seed() -> None:
    features = np.arange(24, dtype=float).reshape(6, 4)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    config = source_feature_jitter_config(synthetic_per_class=3, noise_scale=0.05, random_state=42)

    first = augment_source_with_feature_jitter(features, labels, config=config)
    second = augment_source_with_feature_jitter(features, labels, config=config)

    assert np.allclose(first.features, second.features)
    assert first.labels.tolist() == second.labels.tolist()
    assert first.content_indices.tolist() == second.content_indices.tolist()
    assert np.allclose(first.noise, second.noise)


def test_scale_mode_aliases_and_invalid_values() -> None:
    assert normalize_jitter_scale_mode("pooled") == "global"
    assert normalize_jitter_scale_mode("per-class") == "class"
    assert normalize_jitter_scale_mode("none") == "unit"

    with pytest.raises(ValueError, match="scale_mode"):
        normalize_jitter_scale_mode("bad")

    with pytest.raises(ValueError, match="noise_scale"):
        source_feature_jitter_config(noise_scale=-1.0)

    with pytest.raises(ValueError, match="preserve_original"):
        source_feature_jitter_config(preserve_original="maybe")
