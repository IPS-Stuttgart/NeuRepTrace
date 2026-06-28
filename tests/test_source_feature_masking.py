from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_masking import (
    SOURCE_MASKING_CATEGORY,
    augment_source_with_feature_masking,
    feature_mask_indices,
    normalize_fill_mode,
    normalize_mask_mode,
    source_feature_masking_config,
)


def test_source_feature_masking_appends_synthetic_rows() -> None:
    features = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 11.0, 12.0, 13.0],
            [11.0, 12.0, 13.0, 14.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s1", "s2"], dtype=object)

    result = augment_source_with_feature_masking(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 2, "mask_fraction": 0.5, "random_state": 7},
    )

    assert result.features.shape == (8, 4)
    assert result.labels.shape == (8,)
    assert result.synthetic_mask.tolist() == [False, False, False, False, True, True, True, True]
    assert result.n_synthetic == 4
    assert len(result.masked_feature_indices) == 4
    assert all(mask.size == 2 for mask in result.masked_feature_indices)
    assert result.metadata["source_feature_masking_protocol_category"] == SOURCE_MASKING_CATEGORY
    assert result.metadata["source_feature_masking_uses_heldout_features"] is False
    assert result.metadata["source_feature_masking_uses_heldout_labels"] is False
    assert result.metadata["source_feature_masking_valid_for_strict_source_only"] is True


def test_feature_mean_fill_replaces_masked_values_with_column_means() -> None:
    features = np.asarray([[0.0, 10.0, 20.0], [2.0, 12.0, 22.0]], dtype=float)
    labels = np.asarray([0, 0])

    result = augment_source_with_feature_masking(
        features,
        labels,
        config={
            "synthetic_per_class": 1,
            "mask_fraction": 1.0,
            "fill_mode": "feature_mean",
            "random_state": 3,
        },
    )

    assert np.allclose(result.features[-1], np.asarray([1.0, 11.0, 21.0]))


def test_zero_fill_without_preserving_original_returns_only_synthetic_rows() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    labels = np.asarray(["x", "x"], dtype=object)

    result = augment_source_with_feature_masking(
        features,
        labels,
        config={"synthetic_per_class": 2, "mask_fraction": 1.0, "fill_mode": "zero", "preserve_original": False},
    )

    assert result.features.shape == (2, 2)
    assert np.allclose(result.features, 0.0)
    assert result.synthetic_mask.tolist() == [True, True]
    assert result.labels.tolist() == ["x", "x"]


def test_source_feature_masking_preserves_composite_labels_and_domains() -> None:
    features = np.arange(20, dtype=float).reshape(4, 5)
    labels = [("cat", 1), ("cat", 1), ("dog", 2), ("dog", 2)]
    domains = [("s1", "run1"), ("s1", "run2"), ("s2", "run1"), ("s2", "run2")]

    result = augment_source_with_feature_masking(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "mask_fraction": 0.4, "random_state": 11},
    )

    assert result.features.shape == (6, 5)
    assert result.labels.shape == (6,)
    assert result.labels.tolist()[:4] == labels
    assert result.labels.tolist()[4:] == [("cat", 1), ("dog", 2)]
    assert result.metadata["source_feature_masking_n_classes"] == 2
    assert result.metadata["source_feature_masking_n_source_domains"] == 4


def test_block_mask_indices_are_contiguous() -> None:
    rng = np.random.default_rng(13)
    mask = feature_mask_indices(10, mask_fraction=0.5, mask_mode="block", block_size=3, rng=rng)

    assert mask.size == 3
    assert np.all(np.diff(mask) == 1)


def test_disabled_masking_returns_original_rows_only() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    labels = np.asarray([0, 1])

    result = augment_source_with_feature_masking(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == [0, 1]
    assert not np.any(result.synthetic_mask)
    assert result.n_synthetic == 0


def test_masking_is_reproducible_with_fixed_seed() -> None:
    features = np.arange(24, dtype=float).reshape(6, 4)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    config = source_feature_masking_config(synthetic_per_class=3, mask_fraction=0.5, random_state=42)

    first = augment_source_with_feature_masking(features, labels, config=config)
    second = augment_source_with_feature_masking(features, labels, config=config)

    assert np.allclose(first.features, second.features)
    assert first.labels.tolist() == second.labels.tolist()
    assert first.content_indices.tolist() == second.content_indices.tolist()
    assert [mask.tolist() for mask in first.masked_feature_indices] == [mask.tolist() for mask in second.masked_feature_indices]


def test_aliases_and_invalid_options() -> None:
    assert normalize_mask_mode("contiguous") == "block"
    assert normalize_fill_mode("column-mean") == "feature_mean"
    assert normalize_fill_mode("trial_mean") == "row_mean"

    with pytest.raises(ValueError, match="mask_fraction"):
        source_feature_masking_config(mask_fraction=1.5)

    with pytest.raises(ValueError, match="mask_mode"):
        normalize_mask_mode("bad")

    with pytest.raises(ValueError, match="fill_mode"):
        normalize_fill_mode("bad")


def test_extra_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        augment_source_with_feature_masking(
            [[0.0], [1.0]],
            [0, 1],
            heldout_features=[[0.5]],  # type: ignore[call-arg]
        )
