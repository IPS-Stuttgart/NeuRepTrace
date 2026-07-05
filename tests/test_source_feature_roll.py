from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_roll import (
    SOURCE_ROLL_CATEGORY,
    augment_source_with_feature_roll,
    normalize_roll_mode,
    roll_feature_row,
    sample_roll_shift,
    source_feature_roll_config,
)


def test_source_feature_roll_appends_synthetic_rows() -> None:
    features = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [11.0, 12.0, 13.0]], dtype=float)
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s1", "s2"], dtype=object)

    result = augment_source_with_feature_roll(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 2, "max_shift": 1, "random_state": 7},
    )

    assert result.features.shape == (8, 3)
    assert result.labels.shape == (8,)
    assert result.synthetic_mask.tolist() == [False, False, False, False, True, True, True, True]
    assert result.n_synthetic == 4
    assert np.all(result.shifts != 0)
    assert result.metadata["source_feature_roll_protocol_category"] == SOURCE_ROLL_CATEGORY
    assert result.metadata["source_feature_roll_uses_heldout_features"] is False
    assert result.metadata["source_feature_roll_uses_heldout_labels"] is False
    assert result.metadata["source_feature_roll_valid_for_strict_source_only"] is True


def test_source_feature_roll_counts_mixed_hashable_domains_without_sorting() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=float)
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    domains = np.asarray(["s1", 1, ("session", 2), "s1"], dtype=object)

    result = augment_source_with_feature_roll(features, labels, source_domains=domains)

    assert result.metadata["source_feature_roll_n_source_domains"] == 3


def test_roll_feature_row_circular_and_constant_modes() -> None:
    row = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert np.allclose(roll_feature_row(row, shift=1, mode="circular"), np.asarray([4.0, 1.0, 2.0, 3.0]))
    assert np.allclose(roll_feature_row(row, shift=-2, mode="constant", fill_value=-1.0), np.asarray([3.0, 4.0, -1.0, -1.0]))


def test_source_feature_roll_can_return_only_synthetic_rows() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=object)

    result = augment_source_with_feature_roll(
        features,
        labels,
        config={"synthetic_per_class": 3, "max_shift": 1, "preserve_original": False, "random_state": 3},
    )

    assert result.features.shape == (6, 2)
    assert result.synthetic_mask.tolist() == [True] * 6
    assert result.labels.tolist().count(0) == 3
    assert result.labels.tolist().count(1) == 3


def test_source_feature_roll_disabled_returns_original_rows() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    labels = np.asarray([0, 1])

    result = augment_source_with_feature_roll(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == [0, 1]
    assert result.n_synthetic == 0
    assert result.shifts.shape == (0,)


def test_roll_shift_sampling_can_exclude_zero() -> None:
    rng = np.random.default_rng(13)
    shifts = [sample_roll_shift(2, include_zero_shift=False, rng=rng) for _ in range(50)]

    assert all(shift in {-2, -1, 1, 2} for shift in shifts)


def test_roll_config_aliases_and_validation() -> None:
    assert normalize_roll_mode("wrap") == "circular"
    assert normalize_roll_mode("zero-pad") == "constant"
    assert source_feature_roll_config(include_zero_shift="yes").include_zero_shift is True
    assert source_feature_roll_config(preserve_original="false").preserve_original is False

    with pytest.raises(ValueError, match="roll_mode"):
        normalize_roll_mode("bad")

    with pytest.raises(ValueError, match="max_shift"):
        source_feature_roll_config(max_shift=0)
