from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_balancing import (
    SOURCE_BALANCING_CATEGORY,
    balance_source_classes,
    normalize_balancing_mode,
    resolve_target_count,
    source_class_balancing_config,
)


def test_oversample_balances_to_majority_class() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0], [12.0]], dtype=float)
    labels = np.asarray(["minor", "minor", "major", "major", "major"], dtype=object)

    result = balance_source_classes(features, labels, config={"mode": "oversample", "random_state": 7})

    assert result.features.shape == (6, 1)
    assert result.class_counts_before == {"minor": 2, "major": 3}
    assert result.class_counts_after == {"minor": 3, "major": 3}
    assert result.n_rows == 6
    assert np.count_nonzero(result.synthetic_mask) == 1
    assert result.metadata["source_class_balancing_protocol_category"] == SOURCE_BALANCING_CATEGORY
    assert result.metadata["source_class_balancing_valid_for_strict_source_only"] is True
    assert result.metadata["source_class_balancing_uses_heldout_features"] is False
    assert result.metadata["source_class_balancing_uses_heldout_labels"] is False


def test_undersample_balances_to_minority_class() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0], [12.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1, 1], dtype=object)

    result = balance_source_classes(features, labels, config={"mode": "undersample", "target_count": "min", "random_state": 3})

    assert result.features.shape == (4, 1)
    assert result.class_counts_after == {0: 2, 1: 2}
    assert not np.any(result.synthetic_mask)
    assert np.all(result.sample_weight == 1.0)


def test_weight_mode_keeps_rows_and_returns_inverse_frequency_weights() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0], [12.0]], dtype=float)
    labels = np.asarray(["minor", "minor", "major", "major", "major"], dtype=object)

    result = balance_source_classes(features, labels, config={"mode": "weights"})

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == labels.tolist()
    assert result.selected_indices.tolist() == [0, 1, 2, 3, 4]
    assert result.class_counts_after == result.class_counts_before
    assert np.isclose(result.sample_weight[0], 5.0 / 4.0)
    assert np.isclose(result.sample_weight[2], 5.0 / 6.0)


def test_composite_tuple_labels_are_atomic_for_resampling_and_weights() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0], [12.0]], dtype=float)
    labels = [("face", "early"), ("face", "early"), ("object", "late"), ("object", "late"), ("object", "late")]

    rows = balance_source_classes(features, labels, config={"mode": "oversample", "random_state": 7})
    weighted = balance_source_classes(features, labels, config={"mode": "weights"})

    assert rows.classes.tolist() == [("face", "early"), ("object", "late")]
    assert rows.class_counts_before == {("face", "early"): 2, ("object", "late"): 3}
    assert rows.class_counts_after == {("face", "early"): 3, ("object", "late"): 3}
    assert rows.labels.tolist().count(("face", "early")) == 3
    assert rows.labels.tolist().count(("object", "late")) == 3
    assert weighted.labels.tolist() == labels
    assert np.isclose(weighted.sample_weight[0], 5.0 / 4.0)
    assert np.isclose(weighted.sample_weight[2], 5.0 / 6.0)


def test_target_count_aliases_and_mode_aliases() -> None:
    assert resolve_target_count([2, 5, 8], "median") == 5
    assert resolve_target_count([2, 5, 8], "mean") == 5
    assert resolve_target_count([2, 5, 8], 3) == 3
    assert normalize_balancing_mode("upsample") == "oversample"
    assert normalize_balancing_mode("downsample") == "undersample"
    assert normalize_balancing_mode("inverse-frequency") == "weights"


def test_config_validation() -> None:
    cfg = source_class_balancing_config(mode="over", target_count="max", preserve_order=True)
    assert cfg.mode == "oversample"
    assert cfg.preserve_order is True

    with pytest.raises(ValueError, match="balancing mode"):
        source_class_balancing_config(mode="bad")

    with pytest.raises(ValueError, match="target_count"):
        resolve_target_count([1, 2], 0)


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        balance_source_classes(
            [[0.0], [1.0]],
            [0, 1],
            heldout_features=[[0.5]],  # type: ignore[call-arg]
        )
