from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_roll import augment_source_with_feature_roll


def _is_nan_label(value: object) -> bool:
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def test_source_feature_roll_treats_nan_labels_as_matching_class_values() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]], dtype=float)
    labels = np.asarray([np.nan, np.float64("nan"), "known", "known"], dtype=object)

    result = augment_source_with_feature_roll(
        features,
        labels,
        config={"synthetic_per_class": 1, "max_shift": 1, "preserve_original": False, "random_state": 0},
    )

    values = result.labels.tolist()
    assert result.features.shape == (2, 2)
    assert result.n_synthetic == 2
    assert sum(_is_nan_label(value) for value in values) == 1
    assert values.count("known") == 1
    assert result.metadata["source_feature_roll_n_classes"] == 2


@pytest.mark.parametrize("temporal_nat", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_source_feature_roll_keeps_temporal_nat_separate_from_none(temporal_nat: object) -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]], dtype=float)
    labels = np.empty(4, dtype=object)
    labels[:] = [None, None, temporal_nat, temporal_nat]

    result = augment_source_with_feature_roll(
        features,
        labels,
        config={"synthetic_per_class": 1, "max_shift": 1, "preserve_original": False, "random_state": 0},
    )

    values = result.labels.tolist()
    assert result.features.shape == (2, 2)
    assert result.n_synthetic == 2
    assert values[0] is None
    assert type(values[1]) is type(temporal_nat)
    assert np.isnat(values[1])
    assert result.metadata["source_feature_roll_n_classes"] == 2


def test_source_feature_roll_preserves_rectangular_composite_label_rows() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]], dtype=float)
    labels = np.asarray([["cat", 1], ["cat", 1], ["dog", 2], ["dog", 2]], dtype=object)

    result = augment_source_with_feature_roll(
        features,
        labels,
        config={"synthetic_per_class": 1, "max_shift": 1, "preserve_original": False, "random_state": 0},
    )

    assert result.labels.tolist() == [("cat", 1), ("dog", 2)]
    assert result.metadata["source_feature_roll_n_classes"] == 2
