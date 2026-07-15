from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_balance import (
    compute_source_balance_weights,
    resample_source_rows_balanced,
    summarize_source_groups,
)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and np.isnan(value)


def test_source_balance_resamples_nan_labels_as_one_group() -> None:
    features = np.asarray([[0.0], [1.0], [10.0]], dtype=float)
    labels = np.asarray([float("nan"), np.float64(np.nan), "target"], dtype=object)

    result = resample_source_rows_balanced(
        features,
        labels,
        config={"strategy": "class", "target": "max", "random_state": 2},
    )

    assert result.features.shape == (4, 1)
    assert sum(_is_nan(label) for label in result.labels) == 2
    assert result.labels.tolist().count("target") == 2


@pytest.mark.parametrize("temporal_nat", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_source_balance_keeps_temporal_nat_distinct_from_none(temporal_nat: object) -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [10.0]], dtype=float)
    labels = np.empty(4, dtype=object)
    labels[:] = [None, None, None, temporal_nat]

    summary = summarize_source_groups(labels, strategy="class")

    assert len(summary.group_counts) == 2
    assert summary.group_counts[None] == 3
    nan_keys = [key for key in summary.group_counts if _is_nan(key)]
    assert len(nan_keys) == 1
    assert summary.group_counts[nan_keys[0]] == 1

    weights = compute_source_balance_weights(labels, config={"strategy": "class", "target": "max"})
    assert weights.sample_weights[-1] > weights.sample_weights[0]

    rows = resample_source_rows_balanced(
        features,
        labels,
        config={"strategy": "class", "target": "max", "random_state": 2},
    )
    assert rows.features.shape == (6, 1)
    assert sum(label is None for label in rows.labels) == 3
    assert sum(_is_nan(label) for label in rows.labels) == 3
