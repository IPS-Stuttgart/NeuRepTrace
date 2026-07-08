from __future__ import annotations

import pytest

from neureptrace.decoding.source_outlier import compute_source_outlier_weights


def _row_generator(row: list[float | bool]):
    return (value for value in row)


def test_source_outlier_accepts_one_pass_feature_iterables() -> None:
    feature_rows = (
        _row_generator(row)
        for row in [
            [0.0, 0.0],
            [0.1, 0.0],
            [4.9, 5.0],
            [5.0, 5.1],
        ]
    )

    result = compute_source_outlier_weights(
        feature_rows,
        ["a", "a", "b", "b"],
        config={"threshold_mode": "quantile", "weight_mode": "binary", "use_diagonal_scale": False},
    )

    assert result.sample_weights.shape == (4,)
    assert result.distances.shape == (4,)
    assert result.metadata["source_outlier_n_rows"] == 4


def test_source_outlier_rejects_boolean_inside_one_pass_feature_iterables() -> None:
    feature_rows = (
        _row_generator(row)
        for row in [
            [0.0, False],
            [0.1, 0.0],
            [4.9, 5.0],
            [5.0, 5.1],
        ]
    )

    with pytest.raises(ValueError, match="source_features.*non-boolean"):
        compute_source_outlier_weights(feature_rows, ["a", "a", "b", "b"])
