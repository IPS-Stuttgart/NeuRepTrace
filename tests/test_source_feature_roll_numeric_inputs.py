from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_roll import augment_source_with_feature_roll, roll_feature_row


def _nested_rows(rows):
    return ((value for value in row) for row in rows)


def test_source_roll_materializes_nested_generator_features() -> None:
    result = augment_source_with_feature_roll(
        _nested_rows(([1.0, 2.0], [3.0, 4.0])),
        np.asarray(["left", "right"]),
        config={"synthetic_per_class": 0},
    )

    np.testing.assert_array_equal(result.features, np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_roll_feature_row_materializes_generator_values() -> None:
    result = roll_feature_row((value for value in [1.0, 2.0, 3.0]), shift=1)

    np.testing.assert_array_equal(result, np.asarray([3.0, 1.0, 2.0]))


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[True, False]]),
        [[1.0, np.bool_(False)]],
        _nested_rows(([1.0, True],)),
    ],
)
def test_source_roll_rejects_boolean_features(features) -> None:
    with pytest.raises(ValueError, match="booleans"):
        augment_source_with_feature_roll(
            features,
            np.asarray(["class"]),
            config={"synthetic_per_class": 0},
        )


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[1.0 + 2.0j, 3.0]]),
        [[1.0, np.complex128(2.0 + 3.0j)]],
        _nested_rows(([1.0, 2.0 + 1.0j],)),
    ],
)
def test_source_roll_rejects_complex_features(features) -> None:
    with pytest.raises(ValueError, match="complex"):
        augment_source_with_feature_roll(
            features,
            np.asarray(["class"]),
            config={"synthetic_per_class": 0},
        )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([True, False], "booleans"),
        ([1.0, 2.0 + 3.0j], "complex"),
        (np.asarray([1.0, np.bool_(True)], dtype=object), "booleans"),
        (np.asarray([1.0, 2.0 + 3.0j], dtype=object), "complex"),
    ],
)
def test_roll_feature_row_rejects_non_real_numeric_values(row, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        roll_feature_row(row, shift=1)
