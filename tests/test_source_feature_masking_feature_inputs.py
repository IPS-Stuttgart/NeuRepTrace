from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from neureptrace.decoding.source_masking import augment_source_with_feature_masking


def _row_generator(rows: list[list[Any]]):
    return ((value for value in row) for row in rows)


def test_source_feature_masking_accepts_generator_backed_feature_rows() -> None:
    result = augment_source_with_feature_masking(
        _row_generator([[1.0, 2.0], [3.0, 4.0]]),
        [0, 1],
    )

    assert result.features.dtype == np.float32
    assert np.allclose(result.features, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert result.labels.tolist() == [0, 1]


@pytest.mark.parametrize(
    "feature_factory",
    [
        lambda: np.asarray([[True, False], [False, True]], dtype=bool),
        lambda: np.asarray([[1.0, np.bool_(True)], [2.0, 3.0]], dtype=object),
        lambda: _row_generator([[1.0, True], [2.0, 3.0]]),
    ],
)
def test_source_feature_masking_rejects_boolean_feature_values(
    feature_factory: Callable[[], Any],
) -> None:
    with pytest.raises(ValueError, match="non-boolean"):
        augment_source_with_feature_masking(feature_factory(), [0, 1])
