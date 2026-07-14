from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_range_selector import select_source_range_features


@pytest.mark.parametrize(
    "bad_ranges",
    [
        1.0,
        np.asarray(1.0),
        [[1.0, 2.0]],
        np.asarray([[1.0], [2.0]]),
        np.ones((1, 1, 1)),
    ],
)
def test_select_source_range_features_rejects_nonvector_ranges(bad_ranges) -> None:
    with pytest.raises(ValueError, match="ranges must be one-dimensional"):
        select_source_range_features(bad_ranges)


def test_select_source_range_features_keeps_generator_vector_support() -> None:
    ranges = (value for value in [0.0, 2.0, 4.0])

    selected = select_source_range_features(ranges, min_range=1.0)

    assert selected.tolist() == [1, 2]
