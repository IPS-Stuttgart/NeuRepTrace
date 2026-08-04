from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.array_stats import column_stats


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[True, False], [False, True]], dtype=bool),
        np.asarray([[np.asarray(True), 1.0]], dtype=object),
        [[np.bool_(True), 1.0]],
    ],
)
def test_column_stats_rejects_boolean_values(values) -> None:
    with pytest.raises(ValueError, match="boolean"):
        column_stats(values)


def test_column_stats_preserves_numeric_zero_and_one() -> None:
    result = column_stats([[0.0, 1.0], [1.0, 0.0]])

    np.testing.assert_allclose(result.mean, [0.5, 0.5])
