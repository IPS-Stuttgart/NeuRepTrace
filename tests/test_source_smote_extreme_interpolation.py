from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_smote import interpolate_rows


def test_source_smote_midpoint_of_opposite_float64_extremes_is_finite() -> None:
    limit = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise"):
        row = interpolate_rows(
            [-limit, limit],
            [limit, -limit],
            0.5,
        )

    np.testing.assert_array_equal(row, np.zeros(2, dtype=np.float32))
