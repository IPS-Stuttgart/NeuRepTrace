from __future__ import annotations

import numpy as np
import pytest

from neureptrace.probability_stacking import combine_probability_cube


@pytest.mark.parametrize("pooling", ["linear", "log"])
def test_combine_probability_cube_handles_overflowing_finite_weight_sum(pooling: str) -> None:
    probability_cube = np.asarray(
        [
            [[0.9, 0.1], [0.3, 0.7]],
            [[0.1, 0.9], [0.7, 0.3]],
        ],
        dtype=float,
    )

    combined = combine_probability_cube(
        probability_cube,
        weights=[1e308, 1e308],
        pooling=pooling,
    )
    reference = combine_probability_cube(
        probability_cube,
        weights=[1.0, 1.0],
        pooling=pooling,
    )

    assert np.all(np.isfinite(combined))
    np.testing.assert_allclose(combined, reference)
    np.testing.assert_allclose(combined.sum(axis=1), np.ones(combined.shape[0]))
