from __future__ import annotations

import numpy as np

from neureptrace.decoding.array_stats import column_stats


def test_column_stats_saturates_unrepresentable_finite_scale() -> None:
    max_float = np.finfo(float).max
    values = np.asarray([[max_float], [-max_float]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = column_stats(values)

    np.testing.assert_allclose(result.mean, [0.0])
    np.testing.assert_allclose(result.minimum, [-max_float])
    np.testing.assert_allclose(result.maximum, [max_float])
    assert np.isfinite(result.scale[0])
    assert result.scale[0] == max_float
