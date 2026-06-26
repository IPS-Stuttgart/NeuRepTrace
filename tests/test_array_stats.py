import numpy as np
import pytest

from neureptrace.decoding.array_stats import column_stats


def test_column_stats_computes_column_summaries():
    result = column_stats([[1.0, 4.0], [3.0, 4.0]], scale_floor=0.5)

    np.testing.assert_allclose(result.mean, np.array([2.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(result.scale, np.array([np.sqrt(2.0), 0.5], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(result.minimum, np.array([1.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(result.maximum, np.array([3.0, 4.0], dtype=np.float32))
    assert result.metadata == {"array_stats_rows": 2, "array_stats_columns": 2}


@pytest.mark.parametrize("scale_floor", [True, np.bool_(True)])
def test_column_stats_rejects_boolean_scale_floor(scale_floor):
    with pytest.raises(ValueError, match="scale_floor"):
        column_stats([[1.0]], scale_floor=scale_floor)


@pytest.mark.parametrize("scale_floor", [0.0, -1.0, np.inf, np.nan])
def test_column_stats_rejects_non_positive_or_non_finite_scale_floor(scale_floor):
    with pytest.raises(ValueError, match="scale_floor"):
        column_stats([[1.0]], scale_floor=scale_floor)
