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


def test_column_stats_preserves_large_finite_statistics():
    values = np.asarray([[1e100, -1e100], [2e100, -2e100]], dtype=float)

    result = column_stats(values)

    for summary in (result.mean, result.scale, result.minimum, result.maximum):
        assert summary.dtype == np.float64
        assert np.all(np.isfinite(summary))
    np.testing.assert_allclose(result.mean, [1.5e100, -1.5e100])
    np.testing.assert_allclose(result.scale, [np.sqrt(0.5) * 1e100, np.sqrt(0.5) * 1e100])
    np.testing.assert_allclose(result.minimum, [1e100, -2e100])
    np.testing.assert_allclose(result.maximum, [2e100, -1e100])


def test_column_stats_avoids_overflow_for_extreme_finite_statistics():
    values = np.asarray([[1e308, 1e308], [1e308, -1e308]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = column_stats(values)

    for summary in (result.mean, result.scale, result.minimum, result.maximum):
        assert np.all(np.isfinite(summary))
    np.testing.assert_allclose(result.mean, [1e308, 0.0])
    np.testing.assert_allclose(result.scale, [1e-12, np.sqrt(2.0) * 1e308], rtol=1e-15)
    np.testing.assert_allclose(result.minimum, [1e308, -1e308])
    np.testing.assert_allclose(result.maximum, [1e308, 1e308])


def test_column_stats_accepts_scalar_array_scale_floor():
    result = column_stats([[1.0], [1.0]], scale_floor=np.asarray(0.25))

    np.testing.assert_allclose(result.scale, np.array([0.25], dtype=np.float32))


@pytest.mark.parametrize("scale_floor", [True, np.bool_(True), np.asarray(True)])
def test_column_stats_rejects_boolean_scale_floor(scale_floor):
    with pytest.raises(ValueError, match="scale_floor"):
        column_stats([[1.0]], scale_floor=scale_floor)


@pytest.mark.parametrize("scale_floor", [0.0, -1.0, np.inf, np.nan])
def test_column_stats_rejects_non_positive_or_non_finite_scale_floor(scale_floor):
    with pytest.raises(ValueError, match="scale_floor"):
        column_stats([[1.0]], scale_floor=scale_floor)


@pytest.mark.parametrize("scale_floor", [[0.1], (0.1,), {"value": 0.1}, np.asarray([0.1])])
def test_column_stats_rejects_container_scale_floor(scale_floor):
    with pytest.raises(ValueError, match="scale_floor"):
        column_stats([[1.0]], scale_floor=scale_floor)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[1.0 + 2.0j, 3.0], [4.0, 5.0]], dtype=np.complex128),
        np.asarray([[1.0 + 2.0j, 3.0], [4.0, 5.0]], dtype=object),
        [[np.complex128(1.0 + 2.0j), 3.0], [4.0, 5.0]],
    ],
)
def test_column_stats_rejects_complex_values(values):
    with pytest.raises(ValueError, match="values must contain real-valued entries"):
        column_stats(values)
