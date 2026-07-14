from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_asinh import apply_source_asinh_transform, fit_source_asinh_map, fit_source_asinh_transform, source_asinh_config


def _nested_rows(rows: list[list[float]]):
    return (iter(row) for row in rows)


def test_fit_source_asinh_map_accepts_nested_rows() -> None:
    transform_map = fit_source_asinh_map(
        _nested_rows([[1.0, 2.0], [3.0, 8.0]]),
        config=source_asinh_config(scale_mode="unit"),
    )

    np.testing.assert_allclose(transform_map.scale, [1.0, 1.0])
    assert transform_map.n_source_rows == 2


def test_apply_source_asinh_transform_accepts_nested_rows() -> None:
    transform_map = fit_source_asinh_map([[1.0, 2.0], [3.0, 8.0]], config=source_asinh_config(scale_mode="unit"))

    transformed = apply_source_asinh_transform(_nested_rows([[0.0, 1.0], [2.0, 3.0]]), transform_map)

    np.testing.assert_allclose(transformed, np.arcsinh([[0.0, 1.0], [2.0, 3.0]]))


def test_fit_source_asinh_transform_accepts_nested_rows() -> None:
    result = fit_source_asinh_transform(
        source_features=_nested_rows([[1.0, 2.0], [3.0, 8.0]]),
        test_features=_nested_rows([[0.0, 1.0]]),
        config=source_asinh_config(scale_mode="unit"),
    )

    np.testing.assert_allclose(result.train_features, np.arcsinh([[1.0, 2.0], [3.0, 8.0]]))
    np.testing.assert_allclose(result.test_features, np.arcsinh([[0.0, 1.0]]))
    assert result.metadata["source_asinh_n_source_rows"] == 2
    assert result.metadata["source_asinh_n_test_rows"] == 1


def test_source_asinh_rejects_boolean_feature_values() -> None:
    with pytest.raises(ValueError, match="source_features.*boolean"):
        fit_source_asinh_map([[True, False]], config=source_asinh_config(scale_mode="unit"))

    transform_map = fit_source_asinh_map([[1.0, 2.0]], config=source_asinh_config(scale_mode="unit"))
    with pytest.raises(ValueError, match="features.*boolean"):
        apply_source_asinh_transform([[np.bool_(False), np.bool_(True)]], transform_map)

    with pytest.raises(ValueError, match="test_features.*boolean"):
        fit_source_asinh_transform(
            source_features=[[1.0, 2.0]],
            test_features=_nested_rows([[False, True]]),
            config=source_asinh_config(scale_mode="unit"),
        )


def test_source_asinh_rejects_boolean_numeric_config_values() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        source_asinh_config(multiplier=True)

    with pytest.raises(ValueError, match="epsilon"):
        source_asinh_config(epsilon=np.bool_(True))
