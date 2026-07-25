from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clipping import (
    SOURCE_CLIPPING_CATEGORY,
    apply_feature_clipping,
    fit_source_feature_clipping,
    source_feature_clipping_bounds,
    source_feature_clipping_config,
)


def test_source_feature_clipping_uses_source_bounds_only() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]], dtype=float)
    test = np.asarray([[-5.0, 100.0], [1.5, 11.5]], dtype=float)

    result = fit_source_feature_clipping(
        source_features=source,
        test_features=test,
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert np.allclose(result.lower_bounds, np.asarray([0.0, 10.0]))
    assert np.allclose(result.upper_bounds, np.asarray([2.0, 12.0]))
    assert np.allclose(result.test_features, np.asarray([[0.0, 12.0], [1.5, 11.5]]))
    assert result.metadata["source_feature_clipping_protocol_category"] == SOURCE_CLIPPING_CATEGORY
    assert result.metadata["source_feature_clipping_uses_test_features_for_fitting"] is False
    assert result.metadata["source_feature_clipping_uses_test_labels"] is False
    assert result.metadata["source_feature_clipping_valid_for_strict_source_only"] is True


def test_clipping_bounds_follow_quantiles() -> None:
    source = np.asarray([[0.0], [10.0], [20.0], [30.0], [40.0]], dtype=float)

    lower, upper = source_feature_clipping_bounds(source, lower_quantile=0.25, upper_quantile=0.75)

    assert np.allclose(lower, np.asarray([10.0]))
    assert np.allclose(upper, np.asarray([30.0]))


def test_apply_feature_clipping_can_modify_in_place() -> None:
    features = np.asarray([[-1.0, 5.0], [2.0, 9.0]], dtype=float)

    clipped = apply_feature_clipping(
        features,
        lower_bounds=[0.0, 6.0],
        upper_bounds=[1.0, 8.0],
        copy=False,
    )

    assert clipped is features
    assert np.allclose(features, np.asarray([[0.0, 6.0], [1.0, 8.0]]))


def test_source_feature_clipping_normalizes_copy_booleans() -> None:
    assert source_feature_clipping_config(copy="false").copy is False
    assert source_feature_clipping_config(copy=" off ").copy is False
    assert source_feature_clipping_config(copy="yes").copy is True
    assert source_feature_clipping_config(copy=1).copy is True

    with pytest.raises(ValueError, match="copy must be a boolean"):
        source_feature_clipping_config(copy="maybe")


def test_apply_feature_clipping_normalizes_copy_strings() -> None:
    features = np.asarray([[-1.0, 5.0], [2.0, 9.0]], dtype=float)

    clipped = apply_feature_clipping(
        features,
        lower_bounds=[0.0, 6.0],
        upper_bounds=[1.0, 8.0],
        copy="false",
    )

    assert clipped is features
    assert np.allclose(features, np.asarray([[0.0, 6.0], [1.0, 8.0]]))


@pytest.mark.parametrize(
    "name,lower_bounds,upper_bounds",
    [
        ("lower_bounds", [np.nan], [1.0]),
        ("upper_bounds", [0.0], [np.nan]),
    ],
)
def test_apply_feature_clipping_rejects_nan_bounds(
    name: str,
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must not contain NaN values"):
        apply_feature_clipping(
            [[0.5]],
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )


def test_source_feature_clipping_preserves_values_outside_float32_range() -> None:
    source = np.asarray([[1.0e-50], [1.0e40]], dtype=float)
    test = np.asarray([[1.0e-60], [1.0e50]], dtype=float)

    with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
        result = fit_source_feature_clipping(
            source_features=source,
            test_features=test,
            config={"lower_quantile": 0.0, "upper_quantile": 1.0},
        )

    for values in (
        result.train_features,
        result.test_features,
        result.lower_bounds,
        result.upper_bounds,
    ):
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        assert np.count_nonzero(values) == values.size

    np.testing.assert_array_equal(result.train_features, source)
    np.testing.assert_array_equal(result.test_features, source)
    np.testing.assert_array_equal(result.lower_bounds, [1.0e-50])
    np.testing.assert_array_equal(result.upper_bounds, [1.0e40])


def test_source_feature_clipping_keeps_float32_for_representable_values() -> None:
    result = fit_source_feature_clipping(
        source_features=[[0.0], [1.0]],
        test_features=[[-1.0], [2.0]],
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
    assert result.lower_bounds.dtype == np.float32
    assert result.upper_bounds.dtype == np.float32


def test_source_feature_clipping_rejects_invalid_quantiles() -> None:
    with pytest.raises(ValueError, match="lower_quantile"):
        source_feature_clipping_config(lower_quantile=0.9, upper_quantile=0.1)

    with pytest.raises(ValueError, match="upper_quantile"):
        source_feature_clipping_config(upper_quantile=1.5)


def test_source_feature_clipping_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_feature_clipping(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_feature_clipping_has_no_target_label_api() -> None:
    with pytest.raises(TypeError):
        fit_source_feature_clipping(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            target_labels=[0],  # type: ignore[call-arg]
        )
