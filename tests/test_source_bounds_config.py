from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_bounds import (
    SourceBoundsConfig,
    SourceFeatureBounds,
    apply_source_feature_bounds,
    fit_source_feature_bound_values,
    fit_source_feature_bounds,
    source_bounds_config,
)


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True)])
def test_source_bounds_config_rejects_boolean_quantiles(value) -> None:
    with pytest.raises(ValueError, match="lower_quantile must be a numeric quantile"):
        source_bounds_config(lower_quantile=value, upper_quantile=0.9)


def test_source_bounds_config_normalizes_symmetric_bool_strings() -> None:
    assert source_bounds_config(symmetric="false").symmetric is False
    assert source_bounds_config(symmetric="off").symmetric is False
    assert source_bounds_config(symmetric="yes").symmetric is True
    assert source_bounds_config(symmetric=np.asarray(False)).symmetric is False

    with pytest.raises(ValueError, match="symmetric must be a boolean"):
        source_bounds_config(symmetric="maybe")


def test_source_bounds_dataclass_validates_and_normalizes_fields() -> None:
    cfg = SourceBoundsConfig(
        lower_quantile="0.2",
        upper_quantile=np.asarray(0.8),
        symmetric="on",
        center="zero-center",
    )

    assert cfg.lower_quantile == pytest.approx(0.2)
    assert cfg.upper_quantile == pytest.approx(0.8)
    assert cfg.symmetric is True
    assert cfg.center == "zero"

    with pytest.raises(ValueError, match="upper_quantile must be a numeric quantile"):
        SourceBoundsConfig(lower_quantile=0.1, upper_quantile=np.asarray(False))


def test_source_bounds_materializes_one_pass_feature_iterables() -> None:
    source_rows = ((value for value in row) for row in ((0.0, 2.0), (1.0, 3.0)))
    test_rows = ((value for value in row) for row in ((-1.0, 4.0),))

    result = fit_source_feature_bounds(
        source_features=source_rows,
        test_features=test_rows,
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    np.testing.assert_allclose(result.train_features, np.asarray([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(result.test_features, np.asarray([[0.0, 3.0]], dtype=np.float32))
    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
    assert result.metadata["source_feature_bounds_n_source_rows"] == 2


def test_source_bounds_preserves_values_outside_float32_range() -> None:
    values = np.asarray([[1e100, -1e100, 1e-100, -1e-100]], dtype=np.float64)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = fit_source_feature_bounds(
            source_features=values,
            test_features=values,
            config={"lower_quantile": 0.0, "upper_quantile": 1.0},
        )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    np.testing.assert_array_equal(result.train_features, values)
    np.testing.assert_array_equal(result.test_features, values)
    assert not np.any(result.train_changed_mask)
    assert not np.any(result.test_changed_mask)


def test_source_bounds_accepts_object_arrays_containing_generator_rows() -> None:
    source_rows = np.asarray([iter([0.0, 2.0]), iter([1.0, 3.0])], dtype=object)
    test_rows = np.asarray([iter([-1.0, 4.0])], dtype=object)

    result = fit_source_feature_bounds(
        source_features=source_rows,
        test_features=test_rows,
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    np.testing.assert_allclose(result.train_features, np.asarray([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(result.test_features, np.asarray([[0.0, 3.0]], dtype=np.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"source_features": [[True, 0.0]], "test_features": [[0.0, 1.0]]},
            "source_features must contain numeric feature values, not boolean flags",
        ),
        (
            {"source_features": [[0.0, 1.0]], "test_features": [[False, 1.0]]},
            "test_features must contain numeric feature values, not boolean flags",
        ),
    ],
)
def test_source_bounds_rejects_boolean_feature_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_source_feature_bounds(**kwargs)


def test_source_bounds_rejects_boolean_values_inside_one_pass_iterables() -> None:
    source_rows = ((value for value in row) for row in ((True, 0.0), (1.0, 2.0)))

    with pytest.raises(ValueError, match="source_features must contain numeric feature values, not boolean flags"):
        fit_source_feature_bound_values(source_rows)


def test_apply_source_bounds_rejects_boolean_feature_values() -> None:
    bounds = fit_source_feature_bound_values(
        [[0.0, 1.0], [1.0, 2.0]],
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    with pytest.raises(ValueError, match="features must contain numeric feature values, not boolean flags"):
        apply_source_feature_bounds([[False, 1.0]], bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        SourceFeatureBounds(lower=[0.0], upper=[np.inf], center=[0.0], n_source_rows=2),
        SourceFeatureBounds(lower=[[0.0, 1.0]], upper=[1.0, 2.0], center=[0.0, 0.0], n_source_rows=2),
        SourceFeatureBounds(lower=[False, 0.0], upper=[True, 1.0], center=[0.0, 0.0], n_source_rows=2),
        SourceFeatureBounds(lower=[1.0], upper=[0.0], center=[0.0], n_source_rows=2),
    ],
)
def test_apply_source_bounds_rejects_invalid_reused_bounds(bounds: SourceFeatureBounds) -> None:
    with pytest.raises(ValueError, match="source feature bounds"):
        apply_source_feature_bounds([[0.5]], bounds)


def test_apply_source_bounds_accepts_one_pass_reused_bound_vectors() -> None:
    bounds = SourceFeatureBounds(
        lower=(value for value in [0.0, 1.0]),
        upper=(value for value in [1.0, 3.0]),
        center=[0.0, 0.0],
        n_source_rows=2,
    )

    bounded, changed = apply_source_feature_bounds([[-1.0, 2.0]], bounds)

    np.testing.assert_allclose(bounded, np.asarray([[0.0, 2.0]]))
    np.testing.assert_array_equal(changed, np.asarray([[True, False]]))
