from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.fold_flag import ABS_FEATURE_CATEGORY, absolute_value_features


def test_absolute_value_features_returns_abs_and_metadata() -> None:
    transformed, metadata = absolute_value_features([[-1.0, 2.0], [3.0, -4.0]])

    assert np.allclose(transformed, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert metadata["abs_feature_protocol_category"] == ABS_FEATURE_CATEGORY
    assert metadata["abs_feature_has_fitted_parameters"] is False
    assert metadata["abs_feature_uses_labels"] is False
    assert metadata["abs_feature_valid_for_strict_source_only"] is True


def test_absolute_value_features_accepts_one_pass_nested_rows() -> None:
    features = (iter(row) for row in [[-1.0, 2.0], [3.0, -4.0]])

    transformed, metadata = absolute_value_features(features)

    assert np.array_equal(transformed, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert metadata["abs_feature_n_rows"] == 2
    assert metadata["abs_feature_dim"] == 2


@pytest.mark.parametrize(
    "features",
    [
        [[True, False], [False, True]],
        np.asarray([[True, False], [False, True]]),
        np.asarray([[1.0, np.bool_(True)]], dtype=object),
        (iter(row) for row in [[1.0, False], [2.0, True]]),
    ],
)
def test_absolute_value_features_rejects_boolean_features(features) -> None:
    with pytest.raises(ValueError, match="boolean"):
        absolute_value_features(features)


@pytest.mark.parametrize(
    "features",
    [
        [[3.0 + 4.0j, -5.0 + 12.0j]],
        np.asarray([[3.0 + 4.0j, -5.0 + 12.0j]], dtype=np.complex128),
        np.asarray([[1.0, np.complex128(2.0 + 1.0j)]], dtype=object),
        (iter(row) for row in [[1.0 + 2.0j, 3.0], [4.0, 5.0]]),
    ],
)
def test_absolute_value_features_rejects_complex_features(features) -> None:
    with pytest.raises(ValueError, match="real-valued.*complex"):
        absolute_value_features(features)


def test_absolute_value_features_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        absolute_value_features([1.0, -2.0])


@pytest.mark.parametrize("features", [np.empty((0, 2)), np.empty((2, 0))])
def test_absolute_value_features_rejects_empty_matrix_dimensions(features: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        absolute_value_features(features)


def test_absolute_value_features_rejects_float32_overflow() -> None:
    too_large = np.nextafter(float(np.finfo(np.float32).max), np.inf)

    with np.errstate(over="raise"):
        with pytest.raises(ValueError, match="finite float32"):
            absolute_value_features([[too_large, -too_large]])


def test_absolute_value_features_accepts_float32_boundary() -> None:
    limit = np.finfo(np.float32).max

    transformed, _ = absolute_value_features([[limit, -limit]])

    assert np.array_equal(transformed, np.asarray([[limit, limit]], dtype=np.float32))
    assert np.all(np.isfinite(transformed))


def test_absolute_value_features_rejects_float32_underflow() -> None:
    smallest_nonzero = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    too_small = smallest_nonzero / 4.0
    assert too_small > 0.0

    with np.errstate(under="raise"):
        with pytest.raises(ValueError, match="nonzero float32"):
            absolute_value_features([[too_small, -too_small]])


def test_absolute_value_features_accepts_smallest_nonzero_float32() -> None:
    smallest_nonzero = np.nextafter(np.float32(0.0), np.float32(1.0))

    transformed, _ = absolute_value_features([[smallest_nonzero, -smallest_nonzero]])

    expected = np.asarray([[smallest_nonzero, smallest_nonzero]], dtype=np.float32)
    assert transformed.dtype == np.float32
    assert np.array_equal(transformed, expected)
    assert np.all(transformed != 0.0)
