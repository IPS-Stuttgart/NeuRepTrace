from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_center import (
    SOURCE_CENTER_CATEGORY,
    apply_source_center_transform,
    fit_source_center_map,
    fit_source_center_transform,
    normalize_center_mode,
)


def test_source_center_uses_source_mean_only() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=float)
    rows = np.asarray([[1.0, 19.0], [3.0, 21.0]], dtype=float)

    result = fit_source_center_transform(source_features=source, test_features=rows, config={"center": "mean"})

    assert result.center_map.center.tolist() == [2.0, 20.0]
    assert np.allclose(result.train_features.mean(axis=0), 0.0)
    assert result.test_features.tolist() == [[-1.0, -1.0], [1.0, 1.0]]
    assert result.metadata["source_center_protocol_category"] == SOURCE_CENTER_CATEGORY
    assert result.metadata["source_center_uses_test_features_for_fitting"] is False
    assert result.metadata["source_center_uses_labels"] is False
    assert result.metadata["source_center_valid_for_strict_source_only"] is True


def test_source_center_median_and_zero_modes() -> None:
    median_map = fit_source_center_map([[0.0], [1.0], [100.0]], config={"center": "median"})
    zero_map = fit_source_center_map([[0.0], [1.0], [100.0]], config={"center": "zero"})

    assert median_map.center.tolist() == [1.0]
    assert zero_map.center.tolist() == [0.0]
    assert apply_source_center_transform([[2.0]], median_map).ravel().tolist() == [1.0]


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[True, False], [False, True]], dtype=bool),
        np.asarray([[True, 1.0], [False, 0.0]], dtype=object),
        [[True, 0.0], [False, 1.0]],
        ((value for value in row) for row in [[True, 0.0], [False, 1.0]]),
    ],
)
def test_source_center_rejects_boolean_source_features(features) -> None:
    with pytest.raises(ValueError, match="source_features.*boolean flags"):
        fit_source_center_map(features)


def test_source_center_transform_rejects_boolean_test_features() -> None:
    with pytest.raises(ValueError, match="test_features.*boolean flags"):
        fit_source_center_transform(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            test_features=np.asarray([[True, False]], dtype=bool),
        )


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[1.0 + 2.0j, 3.0], [4.0, 5.0]], dtype=complex),
        np.asarray([[1.0 + 2.0j, 3.0], [4.0, 5.0]], dtype=object),
        [[1.0 + 2.0j, 3.0], [4.0, 5.0]],
        ((value for value in row) for row in [[1.0 + 2.0j, 3.0], [4.0, 5.0]]),
    ],
)
def test_source_center_rejects_complex_source_features(features) -> None:
    with pytest.raises(ValueError, match="source_features.*complex values"):
        fit_source_center_map(features)


def test_source_center_transform_rejects_complex_test_features() -> None:
    with pytest.raises(ValueError, match="test_features.*complex values"):
        fit_source_center_transform(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            test_features=np.asarray([[1.0 + 2.0j, 0.0]], dtype=complex),
        )


def test_source_center_accepts_one_pass_numeric_iterables() -> None:
    source = ((float(i + offset) for offset in (0, 2)) for i in range(3))
    test = ((float(i + offset) for offset in (1, 3)) for i in range(2))

    result = fit_source_center_transform(source_features=source, test_features=test, config={"center": "mean"})

    assert result.train_features.shape == (3, 2)
    assert result.test_features.shape == (2, 2)
    assert result.center_map.center.tolist() == [1.0, 3.0]
    np.testing.assert_allclose(result.train_features.mean(axis=0), np.asarray([0.0, 0.0]))
    np.testing.assert_allclose(result.test_features, np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))


def test_source_center_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_center_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_center_aliases_and_validation() -> None:
    assert normalize_center_mode("avg") == "mean"
    assert normalize_center_mode("med") == "median"
    assert normalize_center_mode("none") == "zero"

    with pytest.raises(ValueError, match="center mode"):
        normalize_center_mode("bad")
