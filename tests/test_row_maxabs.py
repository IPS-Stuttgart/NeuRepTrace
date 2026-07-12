from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.row_maxabs import (
    ROW_MAXABS_CATEGORY,
    RowMaxAbsConfig,
    normalize_rows_maxabs,
    normalize_train_score_rows_maxabs,
    row_maxabs_config,
)


def _nested_generator_rows(rows: list[list[object]]) -> object:
    return ((value for value in row) for row in rows)


def test_normalize_rows_maxabs_returns_scaled_rows_and_scales() -> None:
    normalized, scales = normalize_rows_maxabs([[2.0, -4.0], [0.0, 0.0]], epsilon=1e-6)

    assert np.allclose(scales, np.asarray([4.0, 0.0]))
    assert np.allclose(normalized[0], np.asarray([0.5, -1.0]))
    assert np.allclose(normalized[1], np.asarray([0.0, 0.0]))


def test_normalize_rows_maxabs_materializes_nested_generators() -> None:
    normalized, scales = normalize_rows_maxabs(_nested_generator_rows([[2.0, -4.0], [1.0, 2.0]]))

    assert np.allclose(scales, np.asarray([4.0, 2.0]))
    assert np.allclose(normalized, np.asarray([[0.5, -1.0], [0.5, 1.0]]))


def test_train_score_row_maxabs_metadata() -> None:
    result = normalize_train_score_rows_maxabs(
        train_features=[[2.0, -4.0], [1.0, 2.0]],
        score_features=[[5.0, -10.0]],
    )

    assert np.allclose(result.train_features[0], np.asarray([0.5, -1.0], dtype=np.float32))
    assert np.allclose(result.score_features[0], np.asarray([0.5, -1.0], dtype=np.float32))
    assert result.metadata["row_maxabs_protocol_category"] == ROW_MAXABS_CATEGORY
    assert result.metadata["row_maxabs_has_fitted_parameters"] is False
    assert result.metadata["row_maxabs_uses_labels"] is False
    assert result.metadata["row_maxabs_valid_for_strict_source_only"] is True


def test_train_score_row_maxabs_materializes_nested_generators() -> None:
    result = normalize_train_score_rows_maxabs(
        train_features=_nested_generator_rows([[2.0, -4.0], [1.0, 2.0]]),
        score_features=_nested_generator_rows([[5.0, -10.0]]),
    )

    np.testing.assert_allclose(result.train_features, [[0.5, -1.0], [0.5, 1.0]])
    np.testing.assert_allclose(result.score_features, [[0.5, -1.0]])


def test_train_score_row_maxabs_preserves_large_finite_scales() -> None:
    result = normalize_train_score_rows_maxabs(
        train_features=[[1e100, -5e99]],
        score_features=[[1e80, -1e80]],
    )

    assert result.train_scales.dtype == np.float64
    assert result.score_scales.dtype == np.float64
    assert np.all(np.isfinite(result.train_scales))
    assert np.all(np.isfinite(result.score_scales))
    np.testing.assert_allclose(result.train_scales, [1e100])
    np.testing.assert_allclose(result.score_scales, [1e80])
    np.testing.assert_allclose(result.train_features, [[1.0, -0.5]])
    np.testing.assert_allclose(result.score_features, [[1.0, -1.0]])


def test_train_score_row_maxabs_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        normalize_train_score_rows_maxabs(train_features=[[1.0, 2.0]], score_features=[[1.0]])


@pytest.mark.parametrize(
    "features_factory",
    [
        lambda: [[True, 0.0]],
        lambda: np.asarray([[True, False]]),
        lambda: np.asarray([[1.0, True]], dtype=object),
        lambda: _nested_generator_rows([[1.0, True]]),
    ],
)
def test_row_maxabs_rejects_boolean_feature_values(features_factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="non-boolean feature values"):
        normalize_rows_maxabs(features_factory())  # type: ignore[arg-type]


def test_row_maxabs_rejects_malformed_numeric_features_stably() -> None:
    with pytest.raises(ValueError, match="numeric, non-boolean feature values"):
        normalize_rows_maxabs(object())  # type: ignore[arg-type]


def test_row_maxabs_config_validation() -> None:
    assert row_maxabs_config(epsilon="1e-5").epsilon == 1e-5
    with pytest.raises(ValueError, match="epsilon"):
        row_maxabs_config(epsilon=0.0)


def test_row_maxabs_config_direct_construction_normalizes_epsilon() -> None:
    assert RowMaxAbsConfig(epsilon="1e-5").epsilon == 1e-5  # type: ignore[arg-type]


@pytest.mark.parametrize("epsilon", [True, np.bool_(True), np.asarray(True), np.asarray(True, dtype=object)])
def test_row_maxabs_rejects_boolean_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowMaxAbsConfig(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_maxabs_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_maxabs([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize("epsilon", [np.asarray([1e-5]), np.asarray([1e-5], dtype=object)])
def test_row_maxabs_rejects_array_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowMaxAbsConfig(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_maxabs_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_maxabs([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]
