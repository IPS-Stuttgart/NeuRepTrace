from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_normalization import (
    ROW_NORMALIZATION_CATEGORY,
    RowNormalizationConfig,
    normalize_norm_mode,
    normalize_rows,
    normalize_source_and_test_rows,
    row_normalization_config,
    row_norms,
)


def test_l2_row_normalization_shapes_and_metadata() -> None:
    source = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=float)
    test = np.asarray([[5.0, 12.0]], dtype=float)

    result = normalize_source_and_test_rows(source_features=source, test_features=test)

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert np.allclose(row_norms(result.train_features), 1.0)
    assert np.allclose(row_norms(result.test_features), 1.0)
    assert result.train_norms.tolist() == [5.0, 2.0]
    assert result.metadata["row_normalization_protocol_category"] == ROW_NORMALIZATION_CATEGORY
    assert result.metadata["row_normalization_uses_cross_row_source_statistics"] is False
    assert result.metadata["row_normalization_uses_cross_row_test_statistics"] is False
    assert result.metadata["row_normalization_uses_test_labels"] is False
    assert result.metadata["row_normalization_valid_for_strict_source_only"] is True


def test_l1_and_max_norms() -> None:
    features = np.asarray([[1.0, -2.0, 3.0]], dtype=float)

    assert np.allclose(row_norms(features, norm="l1"), [6.0])
    assert np.allclose(row_norms(features, norm="max"), [3.0])
    l1_rows, _ = normalize_rows(features, config={"norm": "l1"})
    max_rows, _ = normalize_rows(features, config={"norm": "max"})
    assert np.allclose(np.sum(np.abs(l1_rows), axis=1), 1.0)
    assert np.allclose(np.max(np.abs(max_rows), axis=1), 1.0)


def test_l2_norms_avoid_intermediate_overflow() -> None:
    features = np.asarray([[1e200, -1e200]], dtype=float)

    normalized, norms = normalize_rows(features)

    assert np.all(np.isfinite(norms))
    np.testing.assert_allclose(norms, [np.sqrt(2.0) * 1e200])
    np.testing.assert_allclose(normalized, [[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
    np.testing.assert_allclose(row_norms(normalized), [1.0])


def test_l1_normalization_preserves_direction_when_norm_overflows() -> None:
    maximum = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise"):
        normalized, norms = normalize_rows([[maximum, -maximum]], config={"norm": "l1"})

    assert np.isinf(norms[0])
    np.testing.assert_allclose(normalized, [[0.5, -0.5]])


def test_l2_normalization_preserves_direction_when_norm_overflows() -> None:
    maximum = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise"):
        normalized, norms = normalize_rows([[maximum, maximum]], config={"norm": "l2"})

    assert np.isinf(norms[0])
    np.testing.assert_allclose(normalized, [[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]])


def test_high_level_outputs_preserve_precision_when_float32_would_corrupt() -> None:
    result = normalize_source_and_test_rows(
        source_features=[[1e100, 1e50]],
        test_features=[[1e80, 1e30]],
        config={"norm": "max"},
    )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert result.train_features[0, 1] > 0.0
    assert result.test_features[0, 1] > 0.0
    assert result.train_norms.dtype == np.float64
    assert result.test_norms.dtype == np.float64
    np.testing.assert_allclose(result.train_norms, [1e100])
    np.testing.assert_allclose(result.test_norms, [1e80])


def test_high_level_outputs_keep_float32_for_representable_values() -> None:
    result = normalize_source_and_test_rows(
        source_features=[[3.0, 4.0]],
        test_features=[[5.0, 12.0]],
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32


def test_center_rows_before_normalizing() -> None:
    features = np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    normalized, norms = normalize_rows(features, config={"center_rows": True})

    assert np.allclose(normalized.mean(axis=1), 0.0)
    assert np.allclose(norms, [np.sqrt(2.0)])


def test_zero_rows_remain_finite() -> None:
    normalized, norms = normalize_rows([[0.0, 0.0]], config={"epsilon": 1e-6})

    assert np.allclose(normalized, 0.0)
    assert np.allclose(norms, 0.0)


def test_dataclass_config_is_revalidated_when_used() -> None:
    cfg = RowNormalizationConfig(norm="euclidean", center_rows="false", epsilon="1e-6")  # type: ignore[arg-type]

    normalized, norms = normalize_rows([[1.0, 2.0, 3.0]], config=cfg)

    assert np.allclose(normalized, np.asarray([[1.0, 2.0, 3.0]]) / np.sqrt(14.0))
    assert np.allclose(norms, [np.sqrt(14.0)])


def test_row_normalization_accepts_scalar_numpy_bool_array_for_center_rows() -> None:
    config = row_normalization_config(center_rows=np.array(True))  # type: ignore[arg-type]

    normalized, norms = normalize_rows([[1.0, 3.0], [2.0, 6.0]], config=config)

    assert config.center_rows is True
    assert np.allclose(norms, [np.sqrt(2.0), np.sqrt(8.0)])
    assert np.allclose(normalized.mean(axis=1), [0.0, 0.0])


@pytest.mark.parametrize("value", [np.array([True]), np.array([[False]])])
def test_row_normalization_rejects_non_scalar_numpy_bool_arrays(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="center_rows"):
        row_normalization_config(center_rows=value)  # type: ignore[arg-type]


def test_boolean_epsilon_is_rejected() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        row_normalization_config(epsilon=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows([[0.0, 0.0]], config={"epsilon": True})


def test_aliases_and_validation() -> None:
    assert normalize_norm_mode("euclidean") == "l2"
    assert normalize_norm_mode("manhattan") == "l1"
    assert normalize_norm_mode("linf") == "max"

    with pytest.raises(ValueError, match="row norm mode"):
        normalize_norm_mode("bad")


def test_width_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        normalize_source_and_test_rows(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_test_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        normalize_source_and_test_rows(source_features=[[0.0], [1.0]], test_features=[[0.5]], test_labels=[0])  # type: ignore[call-arg]
