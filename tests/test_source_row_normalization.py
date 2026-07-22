from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_row_normalization import (
    ROW_NORMALIZATION_CATEGORY,
    RowNormalizationConfig,
    apply_row_normalization_pair,
    normalize_feature_rows,
    normalize_row_normalization_mode,
    row_normalization_config,
)


def test_l2_row_normalization_pair_metadata() -> None:
    train = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=float)
    eval_rows = np.asarray([[5.0, 12.0]], dtype=float)

    result = apply_row_normalization_pair(train_features=train, eval_features=eval_rows, config={"mode": "l2"})

    assert np.allclose(result.train_features[0], np.asarray([0.6, 0.8]))
    assert np.allclose(result.train_features[1], np.asarray([0.0, 0.0]))
    assert np.allclose(result.eval_features[0], np.asarray([5.0 / 13.0, 12.0 / 13.0]))
    assert result.train_norms.tolist() == [5.0, 0.0]
    assert result.metadata["row_normalization_protocol_category"] == ROW_NORMALIZATION_CATEGORY
    assert result.metadata["row_normalization_fits_eval_statistics"] is False
    assert result.metadata["row_normalization_uses_eval_labels"] is False
    assert result.metadata["row_normalization_valid_for_strict_source_only"] is True


def test_l1_and_max_abs_modes() -> None:
    matrix = np.asarray([[1.0, -3.0, 2.0]], dtype=float)

    l1, l1_norm = normalize_feature_rows(matrix, mode="l1")
    max_abs, max_norm = normalize_feature_rows(matrix, mode="max_abs")

    assert np.allclose(l1, np.asarray([[1.0 / 6.0, -3.0 / 6.0, 2.0 / 6.0]]))
    assert np.allclose(l1_norm, np.asarray([6.0]))
    assert np.allclose(max_abs, np.asarray([[1.0 / 3.0, -1.0, 2.0 / 3.0]]))
    assert np.allclose(max_norm, np.asarray([3.0]))


def test_none_mode_returns_identity_rows() -> None:
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    normalized, norms = normalize_feature_rows(matrix, mode="none")

    assert np.allclose(normalized, matrix)
    assert np.allclose(norms, np.ones(2))


def test_l2_mode_preserves_large_finite_row_direction() -> None:
    matrix = np.asarray([[1e200, -1e200]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        normalized, norms = normalize_feature_rows(matrix, mode="l2")

    expected = np.asarray([[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
    assert np.allclose(normalized, expected)
    assert np.isfinite(norms[0])
    assert np.isclose(norms[0], np.sqrt(2.0) * 1e200)


def test_l1_mode_preserves_direction_when_raw_norm_is_unrepresentable() -> None:
    maximum = np.finfo(float).max
    matrix = np.asarray([[maximum, -maximum]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        normalized, norms = normalize_feature_rows(matrix, mode="l1")

    assert np.allclose(normalized, np.asarray([[0.5, -0.5]]))
    assert np.isinf(norms[0])


def test_none_mode_does_not_corrupt_finite_float64_values() -> None:
    matrix = np.asarray([[1e40, 1e-100]], dtype=float)

    normalized, _ = normalize_feature_rows(matrix, mode="none")

    assert normalized.dtype == np.float64
    assert np.array_equal(normalized, matrix)


def test_pair_preserves_large_representable_norm_metadata() -> None:
    train = np.asarray([[1e200, -1e200]], dtype=float)
    eval_rows = np.asarray([[2e200, 0.0]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = apply_row_normalization_pair(train_features=train, eval_features=eval_rows, config={"mode": "l2"})

    assert result.train_norms.dtype == np.float64
    assert result.eval_norms.dtype == np.float64
    assert np.isfinite(result.train_norms[0])
    assert np.isfinite(result.eval_norms[0])
    assert np.allclose(result.train_features, np.asarray([[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]]))
    assert np.allclose(result.eval_features, np.asarray([[1.0, 0.0]]))


def test_direct_config_is_canonicalized_for_metadata() -> None:
    result = apply_row_normalization_pair(
        train_features=[[1.0, 2.0]],
        eval_features=[[3.0, 4.0]],
        config=RowNormalizationConfig(mode="off", epsilon="1e-6"),
    )

    assert result.metadata["row_normalization"] is False
    assert result.metadata["row_normalization_mode"] == "none"
    assert result.metadata["row_normalization_epsilon"] == 1e-6


def test_aliases_and_validation() -> None:
    assert normalize_row_normalization_mode("euclidean") == "l2"
    assert normalize_row_normalization_mode("manhattan") == "l1"
    assert normalize_row_normalization_mode("max") == "max_abs"
    assert row_normalization_config(epsilon="1e-6").epsilon == 1e-6

    with pytest.raises(ValueError, match="row normalization mode"):
        normalize_row_normalization_mode("bad")

    with pytest.raises(ValueError, match="same feature width"):
        apply_row_normalization_pair(train_features=[[1.0, 2.0]], eval_features=[[1.0]])
