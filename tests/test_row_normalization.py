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


def test_center_rows_before_normalizing() -> None:
    features = np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    normalized, norms = normalize_rows(features, config={"center_rows": True})

    assert np.allclose(normalized.mean(axis=1), 0.0)
    assert np.allclose(norms, [np.sqrt(2.0)])


def test_zero_rows_remain_finite() -> None:
    normalized, norms = normalize_rows([[0.0, 0.0]], config={"epsilon": 1e-6})

    assert np.allclose(normalized, 0.0)
    assert np.allclose(norms, 0.0)


def test_direct_config_normalizes_aliases_and_boolean_strings() -> None:
    cfg = RowNormalizationConfig(norm="euclidean", center_rows="false", epsilon="1e-6")  # type: ignore[arg-type]

    assert cfg.norm == "l2"
    assert cfg.center_rows is False
    assert cfg.epsilon == pytest.approx(1e-6)

    normalized, norms = normalize_rows([[1.0, 2.0, 3.0]], config=cfg)

    assert np.allclose(normalized, np.asarray([[1.0, 2.0, 3.0]]) / np.sqrt(14.0))
    assert np.allclose(norms, [np.sqrt(14.0)])


def test_boolean_epsilon_is_rejected() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        row_normalization_config(epsilon=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon"):
        RowNormalizationConfig(epsilon=True)  # type: ignore[arg-type]
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
