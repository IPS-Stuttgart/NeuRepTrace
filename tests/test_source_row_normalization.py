from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_row_normalization import (
    ROW_NORMALIZATION_CATEGORY,
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


def test_aliases_and_validation() -> None:
    assert normalize_row_normalization_mode("euclidean") == "l2"
    assert normalize_row_normalization_mode("manhattan") == "l1"
    assert normalize_row_normalization_mode("max") == "max_abs"
    assert row_normalization_config(epsilon="1e-6").epsilon == 1e-6

    with pytest.raises(ValueError, match="row normalization mode"):
        normalize_row_normalization_mode("bad")

    with pytest.raises(ValueError, match="same feature width"):
        apply_row_normalization_pair(train_features=[[1.0, 2.0]], eval_features=[[1.0]])
