from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_maxabs import (
    ROW_MAXABS_CATEGORY,
    normalize_rows_maxabs,
    normalize_train_score_rows_maxabs,
    row_maxabs_config,
)


def test_normalize_rows_maxabs_returns_scaled_rows_and_scales() -> None:
    normalized, scales = normalize_rows_maxabs([[2.0, -4.0], [0.0, 0.0]], epsilon=1e-6)

    assert np.allclose(scales, np.asarray([4.0, 0.0]))
    assert np.allclose(normalized[0], np.asarray([0.5, -1.0]))
    assert np.allclose(normalized[1], np.asarray([0.0, 0.0]))


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


def test_train_score_row_maxabs_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        normalize_train_score_rows_maxabs(train_features=[[1.0, 2.0]], score_features=[[1.0]])


def test_row_maxabs_config_validation() -> None:
    assert row_maxabs_config(epsilon="1e-5").epsilon == 1e-5
    with pytest.raises(ValueError, match="epsilon"):
        row_maxabs_config(epsilon=0.0)
