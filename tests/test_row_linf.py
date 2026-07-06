from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_linf import (
    ROW_LINF_CATEGORY,
    normalize_rows_linf,
    normalize_train_test_rows_linf,
    row_linf_config,
)


def test_normalize_rows_linf_returns_unit_max_rows_and_norms() -> None:
    normalized, norms = normalize_rows_linf([[3.0, -6.0], [0.0, 0.0]], epsilon=1e-6)

    assert np.allclose(norms, np.asarray([6.0, 0.0]))
    assert np.allclose(normalized[0], np.asarray([0.5, -1.0]))
    assert np.allclose(normalized[1], np.asarray([0.0, 0.0]))


def test_normalize_train_test_rows_linf_metadata() -> None:
    result = normalize_train_test_rows_linf(
        train_features=[[3.0, -6.0], [5.0, 10.0]],
        test_features=[[8.0, -4.0]],
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert np.allclose(np.max(np.abs(result.train_features), axis=1), 1.0)
    assert np.allclose(np.max(np.abs(result.test_features), axis=1), 1.0)
    assert result.metadata["row_linf_protocol_category"] == ROW_LINF_CATEGORY
    assert result.metadata["row_linf_has_fitted_parameters"] is False
    assert result.metadata["row_linf_uses_labels"] is False
    assert result.metadata["row_linf_valid_for_strict_source_only"] is True


def test_normalize_train_test_rows_linf_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        normalize_train_test_rows_linf(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_row_linf_config_validation() -> None:
    assert row_linf_config(epsilon="1e-5").epsilon == 1e-5

    with pytest.raises(ValueError, match="epsilon"):
        row_linf_config(epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon"):
        row_linf_config(epsilon=True)  # type: ignore[arg-type]
