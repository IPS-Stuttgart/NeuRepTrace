from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_l2 import (
    ROW_L2_CATEGORY,
    RowL2Config,
    normalize_rows_l2,
    normalize_train_test_rows_l2,
    row_l2_config,
)


def test_normalize_rows_l2_returns_unit_l2_rows_and_original_norms() -> None:
    normalized, norms = normalize_rows_l2([[3.0, 4.0], [0.0, 0.0]], epsilon=1e-6)

    assert np.allclose(norms, np.asarray([5.0, 0.0]))
    assert np.allclose(normalized[0], np.asarray([3.0 / 5.0, 4.0 / 5.0]))
    assert np.allclose(normalized[1], np.asarray([0.0, 0.0]))


def test_normalize_train_test_rows_l2_metadata() -> None:
    result = normalize_train_test_rows_l2(
        train_features=[[3.0, 4.0], [0.0, 5.0]],
        test_features=[[-8.0, 6.0]],
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert np.allclose(np.linalg.norm(result.train_features, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(result.test_features, axis=1), 1.0)
    assert result.metadata["row_l2_protocol_category"] == ROW_L2_CATEGORY
    assert result.metadata["row_l2_has_fitted_parameters"] is False
    assert result.metadata["row_l2_uses_labels"] is False
    assert result.metadata["row_l2_valid_for_strict_source_only"] is True


def test_normalize_train_test_rows_l2_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        normalize_train_test_rows_l2(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_row_l2_config_validation() -> None:
    assert row_l2_config(epsilon="1e-5").epsilon == 1e-5

    with pytest.raises(ValueError, match="epsilon"):
        row_l2_config(epsilon=0.0)


def test_row_l2_config_normalizes_direct_dataclass_construction() -> None:
    cfg = RowL2Config(epsilon="1e-5")  # type: ignore[arg-type]

    assert cfg.epsilon == 1e-5


@pytest.mark.parametrize("epsilon", [True, np.bool_(True), np.asarray(True), np.asarray(True, dtype=object)])
def test_row_l2_rejects_boolean_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        row_l2_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_l2([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize("epsilon", [np.asarray([1e-5]), np.asarray([[1e-5]]), np.asarray([1e-5], dtype=object)])
def test_row_l2_rejects_array_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        row_l2_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        RowL2Config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_l2([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]
