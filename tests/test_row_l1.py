from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_l1 import (
    ROW_L1_CATEGORY,
    RowL1Config,
    normalize_rows_l1,
    normalize_train_test_rows_l1,
    row_l1_config,
)


def test_normalize_rows_l1_returns_unit_l1_rows_and_original_norms() -> None:
    normalized, norms = normalize_rows_l1([[1.0, -2.0], [0.0, 0.0]], epsilon=1e-6)

    assert np.allclose(norms, np.asarray([3.0, 0.0]))
    assert np.allclose(normalized[0], np.asarray([1.0 / 3.0, -2.0 / 3.0]))
    assert np.allclose(normalized[1], np.asarray([0.0, 0.0]))


def test_normalize_rows_l1_avoids_overflow_for_large_finite_rows() -> None:
    features = np.asarray([[1e308, -1e308], [0.0, 0.0]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        normalized, norms = normalize_rows_l1(features)

    assert np.isinf(norms[0])
    assert norms[1] == 0.0
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(normalized[0], [0.5, -0.5])
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])


def test_normalize_train_test_rows_l1_metadata() -> None:
    result = normalize_train_test_rows_l1(
        train_features=[[1.0, -2.0], [3.0, 0.0]],
        test_features=[[-4.0, 6.0]],
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert np.allclose(np.sum(np.abs(result.train_features), axis=1), 1.0)
    assert np.allclose(np.sum(np.abs(result.test_features), axis=1), 1.0)
    assert result.metadata["row_l1_protocol_category"] == ROW_L1_CATEGORY
    assert result.metadata["row_l1_has_fitted_parameters"] is False
    assert result.metadata["row_l1_uses_labels"] is False
    assert result.metadata["row_l1_valid_for_strict_source_only"] is True


def test_normalize_train_test_rows_l1_preserves_large_finite_norms() -> None:
    result = normalize_train_test_rows_l1(
        train_features=[[1e100, -5e99]],
        test_features=[[1e80, -1e80]],
    )

    assert result.train_norms.dtype == np.float64
    assert result.test_norms.dtype == np.float64
    assert np.all(np.isfinite(result.train_norms))
    assert np.all(np.isfinite(result.test_norms))
    np.testing.assert_allclose(result.train_norms, [1.5e100])
    np.testing.assert_allclose(result.test_norms, [2e80])
    np.testing.assert_allclose(result.train_features, [[2.0 / 3.0, -1.0 / 3.0]])
    np.testing.assert_allclose(result.test_features, [[0.5, -0.5]])


def test_normalize_train_test_rows_l1_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        normalize_train_test_rows_l1(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_normalize_train_test_rows_l1_rejects_unknown_config_options() -> None:
    with pytest.raises(ValueError, match="Unknown row L1 config option"):
        normalize_train_test_rows_l1(
            train_features=[[1.0, 2.0]],
            test_features=[[1.0, 2.0]],
            config={"epsilon": 1e-6, "epislon": 1e-5},
        )


def test_normalize_train_test_rows_l1_rejects_non_mapping_config() -> None:
    with pytest.raises(ValueError, match="Row L1 config must be a mapping"):
        normalize_train_test_rows_l1(
            train_features=[[1.0, 2.0]],
            test_features=[[1.0, 2.0]],
            config=object(),  # type: ignore[arg-type]
        )


def test_row_l1_config_validation() -> None:
    assert row_l1_config(epsilon="1e-5").epsilon == 1e-5

    with pytest.raises(ValueError, match="epsilon"):
        row_l1_config(epsilon=0.0)


def test_row_l1_config_direct_construction_normalizes_epsilon() -> None:
    assert RowL1Config(epsilon="1e-5").epsilon == 1e-5  # type: ignore[arg-type]


@pytest.mark.parametrize("epsilon", [True, np.bool_(True), np.asarray(True), np.asarray(True, dtype=object)])
def test_row_l1_rejects_boolean_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowL1Config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_l1_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_l1([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize("epsilon", [np.asarray([1e-5]), np.asarray([1e-5], dtype=object)])
def test_row_l1_rejects_array_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowL1Config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_l1_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_l1([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]
