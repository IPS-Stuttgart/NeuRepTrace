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


def test_normalize_rows_l2_preserves_direction_when_norm_exceeds_float64() -> None:
    maximum = np.finfo(float).max
    features = np.asarray([[maximum, maximum], [maximum, -maximum]])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized, norms = normalize_rows_l2(features)

    expected = np.asarray([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    assert np.isinf(norms).all()
    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


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


def test_normalize_train_test_rows_l2_preserves_unrepresentable_norm_rows() -> None:
    maximum = np.finfo(float).max

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = normalize_train_test_rows_l2(
            train_features=[[maximum, maximum]],
            test_features=[[maximum, -maximum]],
        )

    expected = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32) / np.float32(np.sqrt(2.0))
    np.testing.assert_allclose(result.train_features, expected[:1])
    np.testing.assert_allclose(result.test_features, expected[1:])
    assert np.isinf(result.train_norms).all()
    assert np.isinf(result.test_norms).all()


def test_normalize_train_test_rows_l2_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        normalize_train_test_rows_l2(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_row_l2_accepts_one_pass_feature_iterables() -> None:
    train = ((value for value in row) for row in ([3.0, 4.0], [0.0, 5.0]))
    test = ((value for value in row) for row in ([-8.0, 6.0],))

    result = normalize_train_test_rows_l2(train_features=train, test_features=test)
    normalized, norms = normalize_rows_l2(((value for value in row) for row in ([6.0, 8.0],)))

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    np.testing.assert_allclose(normalized, np.asarray([[0.6, 0.8]]))
    np.testing.assert_allclose(norms, np.asarray([10.0]))


@pytest.mark.parametrize(
    "features",
    [
        [[True, False], [False, True]],
        np.asarray([[True, False]], dtype=bool),
        np.asarray([[True, 0.0]], dtype=object),
        ((value for value in row) for row in ([True, 0.0],)),
    ],
)
def test_normalize_rows_l2_rejects_boolean_features(features: object) -> None:
    with pytest.raises(ValueError, match="features.*boolean flags"):
        normalize_rows_l2(features)  # type: ignore[arg-type]


def test_normalize_train_test_rows_l2_rejects_boolean_features() -> None:
    with pytest.raises(ValueError, match="train_features.*boolean flags"):
        normalize_train_test_rows_l2(train_features=[[True, False]], test_features=[[1.0, 2.0]])

    with pytest.raises(ValueError, match="test_features.*boolean flags"):
        normalize_train_test_rows_l2(train_features=[[1.0, 2.0]], test_features=[[False, True]])


def test_row_l2_mapping_config_validation() -> None:
    with pytest.raises(ValueError, match="Unknown row L2 config option"):
        normalize_train_test_rows_l2(
            train_features=[[1.0, 2.0]],
            test_features=[[1.0, 2.0]],
            config={"epislon": 1e-5},
        )

    with pytest.raises(ValueError, match="Row L2 config must be a mapping"):
        normalize_train_test_rows_l2(
            train_features=[[1.0, 2.0]],
            test_features=[[1.0, 2.0]],
            config=object(),  # type: ignore[arg-type]
        )


def test_row_l2_config_validation() -> None:
    assert row_l2_config(epsilon="1e-5").epsilon == 1e-5

    with pytest.raises(ValueError, match="epsilon"):
        row_l2_config(epsilon=0.0)


def test_row_l2_config_direct_construction_normalizes_epsilon() -> None:
    assert RowL2Config(epsilon="1e-5").epsilon == 1e-5  # type: ignore[arg-type]


@pytest.mark.parametrize("epsilon", [True, np.bool_(True), np.asarray(True), np.asarray(True, dtype=object)])
def test_row_l2_rejects_boolean_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowL2Config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_l2_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_l2([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize("epsilon", [np.asarray([1e-5]), np.asarray([1e-5], dtype=object)])
def test_row_l2_rejects_array_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowL2Config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_l2_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_l2([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]
