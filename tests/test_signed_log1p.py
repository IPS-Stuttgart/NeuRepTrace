from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.signed_log1p import (
    SIGNED_LOG1P_CATEGORY,
    signed_log1p_config,
    signed_log1p_transform,
    transform_train_test_signed_log1p,
)


def test_signed_log1p_transform_matches_formula() -> None:
    values = np.asarray([[-3.0, 0.0, 3.0]], dtype=float)

    transformed = signed_log1p_transform(values, scale=3.0)

    assert np.allclose(transformed, np.asarray([[-np.log1p(1.0), 0.0, np.log1p(1.0)]]))


def test_signed_log1p_accepts_one_pass_feature_iterables() -> None:
    values = (row for row in [[-3.0, 0.0, 3.0]])

    transformed = signed_log1p_transform(values, scale=3.0)

    assert np.allclose(transformed, np.asarray([[-np.log1p(1.0), 0.0, np.log1p(1.0)]]))


def test_train_test_signed_log1p_metadata() -> None:
    result = transform_train_test_signed_log1p(
        train_features=[[-1.0, 0.0], [1.0, 2.0]],
        test_features=[[3.0, -3.0]],
        config={"scale": 2.0},
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert result.metadata["signed_log1p_protocol_category"] == SIGNED_LOG1P_CATEGORY
    assert result.metadata["signed_log1p_has_fitted_parameters"] is False
    assert result.metadata["signed_log1p_uses_labels"] is False
    assert result.metadata["signed_log1p_valid_for_strict_source_only"] is True
    assert result.metadata["signed_log1p_scale"] == 2.0


def test_train_test_signed_log1p_accepts_one_pass_feature_iterables() -> None:
    result = transform_train_test_signed_log1p(
        train_features=(row for row in [[-1.0, 0.0], [1.0, 2.0]]),
        test_features=(row for row in [[3.0, -3.0]]),
        config={"scale": 2.0},
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert result.metadata["signed_log1p_n_train_rows"] == 2
    assert result.metadata["signed_log1p_n_test_rows"] == 1


def test_train_test_signed_log1p_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        transform_train_test_signed_log1p(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_signed_log1p_config_validation() -> None:
    assert signed_log1p_config(scale="2.5").scale == 2.5

    with pytest.raises(ValueError, match="scale"):
        signed_log1p_config(scale=0.0)


@pytest.mark.parametrize("scale", [True, np.asarray(2.0), np.asarray([2.0])])
def test_signed_log1p_config_rejects_boolean_and_array_scales(scale: object) -> None:
    with pytest.raises(ValueError, match="scale"):
        signed_log1p_config(scale=scale)  # type: ignore[arg-type]
