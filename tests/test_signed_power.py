from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.signed_power import (
    SIGNED_POWER_CATEGORY,
    SignedPowerConfig,
    signed_power_config,
    signed_power_transform,
    transform_train_test_signed_power,
)


def test_signed_power_transform_matches_formula() -> None:
    transformed = signed_power_transform([[-4.0, -1.0, 0.0, 1.0, 9.0]], power=0.5)

    assert np.allclose(transformed, np.asarray([[-2.0, -1.0, 0.0, 1.0, 3.0]]))


def test_signed_power_transform_accepts_one_pass_feature_iterables() -> None:
    values = (row for row in [[-4.0, -1.0, 0.0, 1.0, 9.0]])

    transformed = signed_power_transform(values, power=0.5)

    assert np.allclose(transformed, np.asarray([[-2.0, -1.0, 0.0, 1.0, 3.0]]))


def test_signed_power_power_changes_output() -> None:
    transformed = signed_power_transform([[4.0, -4.0]], power=1.0)

    assert np.allclose(transformed, np.asarray([[4.0, -4.0]]))


def test_transform_train_test_signed_power_metadata() -> None:
    result = transform_train_test_signed_power(
        train_features=[[4.0, 9.0]],
        test_features=[[-1.0, 0.0]],
    )

    assert result.train_features.shape == (1, 2)
    assert result.test_features.shape == (1, 2)
    assert result.metadata["signed_power_protocol_category"] == SIGNED_POWER_CATEGORY
    assert result.metadata["signed_power_has_fitted_parameters"] is False
    assert result.metadata["signed_power_uses_labels"] is False
    assert result.metadata["signed_power_valid_for_strict_source_only"] is True


def test_transform_train_test_signed_power_accepts_one_pass_feature_iterables() -> None:
    result = transform_train_test_signed_power(
        train_features=(row for row in [[4.0, 9.0], [16.0, 25.0]]),
        test_features=(row for row in [[-1.0, 0.0]]),
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert result.metadata["signed_power_n_train_rows"] == 2
    assert result.metadata["signed_power_n_test_rows"] == 1


def test_signed_power_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        transform_train_test_signed_power(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_signed_power_config_validation() -> None:
    assert signed_power_config(power="2.5").power == 2.5
    with pytest.raises(ValueError, match="power"):
        signed_power_config(power=0.0)


@pytest.mark.parametrize("power", [True, np.asarray(2.0), np.asarray([2.0])])
def test_signed_power_config_rejects_boolean_and_array_powers(power: object) -> None:
    with pytest.raises(ValueError, match="power"):
        signed_power_config(power=power)  # type: ignore[arg-type]


@pytest.mark.parametrize("power", [True, np.asarray(2.0), np.asarray([2.0])])
def test_signed_power_revalidates_config_objects(power: object) -> None:
    with pytest.raises(ValueError, match="power"):
        transform_train_test_signed_power(train_features=[[1.0]], test_features=[[1.0]], config=SignedPowerConfig(power=power))  # type: ignore[arg-type]
