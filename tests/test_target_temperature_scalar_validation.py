from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_temperature_scaling import (
    TargetTemperatureConfig,
    apply_temperature_to_probabilities,
    fit_target_temperature_scaling,
    target_temperature_config,
)


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True)])
def test_target_temperature_rejects_boolean_temperature_scalars(value: object) -> None:
    with pytest.raises(ValueError, match="temperature must be positive and finite"):
        apply_temperature_to_probabilities([[0.8, 0.2]], temperature=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True)])
def test_target_temperature_rejects_boolean_epsilon_scalars(value: object) -> None:
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        target_temperature_config(epsilon=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [np.asarray([1.0]), np.asarray([[1.0]])])
def test_target_temperature_rejects_array_valued_scalar_controls(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="temperature must be positive and finite"):
        apply_temperature_to_probabilities([[0.8, 0.2]], temperature=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        target_temperature_config(epsilon=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="temperature_grid must be positive and finite"):
        target_temperature_config(temperature_grid=[value])  # type: ignore[list-item]


def test_target_temperature_accepts_zero_dimensional_numeric_scalars() -> None:
    config = target_temperature_config(temperature_grid=[np.asarray(2.0)], epsilon=np.asarray(1.0e-9))
    assert config.temperature_grid == (2.0,)
    assert config.epsilon == pytest.approx(1.0e-9)
    transformed = apply_temperature_to_probabilities([[0.8, 0.2]], temperature=np.asarray(2.0))
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0)


def test_direct_config_rejects_invalid_controls_when_fitted() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        fit_target_temperature_scaling(
            calibration_probabilities=[[0.8, 0.2], [0.2, 0.8]],
            calibration_labels=[0, 1],
            probabilities=[[0.7, 0.3]],
            classes=[0, 1],
            config=TargetTemperatureConfig(temperature_grid=(1.0,), epsilon=True),  # type: ignore[arg-type]
        )
