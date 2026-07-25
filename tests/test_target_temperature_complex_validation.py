from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_temperature_scaling import (
    apply_temperature_to_probabilities,
    fit_target_temperature_scaling,
    target_temperature_config,
)


def test_apply_temperature_rejects_complex_probability_arrays_before_float_coercion() -> None:
    probabilities = np.asarray([[0.8 + 0.1j, 0.2 - 0.1j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="probabilities must contain real-valued probabilities"):
        apply_temperature_to_probabilities(probabilities, temperature=1.0)


@pytest.mark.parametrize("complex_argument", ["calibration_probabilities", "probabilities"])
def test_fit_target_temperature_rejects_complex_probability_inputs(complex_argument: str) -> None:
    kwargs = {
        "calibration_probabilities": np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float),
        "calibration_labels": [0, 1],
        "probabilities": np.asarray([[0.6, 0.4]], dtype=float),
        "classes": [0, 1],
        "config": {"temperature_grid": [1.0]},
    }
    kwargs[complex_argument] = np.asarray(kwargs[complex_argument], dtype=np.complex128) + 0.1j

    with pytest.raises(ValueError, match=rf"{complex_argument} must contain real-valued probabilities"):
        fit_target_temperature_scaling(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        np.complex64(1.0 + 0.5j),
        np.complex128(1.0 + 0.5j),
        np.asarray(1.0 + 0.5j),
    ],
)
def test_target_temperature_rejects_complex_scalar_controls(value: object) -> None:
    with pytest.raises(ValueError, match="temperature must be positive and finite"):
        apply_temperature_to_probabilities([[0.8, 0.2]], temperature=value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        target_temperature_config(epsilon=value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="temperature_grid must be positive and finite"):
        target_temperature_config(temperature_grid=[value])  # type: ignore[list-item]
