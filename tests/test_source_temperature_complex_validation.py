from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_temperature import (
    SourceTemperatureConfig,
    apply_temperature,
    fit_source_temperature_scaling,
    source_temperature_config,
)


@pytest.mark.parametrize(
    "value",
    [
        np.complex64(1.0 + 0.5j),
        np.complex128(1.0 + 0.5j),
    ],
)
def test_source_temperature_rejects_complex_scalar_controls(value: object) -> None:
    with pytest.raises(ValueError, match="temperature must be positive and finite"):
        apply_temperature([[0.8, 0.2]], temperature=value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        source_temperature_config(epsilon=value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="temperatures must be positive and finite"):
        source_temperature_config(temperatures=[value])  # type: ignore[list-item]


def test_direct_source_temperature_config_rejects_complex_grid_values() -> None:
    with pytest.raises(ValueError, match="temperatures must be positive and finite"):
        SourceTemperatureConfig(temperatures=(np.complex128(1.0 + 0.5j),))  # type: ignore[arg-type]


@pytest.mark.parametrize("argument", ["source_probabilities", "test_probabilities"])
def test_source_temperature_fit_rejects_complex_probability_rows(argument: str) -> None:
    kwargs = {
        "source_probabilities": np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float),
        "source_labels": [0, 1],
        "test_probabilities": np.asarray([[0.6, 0.4]], dtype=float),
        "classes": [0, 1],
        "config": {"temperatures": [1.0]},
    }
    kwargs[argument] = np.asarray(kwargs[argument], dtype=np.complex128) + np.complex128(0.1j)

    with pytest.raises(ValueError, match=rf"{argument} must contain real-valued probabilities"):
        fit_source_temperature_scaling(**kwargs)  # type: ignore[arg-type]
