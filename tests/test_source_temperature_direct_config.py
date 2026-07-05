from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_temperature import SourceTemperatureConfig, fit_source_temperature_scaling


def test_source_temperature_config_dataclass_normalizes_direct_construction() -> None:
    cfg = SourceTemperatureConfig(temperatures="0.5; 1", epsilon=np.float64(1e-9))

    assert cfg.temperatures == (0.5, 1.0)
    assert np.isclose(cfg.epsilon, 1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperatures": (True,)},
        {"temperatures": (np.bool_(True),)},
        {"temperatures": (np.asarray([1.0]),)},
        {"temperatures": ()},
        {"epsilon": True},
        {"epsilon": np.asarray([1e-9])},
        {"epsilon": 1.0},
    ],
)
def test_source_temperature_config_dataclass_rejects_invalid_direct_construction(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SourceTemperatureConfig(**kwargs)


def test_fit_source_temperature_scaling_accepts_normalized_direct_config_object() -> None:
    result = fit_source_temperature_scaling(
        source_probabilities=[[0.6, 0.4], [0.4, 0.6]],
        source_labels=[0, 1],
        test_probabilities=[[0.5, 0.5]],
        config=SourceTemperatureConfig(temperatures="1,2", epsilon="1e-9"),
    )

    assert result.temperature in {1.0, 2.0}
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.isclose(result.metadata["source_temperature_epsilon"], 1e-9)
