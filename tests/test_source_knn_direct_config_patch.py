from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_knn import SourceKNNConfig
from neureptrace.decoding.source_temperature import SourceTemperatureConfig


def test_source_knn_config_direct_construction_normalizes_values() -> None:
    cfg = SourceKNNConfig(
        k=np.asarray("2"),
        weights="inverse-distance",
        standardize=np.asarray(False),
        epsilon=np.asarray("1e-4"),
    )

    assert cfg.k == 2
    assert cfg.weights == "distance"
    assert cfg.standardize is False
    assert cfg.epsilon == pytest.approx(1e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": np.asarray([1])}, "k"),
        ({"k": True}, "k"),
        ({"weights": "bad"}, "weight mode"),
        ({"standardize": np.asarray([False])}, "standardize"),
        ({"standardize": "sometimes"}, "standardize"),
        ({"epsilon": np.asarray([1e-4])}, "epsilon"),
        ({"epsilon": False}, "epsilon"),
    ],
)
def test_source_knn_config_direct_construction_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SourceKNNConfig(**kwargs)


def test_source_temperature_config_direct_construction_normalizes_values() -> None:
    cfg = SourceTemperatureConfig(temperatures="0.5;1;2", epsilon="1e-9")

    assert cfg.temperatures == (0.5, 1.0, 2.0)
    assert cfg.epsilon == pytest.approx(1e-9)

    array_cfg = SourceTemperatureConfig(
        temperatures=np.asarray([np.float64(0.5), np.int64(1)]),
        epsilon=np.float64(1e-9),
    )

    assert array_cfg.temperatures == (0.5, 1.0)
    assert array_cfg.epsilon == pytest.approx(1e-9)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperatures": (True,)}, "temperatures"),
        ({"temperatures": (np.asarray([1.0]),)}, "temperatures"),
        ({"temperatures": ()}, "temperatures"),
        ({"epsilon": np.asarray([1e-9])}, "epsilon"),
        ({"epsilon": 1.0}, "epsilon"),
    ],
)
def test_source_temperature_config_direct_construction_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SourceTemperatureConfig(**kwargs)
