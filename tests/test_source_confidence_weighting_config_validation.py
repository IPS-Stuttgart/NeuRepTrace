from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_confidence_weighting import SourceConfidenceWeightConfig, source_confidence_weight_config


def test_source_confidence_weighting_config_validation_smoke():
    assert source_confidence_weight_config(normalize_weights="false").normalize_weights is False


def test_source_confidence_weighting_dataclass_normalizes_direct_construction() -> None:
    cfg = SourceConfidenceWeightConfig(
        mode="max-prob",
        min_weight="0.2",
        normalize_weights="false",
        epsilon=np.asarray(1e-9),
    )

    assert cfg.mode == "confidence"
    assert cfg.min_weight == 0.2
    assert cfg.normalize_weights is False
    assert cfg.epsilon == 1e-9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"min_weight": True},
        {"min_weight": np.asarray([0.2])},
        {"min_weight": -0.1},
        {"min_weight": 1.1},
        {"normalize_weights": np.asarray([True])},
        {"normalize_weights": 2},
        {"epsilon": False},
        {"epsilon": np.asarray([1e-12])},
        {"epsilon": 0.0},
    ],
)
def test_source_confidence_weighting_dataclass_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SourceConfidenceWeightConfig(**kwargs)
