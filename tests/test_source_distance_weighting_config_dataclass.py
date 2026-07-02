from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_distance_weighting import (
    SourceDistanceWeightConfig,
    compute_source_distance_weights,
    source_distance_weight_config,
)


def test_dataclass_source_distance_weight_config_is_normalized_before_use() -> None:
    features = np.asarray([[0.0], [0.1], [0.2], [10.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=object)
    config = SourceDistanceWeightConfig(
        group_mode="pooled",
        temperature="2.0",
        min_weight="0.25",
        normalize_weights="false",
        robust="false",
        epsilon="1e-8",
    )

    result = compute_source_distance_weights(features, labels, config=config)

    assert result.metadata["source_distance_weighting_group_mode"] == "global"
    assert result.metadata["source_distance_weighting_temperature"] == 2.0
    assert result.metadata["source_distance_weighting_min_weight"] == 0.25
    assert result.metadata["source_distance_weighting_normalize_weights"] is False
    assert result.metadata["source_distance_weighting_robust"] is False
    assert result.metadata["source_distance_weighting_uses_source_labels"] is False
    assert not np.isclose(float(np.mean(result.sample_weights)), 1.0)


def test_source_distance_weight_config_accepts_numpy_scalar_controls() -> None:
    config = source_distance_weight_config(
        temperature=np.asarray(2.0),
        min_weight=np.asarray(0.25),
        normalize_weights=np.asarray(False),
        robust=np.asarray(False),
        epsilon=np.asarray(1e-8),
    )

    assert config.temperature == 2.0
    assert config.min_weight == 0.25
    assert config.normalize_weights is False
    assert config.robust is False
    assert config.epsilon == 1e-8


def test_dataclass_source_distance_weight_config_accepts_numpy_scalar_controls() -> None:
    features = np.asarray([[0.0], [0.1], [0.2], [10.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=object)
    config = SourceDistanceWeightConfig(
        group_mode="pooled",
        temperature=np.asarray(2.0),  # type: ignore[arg-type]
        min_weight=np.asarray(0.25),  # type: ignore[arg-type]
        normalize_weights=np.asarray(False),  # type: ignore[arg-type]
        robust=np.asarray(False),  # type: ignore[arg-type]
        epsilon=np.asarray(1e-8),  # type: ignore[arg-type]
    )

    result = compute_source_distance_weights(features, labels, config=config)

    assert result.metadata["source_distance_weighting_group_mode"] == "global"
    assert result.metadata["source_distance_weighting_temperature"] == 2.0
    assert result.metadata["source_distance_weighting_min_weight"] == 0.25
    assert result.metadata["source_distance_weighting_normalize_weights"] is False
    assert result.metadata["source_distance_weighting_robust"] is False
    assert not np.isclose(float(np.mean(result.sample_weights)), 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": np.asarray([2.0])}, "temperature"),
        ({"temperature": np.asarray(True)}, "temperature"),
        ({"min_weight": np.asarray([0.25])}, "min_weight"),
        ({"normalize_weights": np.asarray([False])}, "normalize_weights"),
        ({"robust": np.asarray([False])}, "robust"),
        ({"epsilon": np.asarray([1e-8])}, "epsilon"),
    ],
)
def test_source_distance_weight_config_rejects_numpy_vector_or_bool_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source_distance_weight_config(**kwargs)  # type: ignore[arg-type]
