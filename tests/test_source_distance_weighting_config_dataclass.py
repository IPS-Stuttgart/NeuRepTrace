from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_distance_weighting import SourceDistanceWeightConfig, compute_source_distance_weights


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
