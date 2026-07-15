from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_distance_weighting import compute_source_distance_weights


def test_distance_scores_preserve_large_finite_values() -> None:
    features = np.asarray([[0.0], [0.0], [0.0], [0.0], [1.0e20]], dtype=float)
    labels = np.zeros(features.shape[0], dtype=int)

    with np.errstate(over="raise", invalid="raise"):
        result = compute_source_distance_weights(
            features,
            labels,
            config={"group_mode": "global"},
        )

    assert result.distance_scores.dtype == np.float64
    assert np.isfinite(result.distance_scores).all()
    assert result.distance_scores[-1] > np.finfo(np.float32).max
    assert result.metadata["source_distance_weighting_score_max"] == result.distance_scores[-1]
