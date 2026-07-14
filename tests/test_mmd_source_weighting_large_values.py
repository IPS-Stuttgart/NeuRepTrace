from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mmd_source_weighting import mmd_source_group_weights


def test_mmd_source_weighting_handles_large_finite_feature_vectors() -> None:
    magnitude = 1.0e200
    target_features = np.asarray(
        [
            [magnitude, -magnitude],
            [magnitude, -magnitude],
        ],
        dtype=float,
    )
    source_features = {
        "near": target_features.copy(),
        "far": -target_features,
    }

    with np.errstate(over="raise", invalid="raise"):
        result = mmd_source_group_weights(
            source_features,
            target_features,
            gamma="scale",
            temperature=0.25,
        )

    assert np.isfinite(list(result.mmd_squared.values())).all()
    assert result.mmd_squared["near"] == pytest.approx(0.0)
    assert result.mmd_squared["far"] == pytest.approx(2.0)
    assert result.weights["near"] > result.weights["far"]
