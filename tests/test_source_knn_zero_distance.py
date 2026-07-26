from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_knn import fit_source_knn_decoder


def test_distance_weighting_uses_only_exact_source_matches() -> None:
    result = fit_source_knn_decoder(
        source_features=[[0.0], [1.0], [2.0]],
        source_labels=["exact", "far", "far"],
        test_features=[[0.0]],
        config={
            "k": 3,
            "weights": "distance",
            "standardize": False,
            "epsilon": 10.0,
        },
    )

    assert result.predictions.tolist() == ["exact"]
    np.testing.assert_allclose(result.probabilities, [[1.0, 0.0]])
