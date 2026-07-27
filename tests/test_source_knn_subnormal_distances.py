from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_knn import fit_source_knn_decoder


def test_distance_weighting_stays_finite_for_subnormal_distances() -> None:
    tiny = np.nextafter(0.0, 1.0)
    result = fit_source_knn_decoder(
        source_features=[[0.0], [4.0 * tiny]],
        source_labels=["near", "far"],
        test_features=[[tiny]],
        config={
            "k": 2,
            "weights": "distance",
            "standardize": False,
            "epsilon": tiny,
        },
    )

    assert result.predictions.tolist() == ["near"]
    assert np.all(np.isfinite(result.probabilities))
    np.testing.assert_allclose(result.probabilities, [[0.75, 0.25]])
