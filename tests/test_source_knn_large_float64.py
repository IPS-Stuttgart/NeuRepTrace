from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_knn import fit_source_knn_decoder


def test_source_knn_preserves_large_float64_raw_distances() -> None:
    source = np.asarray([[1e200, 0.0], [-1e200, 0.0]], dtype=float)
    labels = np.asarray(["positive", "negative"], dtype=object)
    test = np.asarray([[1e200 + 1e185, 0.0]], dtype=float)

    result = fit_source_knn_decoder(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config={"k": 1, "weights": "distance", "standardize": False},
    )

    assert result.predictions.tolist() == ["positive"]
    assert np.all(np.isfinite(result.reference.features))
    assert np.all(np.isfinite(result.neighbor_distances))
    assert result.neighbor_distances[0, 0] > np.finfo(np.float32).max
    np.testing.assert_allclose(result.probabilities, [[1.0, 0.0]])
