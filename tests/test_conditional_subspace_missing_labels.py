from __future__ import annotations

import numpy as np

from neureptrace.decoding.conditional_subspace import fit_jda


def test_fit_jda_treats_distinct_nan_labels_as_one_class() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=float,
    )
    source_labels = [float("nan"), float("nan"), "right", "right"]
    target_features = np.asarray([[0.05, 0.0], [3.05, 3.0]], dtype=float)

    result = fit_jda(
        source_features,
        source_labels,
        target_features,
        n_components=1,
        max_iterations=3,
    )

    assert np.isnan(result.pseudo_labels[0])
    assert result.pseudo_labels[1] == "right"
    assert np.all(np.isfinite(result.source_features))
    assert np.all(np.isfinite(result.target_features))
