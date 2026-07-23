from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_bagging import fit_source_bagging_decoder


def test_source_bagging_class_repair_preserves_without_replacement_sampling() -> None:
    source_features = np.arange(8, dtype=float).reshape(-1, 1)
    source_labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2], dtype=object)
    test_features = np.asarray([[0.5], [3.5], [6.5]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={
            "n_estimators": 1,
            "sample_fraction": 0.5,
            "bootstrap_rows": False,
            "class_balanced": False,
            "random_state": 4,
        },
    )

    sampled_rows = result.row_indices[0]
    assert sampled_rows.size == 4
    assert np.unique(sampled_rows).size == sampled_rows.size
    assert set(source_labels[sampled_rows].tolist()) == {0, 1, 2}
