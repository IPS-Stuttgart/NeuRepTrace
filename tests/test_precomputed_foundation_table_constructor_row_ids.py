from __future__ import annotations

import numpy as np

from neureptrace.decoding.precomputed_foundation import PrecomputedFoundationFeatureTable, align_precomputed_foundation_features


def test_direct_feature_table_preserves_matrix_encoded_composite_row_ids() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    row_ids = np.asarray(
        [
            ["sub-01", "trial-0"],
            ["sub-02", "trial-1"],
        ],
        dtype=object,
    )

    table = PrecomputedFoundationFeatureTable(
        features=features,
        row_ids=row_ids,
        feature_names=("alpha", "beta"),
    )
    aligned = align_precomputed_foundation_features(table, [("sub-02", "trial-1")])

    assert table.row_ids == (("sub-01", "trial-0"), ("sub-02", "trial-1"))
    assert np.allclose(aligned, np.asarray([[0.0, 1.0]], dtype=np.float32))
