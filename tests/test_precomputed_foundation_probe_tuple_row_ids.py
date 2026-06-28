from __future__ import annotations

import numpy as np

from neureptrace.decoding.precomputed_foundation import fit_precomputed_foundation_probe, make_precomputed_foundation_feature_table


def test_fit_probe_preserves_matrix_encoded_tuple_row_ids() -> None:
    features = np.asarray([[4.0, 0.0], [0.0, 4.0], [5.0, 0.0], [0.0, 5.0]], dtype=float)
    table = make_precomputed_foundation_feature_table(features, row_ids=[(1, 0), (1, 1), (2, 0), (2, 1)])

    result = fit_precomputed_foundation_probe(
        feature_table=table,
        train_row_ids=np.asarray([[1, 0], [1, 1]]),
        train_labels=[0, 1],
        test_row_ids=np.asarray([[2, 0], [2, 1]]),
        classifier_C=1000.0,
    )

    assert np.allclose(result.train_features, np.asarray([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32))
    assert np.allclose(result.test_features, np.asarray([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32))
    assert result.predictions.shape == (2,)
