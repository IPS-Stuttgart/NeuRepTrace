from __future__ import annotations

import numpy as np

from neureptrace.decoding.transfer import cross_validate_feature_decoding


class _ConstantClassifier:
    def __init__(self, label: np.int64) -> None:
        self.label = label

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.label, dtype=np.int64)


def test_cross_validation_preserves_large_integer_predictions() -> None:
    large_label = np.int64(2**53 + 1)
    other_label = np.int64(2**53 + 3)
    labels = np.asarray([large_label, other_label, large_label, other_label], dtype=np.int64)

    result = cross_validate_feature_decoding(
        np.asarray([[-2.0], [1.0], [-1.0], [2.0]]),
        labels,
        n_folds=2,
        components_pca=float("inf"),
        fit_model=lambda _features, _labels: _ConstantClassifier(large_label),
    )

    assert result.predictions.dtype == object
    assert [int(value) for value in result.predictions] == [int(large_label)] * labels.size
    assert result.accuracy == 0.5
