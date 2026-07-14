from __future__ import annotations

import numpy as np

from neureptrace.decoding import transfer


class _ConstantClassifier:
    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.label)


def test_append_null_class_features_preserves_non_integral_numeric_null_label() -> None:
    features, labels = transfer.append_null_class_features(
        [[1.0], [2.0]],
        np.asarray([1, 2], dtype=int),
        [[0.1], [0.2]],
        null_label=0.5,
    )

    assert features.tolist() == [[1.0], [2.0], [0.1], [0.2]]
    assert labels.dtype == object
    assert labels.tolist() == [1, 2, 0.5, 0.5]


def test_cross_validation_excludes_losslessly_stored_numeric_null_rows() -> None:
    result = transfer.cross_validate_feature_decoding(
        np.asarray([[-2.0], [1.0], [-1.0], [2.0]]),
        np.asarray([1, 2, 1, 2], dtype=int),
        null_features=np.zeros((4, 1), dtype=float),
        null_label=0.5,
        n_folds=2,
        components_pca=float("inf"),
        fit_model=lambda _features, _labels: _ConstantClassifier(1),
    )

    assert result.predictions.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert result.accuracy == 0.5
