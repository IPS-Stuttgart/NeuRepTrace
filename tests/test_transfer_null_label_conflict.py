import numpy as np
import pytest

from neureptrace.decoding.transfer import append_null_class_features, cross_validate_feature_decoding


class _ConstantClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, features):
        return np.full(features.shape[0], self.label)


def test_cross_validate_feature_decoding_preserves_observed_zero_label_without_null_features():
    result = cross_validate_feature_decoding(
        np.array([[-2.0], [2.0], [-1.0], [1.0]]),
        np.array([0, 1, 0, 1]),
        n_folds=2,
        components_pca=float("inf"),
        fit_model=lambda _features, _labels: _ConstantClassifier(0),
    )

    assert result.predictions.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result.accuracy == 0.5


def test_append_null_class_features_rejects_ambiguous_observed_null_label():
    with pytest.raises(ValueError, match="null_label must not overlap observed labels"):
        append_null_class_features(
            np.array([[-2.0], [2.0]]),
            np.array([0, 1]),
            np.array([[0.1], [0.2]]),
        )


def test_append_null_class_features_allows_disjoint_null_label_for_zero_based_classes():
    features, labels = append_null_class_features(
        np.array([[-2.0], [2.0]]),
        np.array([0, 1]),
        np.array([[0.1], [0.2]]),
        null_label=-1,
    )

    assert features.tolist() == [[-2.0], [2.0], [0.1], [0.2]]
    assert labels.tolist() == [0, 1, -1, -1]
