import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from neureptrace.decoding.transfer_component_analysis import fit_tca_transfer_classifier


class _NearestCompositeLabelClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, features, labels, sample_weight=None):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=object)
        self.classes_ = _unique_labels(labels)
        self.means_ = np.vstack([features[_label_mask(labels, label)].mean(axis=0) for label in self.classes_])
        return self

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        distances = np.sum((features[:, None, :] - self.means_[None, :, :]) ** 2, axis=2)
        predictions = np.empty(features.shape[0], dtype=object)
        for index, class_index in enumerate(np.argmin(distances, axis=1)):
            predictions[index] = self.classes_[class_index]
        return predictions


def _values_equal(left, right):
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _unique_labels(labels):
    unique = []
    for label in labels:
        if not any(_values_equal(label, existing) for existing in unique):
            unique.append(label)
    vector = np.empty(len(unique), dtype=object)
    for index, label in enumerate(unique):
        vector[index] = label
    return vector


def _label_mask(labels, target):
    return np.asarray([_values_equal(label, target) for label in labels], dtype=bool)


def test_tca_transfer_classifier_preserves_rectangular_composite_source_labels():
    result = fit_tca_transfer_classifier(
        source_features=np.array(
            [
                [-3.0, -3.0],
                [-2.6, -2.4],
                [2.4, 2.6],
                [3.0, 3.0],
            ]
        ),
        source_labels=np.array(
            [
                ["left", "seen"],
                ["left", "seen"],
                ["right", "seen"],
                ["right", "seen"],
            ],
            dtype=object,
        ),
        target_features=np.array([[-2.8, -2.7], [2.8, 2.9]]),
        classifier=_NearestCompositeLabelClassifier(),
        n_components=2,
        kernel="linear",
    )

    assert result.classes.dtype == object
    assert result.classes.tolist() == [("left", "seen"), ("right", "seen")]
    assert result.predictions.dtype == object
    assert result.predictions.tolist() == [("left", "seen"), ("right", "seen")]
    assert result.metadata["tca_classifier_n_classes"] == 2
