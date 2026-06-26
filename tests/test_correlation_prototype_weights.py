import numpy as np
import pytest

from neureptrace.decoding.classifiers import train_multiclass_classifier


def test_weighted_correlation_prototype_rejects_class_without_mass():
    features = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]])
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="positive total weight"):
        train_multiclass_classifier(
            features,
            labels,
            "correlation-prototype",
            None,
            sample_weight=np.array([1.0, 1.0, 0.0, 0.0]),
        )
