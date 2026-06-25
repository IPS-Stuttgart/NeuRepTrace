from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401 - installs runtime compatibility patches
from neureptrace.decoding.classifiers import (
    CorrelationPrototypeClassifier,
    encode_classifier_labels,
    train_multiclass_classifier,
)


def test_encode_classifier_labels_preserves_tuple_labels_as_atomic_classes() -> None:
    labels = [("visual", "left"), ("motor", "right"), ("visual", "left")]

    classes, encoded = encode_classifier_labels(labels)

    assert classes.shape == (2,)
    assert all(isinstance(class_label, tuple) for class_label in classes)
    assert encoded.shape == (3,)
    assert encoded[0] == encoded[2]
    assert encoded[0] != encoded[1]


def test_train_multiclass_classifier_predicts_original_tuple_labels() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ]
    )
    labels = [
        ("visual", "left"),
        ("visual", "left"),
        ("motor", "right"),
        ("motor", "right"),
    ]

    model = train_multiclass_classifier(features, labels, "knn", 1)
    predictions = model.predict(features)

    assert predictions.dtype == object
    assert predictions.tolist() == labels


def test_correlation_prototype_classifier_accepts_tuple_labels_directly() -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
    )
    labels = [
        ("visual", "left"),
        ("visual", "left"),
        ("motor", "right"),
        ("motor", "right"),
    ]

    model = CorrelationPrototypeClassifier().fit(features, labels)
    predictions = model.predict(features)

    assert predictions.dtype == object
    assert predictions.tolist() == labels


def test_correlation_prototype_tuple_labels_preserve_zero_weight_class_guard() -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
    )
    labels = [
        ("visual", "left"),
        ("visual", "left"),
        ("motor", "right"),
        ("motor", "right"),
    ]
    sample_weight = [1.0, 1.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="positive total weight"):
        CorrelationPrototypeClassifier().fit(
            features,
            labels,
            sample_weight=sample_weight,
        )
