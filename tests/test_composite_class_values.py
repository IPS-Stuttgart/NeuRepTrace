from __future__ import annotations

import importlib

import numpy as np

_module = importlib.import_module("neureptrace.decoding." + "label_shift")
adapt_label_shift_probabilities = _module.adapt_label_shift_probabilities
soft_confusion_matrix = _module.soft_confusion_matrix


def test_class_prior_accepts_rectangular_numpy_composite_values() -> None:
    target_probabilities = np.asarray([[0.9, 0.1], [0.85, 0.15], [0.2, 0.8]])
    source_labels = np.asarray([("left", 1), ("right", 2), ("left", 1), ("right", 2)], dtype=object)
    classes = np.asarray([("left", 1), ("right", 2)], dtype=object)

    result = adapt_label_shift_probabilities(
        target_probabilities,
        method="em",
        source_labels=source_labels,
        classes=classes,
        max_iter=20,
    )

    assert result.classes == (("left", 1), ("right", 2))
    assert np.allclose(result.source_prior, (0.5, 0.5))
    assert result.metadata["label_shift_n_classes"] == 2


def test_soft_confusion_accepts_rectangular_numpy_composite_values() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
    labels = np.asarray([("left", 1), ("left", 1), ("right", 2), ("right", 2)], dtype=object)
    classes = np.asarray([("left", 1), ("right", 2)], dtype=object)

    confusion = soft_confusion_matrix(probabilities, labels, classes=classes)

    assert confusion.shape == (2, 2)
    assert np.allclose(confusion.sum(axis=0), 1.0)
    assert confusion[0, 0] > confusion[1, 0]
    assert confusion[1, 1] > confusion[0, 1]
