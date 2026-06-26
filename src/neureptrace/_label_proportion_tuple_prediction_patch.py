"""Preserve tuple-valued weak label-proportion predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

import neureptrace.decoding.label_proportions as _label_proportions


def _object_label_vector(labels: tuple[Any, ...]) -> np.ndarray:
    """Return a 1D object array without letting NumPy expand tuple labels."""

    label_vector = np.empty(len(labels), dtype=object)
    label_vector[:] = list(labels)
    return label_vector


def _predict_labels_from_label_proportions(result: _label_proportions.WeakLabelProportionCalibrationResult) -> np.ndarray:
    """Return argmax labels while treating composite class ids atomically."""

    class_vector = _object_label_vector(tuple(result.classes))
    return class_vector[np.argmax(result.probabilities, axis=1)]


def install() -> None:
    """Install the tuple-label-safe weak label-proportion predictor."""

    _label_proportions.predict_labels_from_label_proportions = _predict_labels_from_label_proportions
