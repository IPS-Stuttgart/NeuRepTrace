import numpy as np
import pytest

from neureptrace.temporal_smoothing import _top_k_accuracy, _top_k_accuracy_from_label_values


def test_temporal_smoothing_top_k_rejects_non_positive_or_fractional_k() -> None:
    probabilities = np.array([[0.5, 0.5], [0.1, 0.9]])
    labels = np.array([0, 1])

    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            _top_k_accuracy(probabilities, labels, k=value)

        with pytest.raises(ValueError, match="k must be a positive integer"):
            _top_k_accuracy_from_label_values(probabilities, labels, (0, 1), k=value)


def test_temporal_smoothing_top_k_still_caps_to_class_count() -> None:
    probabilities = np.array([[0.5, 0.5], [0.1, 0.9]])
    labels = np.array([1, 0])

    assert _top_k_accuracy(probabilities, labels, k=3) == 1.0
    assert _top_k_accuracy_from_label_values(probabilities, labels, (0, 1), k=3) == 1.0
