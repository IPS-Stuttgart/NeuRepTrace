import numpy as np
import pytest

from neureptrace.decoding.mcca import class_alignment_matrices, fit_class_mcca
from neureptrace.decoding.mcca_target import class_alignment_matrix


def _source_alignment_inputs():
    features = {
        "a": np.array([[1.0], [2.0], [3.0], [4.0]]),
        "b": np.array([[11.0], [12.0], [13.0], [14.0]]),
    }
    labels = {
        "a": np.array([0, 0, 1, 1]),
        "b": np.array([0, 0, 1, 1]),
    }
    return features, labels


def test_source_class_repetition_rejects_fractional_count():
    features, labels = _source_alignment_inputs()

    with pytest.raises(ValueError, match="positive integer or None"):
        class_alignment_matrices(
            features,
            labels,
            sample_mode="class_repetition",
            n_repetitions_per_class=1.5,
        )


def test_fit_class_mcca_rejects_fractional_count_before_truncation():
    features, labels = _source_alignment_inputs()

    with pytest.raises(ValueError, match="positive integer or None"):
        fit_class_mcca(
            features,
            labels,
            sample_mode="class_repetition",
            n_repetitions_per_class=np.float64(1.5),
        )


def test_target_class_repetition_rejects_fractional_count():
    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="positive integer or None"):
        class_alignment_matrix(
            features,
            labels,
            sample_mode="class_repetition",
            n_repetitions_per_class=1.5,
        )
