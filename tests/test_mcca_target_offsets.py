import numpy as np
import pytest

from neureptrace.decoding.mcca_target import class_alignment_matrix


def test_class_alignment_matrix_rejects_boolean_selected_offsets():
    features = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    labels = np.array([1, 1, 1, 2, 2, 2])

    with pytest.raises(ValueError, match="integer offsets"):
        class_alignment_matrix(
            features,
            labels,
            classes=np.array([1, 2]),
            sample_mode="class_repetition",
            selected_offsets_by_class={
                0: np.array([True, False]),
                1: np.array([0, 1]),
            },
        )
