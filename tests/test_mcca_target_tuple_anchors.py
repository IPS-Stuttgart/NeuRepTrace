import numpy as np

from neureptrace.decoding.mcca_target import class_alignment_matrix


def _object_vector(values):
    vector = np.empty(len(values), dtype=object)
    vector[:] = list(values)
    return vector


def test_class_alignment_matrix_preserves_tuple_anchor_labels():
    features = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 10.0],
            [0.0, 14.0],
        ]
    )
    labels = [
        ("run-01", "stim-a"),
        ("run-01", "stim-a"),
        ("run-01", "stim-b"),
        ("run-01", "stim-b"),
    ]
    classes = _object_vector([("run-01", "stim-a"), ("run-01", "stim-b")])

    aligned = class_alignment_matrix(features, labels, classes=classes, sample_mode="class_mean")

    np.testing.assert_allclose(aligned, np.array([[2.0, 0.0], [0.0, 12.0]]))


def test_class_repetition_matrix_preserves_tuple_anchor_labels_with_offsets():
    features = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [10.0],
            [20.0],
            [30.0],
        ]
    )
    labels = [
        ("run-01", "stim-a"),
        ("run-01", "stim-a"),
        ("run-01", "stim-a"),
        ("run-01", "stim-b"),
        ("run-01", "stim-b"),
        ("run-01", "stim-b"),
    ]
    classes = _object_vector([("run-01", "stim-a"), ("run-01", "stim-b")])

    aligned = class_alignment_matrix(
        features,
        labels,
        classes=classes,
        sample_mode="class_repetition",
        selected_offsets_by_class={0: np.array([0, 2]), 1: np.array([1, 2])},
    )

    np.testing.assert_allclose(aligned, np.array([[1.0], [3.0], [20.0], [30.0]]))
