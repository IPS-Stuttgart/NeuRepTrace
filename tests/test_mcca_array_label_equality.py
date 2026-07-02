import numpy as np

from neureptrace.decoding.mcca import class_alignment_matrices
from neureptrace.decoding.mcca_target import class_alignment_matrix


def _array_label_vector(values):
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = np.asarray(value, dtype=object)
    return vector


def _label_tuples(labels):
    return [tuple(np.asarray(label, dtype=object).tolist()) for label in labels]


def test_mcca_class_alignment_accepts_array_valued_object_labels():
    features = {
        "a": np.array([[1.0], [3.0], [10.0], [30.0]]),
        "b": np.array([[101.0], [103.0], [110.0], [130.0]]),
    }
    labels = {
        "a": _array_label_vector(
            [
                ["run-01", "stim-a"],
                ["run-01", "stim-a"],
                ["run-01", "stim-b"],
                ["run-01", "stim-b"],
            ]
        ),
        "b": _array_label_vector(
            [
                ["run-01", "stim-a"],
                ["run-01", "stim-a"],
                ["run-01", "stim-b"],
                ["run-01", "stim-b"],
            ]
        ),
    }

    mean_alignment = class_alignment_matrices(features, labels, sample_mode="class_mean")

    assert _label_tuples(mean_alignment.classes) == [("run-01", "stim-a"), ("run-01", "stim-b")]
    np.testing.assert_allclose(mean_alignment.aligned_by_subject["a"], [[2.0], [20.0]])
    np.testing.assert_allclose(mean_alignment.aligned_by_subject["b"], [[102.0], [120.0]])

    repetition_alignment = class_alignment_matrices(
        features,
        labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        repetition_selection="first",
    )
    np.testing.assert_allclose(repetition_alignment.aligned_by_subject["a"].ravel(), [1.0, 3.0, 10.0, 30.0])


def test_target_mcca_class_alignment_accepts_explicit_array_valued_classes():
    features = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]])
    labels = _array_label_vector(
        [
            ["run-01", "stim-a"],
            ["run-01", "stim-b"],
            ["run-01", "stim-a"],
            ["run-01", "stim-b"],
        ]
    )
    classes = _array_label_vector([["run-01", "stim-b"], ["run-01", "stim-a"]])

    aligned = class_alignment_matrix(features, labels, classes=classes, sample_mode="class_mean")

    np.testing.assert_allclose(aligned, [[0.0, 3.0], [2.0, 0.0]])
