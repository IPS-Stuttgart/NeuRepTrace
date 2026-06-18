import numpy as np

from neureptrace.decoding.mcca import class_alignment_matrices, fit_class_mcca


def _feature_blocks():
    features = {
        "s1": np.array(
            [
                [1.0, 0.0, 0.0],
                [1.1, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 1.1, 0.0],
            ]
        ),
        "s2": np.array(
            [
                [0.9, 0.0, 0.1],
                [1.0, 0.1, 0.1],
                [0.0, 0.9, 0.1],
                [0.1, 1.0, 0.1],
            ]
        ),
    }
    return features


def _tuple_label_vector():
    labels = np.empty(4, dtype=object)
    for index, value in enumerate(
        [
            ("face", "famous"),
            ("face", "famous"),
            ("face", "scrambled"),
            ("face", "scrambled"),
        ]
    ):
        labels[index] = value
    return labels


def test_mcca_class_mean_accepts_tuple_anchor_values():
    features = _feature_blocks()
    labels = {subject: _tuple_label_vector() for subject in features}

    alignment = class_alignment_matrices(features, labels, sample_mode="class_mean")

    assert alignment.n_classes == 2
    assert alignment.n_alignment_rows == 2
    assert alignment.aligned_by_subject["s1"].shape == (2, 3)


def test_mcca_class_repetition_accepts_tuple_anchor_values():
    features = _feature_blocks()
    labels = {subject: _tuple_label_vector() for subject in features}

    model, alignment = fit_class_mcca(
        features,
        labels,
        sample_mode="class_repetition",
        n_repetitions_per_class=2,
        n_components=1,
    )

    assert alignment.n_alignment_rows == 4
    assert alignment.n_repetitions_per_class == 2
    assert model.n_components == 1
    assert model.transform("s1", features["s1"]).shape == (4, 1)
