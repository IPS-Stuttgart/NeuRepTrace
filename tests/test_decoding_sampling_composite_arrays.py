import numpy as np

from neureptrace.decoding import select_class_limited_indices


def test_class_limiter_keeps_numpy_composite_rows_atomic():
    labels = np.asarray(
        [
            ("face", "early"),
            ("house", "late"),
            ("face", "early"),
            ("house", "late"),
            ("face", "early"),
            ("house", "late"),
        ],
        dtype=object,
    )

    first = select_class_limited_indices(labels, 2, selection="first")
    random = select_class_limited_indices(labels, 2, selection="random", seed=0)

    assert first.tolist() == [0, 1, 2, 3]
    assert len(random) == 4
    assert np.all(random < labels.shape[0])


def test_class_limiter_keeps_sequence_of_numpy_array_labels_atomic():
    labels = [
        np.asarray(("face", "early"), dtype=object),
        np.asarray(("house", "late"), dtype=object),
        np.asarray(("face", "early"), dtype=object),
        np.asarray(("house", "late"), dtype=object),
        np.asarray(("face", "early"), dtype=object),
        np.asarray(("house", "late"), dtype=object),
    ]

    first = select_class_limited_indices(labels, 2, selection="first")
    random = select_class_limited_indices(labels, 2, selection="random", seed=0)

    assert first.tolist() == [0, 1, 2, 3]
    assert len(random) == 4
    assert np.all(random < len(labels))
