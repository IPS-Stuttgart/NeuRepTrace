from __future__ import annotations

import numpy as np

from neureptrace._object_label_utils import label_counts, values_equal


def test_values_equal_matches_dict_labels_with_array_values() -> None:
    left = {"subject": "a", "signature": np.asarray([1.0, np.nan])}
    same = {"signature": np.asarray([1.0, np.nan]), "subject": "a"}
    different = {"subject": "a", "signature": np.asarray([1.0, 2.0])}

    assert values_equal(left, same)
    assert values_equal(same, left)
    assert not values_equal(left, different)


def test_label_counts_merges_equal_dict_labels_with_array_values() -> None:
    labels = [
        {"subject": "a", "signature": np.asarray([1, 2])},
        {"signature": np.asarray([1, 2]), "subject": "a"},
        {"subject": "b", "signature": np.asarray([1, 2])},
    ]

    unique, counts = label_counts(labels)

    assert counts.tolist() == [2, 1]
    assert values_equal(unique[0], labels[0])
    assert values_equal(unique[1], labels[2])
