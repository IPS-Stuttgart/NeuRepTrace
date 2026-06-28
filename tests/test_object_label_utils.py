from __future__ import annotations

import numpy as np

from neureptrace._object_label_utils import label_counts, label_equal_mask, replace_null_class_predictions, values_equal


def test_values_equal_accepts_numpy_array_labels() -> None:
    assert values_equal(np.array(["left", 2]), np.array(["left", 2]))
    assert not values_equal(np.array(["left", 2]), np.array(["left", 3]))
    assert not values_equal(np.array(["left", 2]), np.array([["left", 2]]))


def test_values_equal_treats_nan_labels_as_matching() -> None:
    assert values_equal(float("nan"), np.nan)
    assert values_equal(np.array(["trial", np.nan], dtype=object), np.array(["trial", np.nan], dtype=object))
    assert not values_equal(np.array(["trial", np.nan], dtype=object), np.array(["trial", 1.0], dtype=object))


def test_label_helpers_match_and_count_nan_labels() -> None:
    labels = np.array([np.nan, 1.0, np.nan], dtype=object)

    assert label_equal_mask(labels, np.nan).tolist() == [True, False, True]

    unique, counts = label_counts(labels)
    assert counts.tolist() == [2, 1]
    assert values_equal(unique[0], np.nan)
    assert values_equal(unique[1], 1.0)


def test_label_helpers_match_and_count_numpy_array_labels() -> None:
    labels = [np.array(["subject-a", 1]), np.array(["subject-a", 1]), np.array(["subject-b", 2])]

    assert label_equal_mask(labels, np.array(["subject-a", 1])).tolist() == [True, True, False]

    unique, counts = label_counts(labels)
    assert counts.tolist() == [2, 1]
    assert values_equal(unique[0], np.array(["subject-a", 1]))
    assert values_equal(unique[1], np.array(["subject-b", 2]))


def test_replace_null_class_predictions_promotes_unrepresentable_fallback_label() -> None:
    repaired = replace_null_class_predictions(np.asarray([0, 0]), null_label=0, fallback_label="target")

    assert repaired.dtype == object
    assert repaired.tolist() == ["target", "target"]


def test_replace_null_class_predictions_preserves_tuple_label_atoms() -> None:
    repaired = replace_null_class_predictions([0, ("subject-a", 1), ("subject-b", 2), 0], null_label=0)

    assert repaired.dtype == object
    assert repaired.tolist() == [("subject-a", 1), ("subject-a", 1), ("subject-b", 2), ("subject-a", 1)]
