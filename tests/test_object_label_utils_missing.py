from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace._object_label_utils import label_accuracy, label_counts, label_equal_mask, values_equal


def test_values_equal_matches_pandas_missing_scalar_and_composite_labels() -> None:
    assert values_equal(pd.NA, pd.NA)
    assert values_equal((pd.NA, "cue"), (pd.NA, "cue"))
    assert values_equal(
        np.asarray([pd.NA, "cue"], dtype=object),
        np.asarray([pd.NA, "cue"], dtype=object),
    )
    assert not values_equal((pd.NA, "cue"), (pd.NA, "other"))


def test_label_counts_collapses_repeated_pandas_missing_labels() -> None:
    labels = np.empty(5, dtype=object)
    labels[:] = [pd.NA, "seen", pd.NA, (pd.NA, "cue"), (pd.NA, "cue")]

    unique, counts = label_counts(labels)

    assert pd.isna(unique[0])
    assert unique[1] == "seen"
    assert pd.isna(unique[2][0])
    assert unique[2][1] == "cue"
    np.testing.assert_array_equal(counts, np.asarray([2, 1, 2]))


def test_label_equal_mask_and_accuracy_accept_pandas_missing_labels() -> None:
    labels = np.empty(4, dtype=object)
    labels[:] = [pd.NA, "seen", (pd.NA, "cue"), (pd.NA, "other")]

    np.testing.assert_array_equal(label_equal_mask(labels, pd.NA), np.asarray([True, False, False, False]))
    np.testing.assert_array_equal(
        label_equal_mask(labels, (pd.NA, "cue")),
        np.asarray([False, False, True, False]),
    )
    assert label_accuracy(labels, [pd.NA, "seen", (pd.NA, "cue"), (pd.NA, "cue")]) == 0.75
