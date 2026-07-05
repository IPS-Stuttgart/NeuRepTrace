from __future__ import annotations

import numpy as np

from neureptrace._correlation_prototype_sample_weight_patch import _zero_weight_classes


def test_zero_weight_classes_preserves_tuple_labels_as_atomic_classes() -> None:
    labels = [
        ("visual", "left"),
        ("visual", "left"),
        ("motor", "right"),
    ]
    sample_weight = [1.0, 0.0, 0.0]

    assert _zero_weight_classes(labels, sample_weight) == [("motor", "right")]


def test_zero_weight_classes_preserves_list_labels_as_atomic_classes() -> None:
    labels = [
        ["visual", "left"],
        ["motor", "right"],
        ["visual", "left"],
    ]
    sample_weight = [1.0, 0.0, 0.0]

    assert _zero_weight_classes(labels, sample_weight) == [("motor", "right")]


def test_zero_weight_classes_preserves_matrix_rows_as_atomic_labels() -> None:
    labels = np.asarray(
        [
            ["visual", "left"],
            ["motor", "right"],
            ["visual", "left"],
        ],
        dtype=object,
    )
    sample_weight = np.asarray([1.0, 0.0, 0.0])

    assert _zero_weight_classes(labels, sample_weight) == [("motor", "right")]
