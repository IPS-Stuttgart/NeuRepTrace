from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import select_class_limited_indices


def _parts(*values: str):
    return (value for value in values)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("first", [0, 2]),
        ("random", [1, 3]),
    ],
)
def test_class_limit_groups_equal_generator_backed_labels(
    selection: str,
    expected: list[int],
) -> None:
    labels = [
        _parts("face", "early"),
        _parts("face", "early"),
        _parts("house", "late"),
        _parts("house", "late"),
    ]

    selected = select_class_limited_indices(
        labels,
        1,
        selection=selection,
        seed=0,
    )

    assert selected.tolist() == expected


def test_class_limit_materializes_nested_generators_in_object_arrays() -> None:
    labels = np.empty(4, dtype=object)
    labels[:] = [
        ("task", _parts("face", "early")),
        ("task", _parts("face", "early")),
        ("task", _parts("house", "late")),
        ("task", _parts("house", "late")),
    ]

    selected = select_class_limited_indices(labels, 1, selection="first")

    assert selected.tolist() == [0, 2]
