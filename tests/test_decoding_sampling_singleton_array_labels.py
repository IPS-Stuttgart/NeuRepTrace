from __future__ import annotations

import numpy as np

from neureptrace.decoding import select_class_limited_indices


def test_class_limiter_keeps_singleton_array_labels_distinct_from_scalars() -> None:
    labels = [1, np.asarray([1]), 1, np.asarray([1])]

    first = select_class_limited_indices(labels, 1, selection="first")
    random = select_class_limited_indices(labels, 1, selection="random", seed=0)

    assert first.tolist() == [0, 1]
    assert len(random) == 2


def test_class_limiter_groups_equivalent_singleton_composite_containers() -> None:
    labels = [np.asarray([1]), [1], (1,), np.asarray([1])]

    first = select_class_limited_indices(labels, 2, selection="first")
    random = select_class_limited_indices(labels, 2, selection="random", seed=0)

    assert first.tolist() == [0, 1]
    assert len(random) == 2


def test_class_limiter_keeps_zero_dimensional_arrays_scalar() -> None:
    labels = [1, np.asarray(1), 1, np.asarray(1)]

    selected = select_class_limited_indices(labels, 1, selection="first")

    assert selected.tolist() == [0]
