from __future__ import annotations

import numpy as np

from neureptrace.decoding import select_class_limited_indices


def _numeric_composite_labels() -> np.ndarray:
    return np.asarray(
        [
            (1, 10),
            (2, 20),
            (1, 10),
            (2, 20),
            (1, 10),
            (2, 20),
        ],
        dtype=int,
    )


def test_class_limiter_keeps_numeric_composite_rows_atomic_when_uncapped() -> None:
    labels = _numeric_composite_labels()

    selected = select_class_limited_indices(labels, None)

    assert selected.tolist() == list(range(labels.shape[0]))


def test_class_limiter_keeps_numeric_composite_rows_atomic_when_capped() -> None:
    labels = _numeric_composite_labels()

    first = select_class_limited_indices(labels, 2, selection="first")
    random = select_class_limited_indices(labels, 2, selection="random", seed=0)

    assert first.tolist() == [0, 1, 2, 3]
    assert len(random) == 4
    assert np.all(random < labels.shape[0])


def test_class_limiter_keeps_numeric_column_vectors_as_scalar_labels() -> None:
    labels = np.asarray([[1], [2], [1], [2], [1], [2]], dtype=int)

    selected = select_class_limited_indices(labels, 2, selection="first")

    assert selected.tolist() == [0, 1, 2, 3]
