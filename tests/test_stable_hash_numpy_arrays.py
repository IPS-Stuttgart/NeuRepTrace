from __future__ import annotations

import numpy as np

from neureptrace.observations import stable_hash


def test_stable_hash_distinguishes_large_arrays_with_hidden_differences() -> None:
    first = np.arange(2000, dtype=np.int64)
    second = first.copy()
    second[len(second) // 2] = -999

    with np.printoptions(threshold=1000):
        assert str(first) == str(second)

    assert stable_hash({"weights": first}, length=64) != stable_hash(
        {"weights": second},
        length=64,
    )


def test_stable_hash_numpy_array_is_independent_of_print_options() -> None:
    values = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
    expected = stable_hash({"values": values}, length=64)

    with np.printoptions(threshold=4, edgeitems=1, linewidth=12, precision=2):
        actual = stable_hash({"values": values}, length=64)

    assert actual == expected


def test_stable_hash_numpy_array_tracks_shape_dtype_and_values() -> None:
    matrix = np.arange(12, dtype=np.float64).reshape(3, 4)

    assert stable_hash(matrix, length=64) == stable_hash(
        np.asfortranarray(matrix),
        length=64,
    )
    assert stable_hash(np.asarray([1, 2], dtype=np.int32), length=64) != stable_hash(
        np.asarray([1, 2], dtype=np.int64),
        length=64,
    )
    assert stable_hash(np.asarray([1, 2]), length=64) != stable_hash(
        np.asarray([[1, 2]]),
        length=64,
    )


def test_stable_hash_numpy_object_arrays_use_semantic_values() -> None:
    first = np.asarray([{"classes": {"left", "right"}}], dtype=object)
    second = np.asarray([{"classes": {"right", "left"}}], dtype=object)

    assert stable_hash(first, length=64) == stable_hash(second, length=64)
