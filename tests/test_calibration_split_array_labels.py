from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401  # importing the package installs runtime patches
from neureptrace import _tuple_label_calibration_split_patch as calibration_split_patch
from neureptrace.bushmeg_all_protocols import select_bushmeg_target_calibration_split
from neureptrace.decoding.few_shot import select_few_shot_target_calibration_split


def _array_label_vector(rows: list[list[int]]) -> np.ndarray:
    labels = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        labels[index] = np.asarray(row, dtype=int)
    return labels


def _assert_contains_array_label(labels: np.ndarray, expected: list[int]) -> None:
    expected_array = np.asarray(expected, dtype=int)
    assert any(np.array_equal(label, expected_array) for label in labels)


def _assert_one_row_per_array_class(labels: np.ndarray, indices: np.ndarray) -> None:
    selected = labels[indices]
    assert selected.shape[0] == 2
    _assert_contains_array_label(selected, [1, 0])
    _assert_contains_array_label(selected, [0, 1])


def test_calibration_split_equality_handles_array_valued_labels() -> None:
    assert calibration_split_patch._values_equal(np.asarray([1, 0]), np.asarray([1, 0]))
    assert not calibration_split_patch._values_equal(np.asarray([1, 0]), np.asarray([0, 1]))


def test_protocol3_calibration_split_groups_repeated_array_valued_labels() -> None:
    labels = _array_label_vector([[1, 0], [1, 0], [0, 1], [0, 1]])

    split = select_bushmeg_target_calibration_split(
        labels,
        per_class=1,
        seed=17,
        min_evaluation_per_class=1,
        context=("array-valued-labels",),
    )

    assert split.skipped is False, split.skip_reason
    assert split.n_classes == 2
    _assert_one_row_per_array_class(labels, split.calibration_indices)
    _assert_one_row_per_array_class(labels, split.evaluation_indices)
    assert np.intersect1d(split.calibration_indices, split.evaluation_indices).size == 0


def test_few_shot_calibration_split_groups_repeated_array_valued_labels() -> None:
    labels = _array_label_vector([[1, 0], [1, 0], [0, 1], [0, 1]])

    split = select_few_shot_target_calibration_split(
        labels,
        per_class=1,
        seed=17,
        min_evaluation_per_class=1,
        context=("array-valued-labels",),
    )

    _assert_one_row_per_array_class(labels, split.calibration_indices)
    _assert_one_row_per_array_class(labels, split.evaluation_indices)
    assert np.intersect1d(split.calibration_indices, split.evaluation_indices).size == 0
