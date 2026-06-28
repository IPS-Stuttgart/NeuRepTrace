from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_all_protocols import select_bushmeg_target_calibration_split


def _count_matches(values: np.ndarray, class_value: object) -> int:
    return sum(1 for value in values if value == class_value)


def test_tuple_label_calibration_split_patch_installed_for_mixed_composite_labels() -> None:
    labels = np.empty(6, dtype=object)
    labels[:] = ["face", 1, "face", 1, ("run-1", "cat"), ("run-1", "cat")]

    split = select_bushmeg_target_calibration_split(
        labels,
        per_class=1,
        seed=17,
        context=("mixed-labels",),
    )

    assert split.skipped is False
    assert split.n_classes == 3
    assert split.calibration_indices.size == 3
    assert split.evaluation_indices.size == 3
    for class_value in ("face", 1, ("run-1", "cat")):
        assert _count_matches(labels[split.calibration_indices], class_value) == 1
        assert _count_matches(labels[split.evaluation_indices], class_value) == 1
