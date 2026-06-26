from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.bushmeg_all_protocols import category3_calibration_evaluation_split, select_bushmeg_target_calibration_split
import neureptrace.decoding.few_shot as few_shot


def _tuple_labels() -> list[tuple[str, str]]:
    return [
        ("run-01", "face"),
        ("run-01", "face"),
        ("run-01", "face"),
        ("run-02", "object"),
        ("run-02", "object"),
        ("run-02", "object"),
    ]


def _labels_at(labels: list[tuple[str, str]], indices: np.ndarray) -> list[tuple[str, str]]:
    return [labels[int(index)] for index in indices]


def test_all_protocols_protocol3_split_treats_tuple_labels_atomically() -> None:
    labels = _tuple_labels()
    assert np.asarray(labels, dtype=object).ndim == 2

    split = select_bushmeg_target_calibration_split(
        labels,
        per_class=1,
        seed=13,
        context=("tuple-label-target", "few-shot"),
    )

    assert split.skipped is False
    assert split.calibration_indices.size == 2
    assert split.evaluation_indices.size == 4
    calibration_labels = _labels_at(labels, split.calibration_indices)
    evaluation_labels = _labels_at(labels, split.evaluation_indices)
    for class_label in sorted(set(labels)):
        assert calibration_labels.count(class_label) == 1
        assert evaluation_labels.count(class_label) >= 1


def test_category3_split_accepts_composite_numpy_label_rows() -> None:
    labels = np.asarray(_tuple_labels(), dtype=object)

    calibration_indices, evaluation_indices = category3_calibration_evaluation_split(
        labels,
        calibration_per_class=1,
        seed=21,
    )

    assert calibration_indices.size == 2
    assert evaluation_indices.size == 4
    assert np.intersect1d(calibration_indices, evaluation_indices).size == 0


def test_few_shot_split_treats_tuple_labels_atomically() -> None:
    labels = _tuple_labels()
    assert np.asarray(labels, dtype=object).ndim == 2

    split = few_shot.select_few_shot_target_calibration_split(
        labels,
        per_class=1,
        seed=13,
        context=("tuple-label-target", "few-shot"),
    )

    assert split.calibration_indices.size == 2
    assert split.evaluation_indices.size == 4
    calibration_labels = _labels_at(labels, split.calibration_indices)
    evaluation_labels = _labels_at(labels, split.evaluation_indices)
    for class_label in sorted(set(labels)):
        assert calibration_labels.count(class_label) == 1
        assert evaluation_labels.count(class_label) >= 1


def test_few_shot_tuple_label_patch_preserves_duplicate_index_guard() -> None:
    with pytest.raises(ValueError, match="duplicate target row indices"):
        few_shot.select_few_shot_target_calibration_split(_tuple_labels(), target_indices=[0, 0, 1], per_class=1)


def test_few_shot_probability_alignment_accepts_tuple_class_order() -> None:
    class Model:
        pass

    model = Model()
    model.classes_ = np.empty(2, dtype=object)
    model.classes_[:] = [("run-01", "face"), ("run-02", "object")]

    aligned = few_shot._align_probability_columns(
        np.asarray([[0.25, 0.75]]),
        model=model,
        classes=[("run-02", "object"), ("run-01", "face")],
    )

    assert np.allclose(aligned, [[0.75, 0.25]])
