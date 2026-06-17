"""Patch target-calibration split handling for composite alignment anchors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mne_alignment_calibration_anchor_patch_installed"


def _object_value_vector(values: Iterable[object]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _object_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return _object_value_vector([array.item()])
    if array.ndim == 1:
        return array.reshape(-1)
    rows = [tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)]
    return _object_value_vector(rows)


def _values_equal(left: object, right: object) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _value_mask(values: Sequence[Any] | np.ndarray, target: object) -> np.ndarray:
    return np.asarray([_values_equal(value, target) for value in _object_vector(values)], dtype=bool)


def _ordered_unique(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    unique: list[object] = []
    for value in _object_vector(values):
        if not any(_values_equal(existing, value) for existing in unique):
            unique.append(value)
    return _object_value_vector(unique)


def install() -> None:
    from neureptrace import mne_time_decode

    if getattr(mne_time_decode, _PATCH_MARKER, False):
        return

    def _alignment_target_calibration_split(
        *,
        labels: np.ndarray,
        test_idx: np.ndarray,
        alignment_config,
        anchor_values: np.ndarray | None,
        seed: int,
        context: Sequence[object],
    ):
        test_idx_array = np.asarray(test_idx, dtype=int)
        if not alignment_config.target_calibrated:
            return mne_time_decode.AlignmentTargetCalibrationSplit(
                evaluation_indices=test_idx_array,
                calibration_indices=np.asarray([], dtype=int),
            )
        if test_idx_array.size == 0:
            raise ValueError("target_calibrated_alignment requires at least one held-out target row.")

        if anchor_values is None:
            target_values = _object_vector(np.asarray(labels)[test_idx_array])
        else:
            target_values = _object_vector(np.asarray(anchor_values, dtype=object)[test_idx_array])
        if target_values.shape[0] != test_idx_array.shape[0]:
            raise ValueError("target calibration anchors must have one value per held-out target row.")

        per_anchor = int(alignment_config.target_calibration_per_anchor)
        calibration_mask = np.zeros(test_idx_array.shape[0], dtype=bool)
        for anchor_position, anchor in enumerate(_ordered_unique(target_values)):
            positions = np.flatnonzero(_value_mask(target_values, anchor))
            if positions.size <= per_anchor:
                raise ValueError(
                    "target_calibrated_alignment needs at least "
                    f"{per_anchor + 1} held-out target rows for anchor {anchor!r} so calibration rows are disjoint from scored rows."
                )
            seed_value = int(
                mne_time_decode.stable_hash(
                    {
                        "seed": seed,
                        "context": tuple(context),
                        "anchor_position": anchor_position,
                        "anchor": anchor,
                    }
                ),
                16,
            ) % (2**32)
            rng = np.random.default_rng(seed_value)
            calibration_mask[rng.choice(positions, size=per_anchor, replace=False)] = True

        evaluation_indices = test_idx_array[~calibration_mask]
        calibration_indices = test_idx_array[calibration_mask]
        if evaluation_indices.size == 0:
            raise ValueError("target_calibrated_alignment left no target rows for scoring.")
        missing_eval_classes = [
            class_label
            for class_label in np.unique(labels[test_idx_array])
            if not np.any(labels[evaluation_indices] == class_label)
        ]
        if missing_eval_classes:
            raise ValueError(
                "target_calibrated_alignment removed all scored rows for target classes: "
                f"{missing_eval_classes!r}. Reduce alignment_target_calibration_per_anchor or use more target trials."
            )
        return mne_time_decode.AlignmentTargetCalibrationSplit(
            evaluation_indices=evaluation_indices,
            calibration_indices=calibration_indices,
        )

    mne_time_decode._alignment_target_calibration_split = _alignment_target_calibration_split
    setattr(mne_time_decode, _PATCH_MARKER, True)
