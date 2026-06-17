from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401 - installs runtime patches
from neureptrace import mne_time_decode
from neureptrace.decoding.source_alignment import TARGET_CALIBRATED_ALIGNMENT, source_alignment_config


def test_target_calibration_split_preserves_composite_anchor_rows() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    test_idx = np.arange(labels.size)
    anchors = np.array(
        [
            ["run-01", "stim-a"],
            ["run-01", "stim-a"],
            ["run-01", "stim-b"],
            ["run-01", "stim-b"],
            ["run-02", "stim-a"],
            ["run-02", "stim-a"],
            ["run-02", "stim-b"],
            ["run-02", "stim-b"],
        ],
        dtype=object,
    )
    config = source_alignment_config(
        method="mcca",
        anchor_mode="stimulus_id_repetition",
        target_projection=TARGET_CALIBRATED_ALIGNMENT,
        target_calibration_per_anchor=1,
    )

    split = mne_time_decode._alignment_target_calibration_split(
        labels=labels,
        test_idx=test_idx,
        alignment_config=config,
        anchor_values=anchors,
        seed=13,
        context=("unit", "composite-anchor"),
    )

    def selected_anchor_tuples(indices: np.ndarray) -> list[tuple[object, ...]]:
        return [tuple(row.tolist()) for row in anchors[indices]]

    expected = {
        ("run-01", "stim-a"),
        ("run-01", "stim-b"),
        ("run-02", "stim-a"),
        ("run-02", "stim-b"),
    }
    calibration_anchors = selected_anchor_tuples(split.calibration_indices)
    evaluation_anchors = selected_anchor_tuples(split.evaluation_indices)

    assert split.calibration_indices.size == 4
    assert split.evaluation_indices.size == 4
    assert set(calibration_anchors) == expected
    assert set(evaluation_anchors) == expected
    assert all(calibration_anchors.count(anchor) == 1 for anchor in expected)
    assert all(evaluation_anchors.count(anchor) == 1 for anchor in expected)
