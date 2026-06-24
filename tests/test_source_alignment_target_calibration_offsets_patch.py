import numpy as np

from neureptrace.decoding import source_alignment


def _toy_target_alignment_inputs():
    features = np.arange(8, dtype=float).reshape(4, 2)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    classes = np.asarray(["left", "right"], dtype=object)
    stale_source_offsets = {
        0: np.asarray([4, 5], dtype=int),
        1: np.asarray([3, 4], dtype=int),
    }
    return features, labels, classes, stale_source_offsets


def test_calibrated_repetition_projection_offsets_are_local_to_target_calibration_rows():
    config = source_alignment.source_alignment_config(
        method="contrastive",
        anchor_mode="class_repetition",
        repetition_cap=2,
        target_projection=source_alignment.TARGET_CALIBRATED_ALIGNMENT,
        target_calibration_per_anchor=2,
    )
    features, labels, classes, stale_source_offsets = _toy_target_alignment_inputs()

    # These are valid source-fit offsets in a larger source subject, but they are
    # stale for the separately prepared two-repetition-per-class target
    # calibration matrix.  Calibrated target projections must rebase offsets to
    # the calibration subset instead of reusing source-subject offsets.
    target_anchors = source_alignment._target_alignment_matrix(
        features,
        labels,
        classes=classes,
        config=config,
        n_repetitions_per_class=2,
        selected_offsets_by_class=stale_source_offsets,
    )

    np.testing.assert_array_equal(target_anchors, features)


def test_oracle_repetition_projection_offsets_are_local_to_target_rows():
    config = source_alignment.source_alignment_config(
        method="procrustes",
        anchor_mode="class_repetition",
        repetition_cap=2,
        target_projection=source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    )
    features, labels, classes, stale_source_offsets = _toy_target_alignment_inputs()

    # Oracle projections fit from held-out target rows, but source-fit offsets are
    # still source-subject-local.  They must be rebased to the target row order
    # before building the target projection matrix.
    target_anchors = source_alignment._target_alignment_matrix(
        features,
        labels,
        classes=classes,
        config=config,
        n_repetitions_per_class=2,
        selected_offsets_by_class=stale_source_offsets,
    )

    np.testing.assert_array_equal(target_anchors, features)
