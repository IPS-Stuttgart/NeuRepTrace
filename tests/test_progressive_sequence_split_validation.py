import numpy as np
import pytest

from neureptrace.decoding.progressive_sequence_finetune import (
    NestedTrialCalibrationSplit,
    fit_progressive_sequence_target_calibrated_decoder,
)


def _fit_with_split(split: NestedTrialCalibrationSplit):
    source_features = np.zeros((1, 2, 1), dtype=np.float32)
    source_labels = np.array([[0, 1]])
    target_features = np.zeros((3, 2, 1), dtype=np.float32)
    target_labels = np.array([[0, 1], [1, 0], [0, 1]])
    return fit_progressive_sequence_target_calibrated_decoder(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_labels=target_labels,
        split=split,
    )


@pytest.mark.parametrize(
    ("calibration_indices", "evaluation_indices", "message"),
    [
        ([True], [2], "calibration_indices values must be an integer"),
        ([0.5], [2], "calibration_indices values must be an integer"),
        ([0, 0], [2], "calibration_indices must not contain duplicate"),
        ([0], [1, 1], "evaluation_indices must not contain duplicate"),
    ],
)
def test_progressive_sequence_helper_rejects_lossy_or_repeated_split_indices(
    calibration_indices,
    evaluation_indices,
    message,
):
    split = NestedTrialCalibrationSplit(
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        calibration_pool_indices=np.array([0]),
        per_stratum=1,
        max_per_stratum=1,
        seed=13,
    )

    with pytest.raises(ValueError, match=message):
        _fit_with_split(split)
