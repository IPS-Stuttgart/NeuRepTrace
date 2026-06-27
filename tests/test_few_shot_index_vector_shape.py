from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.few_shot import (
    FewShotTargetCalibrationSplit,
    fit_few_shot_target_calibrated_decoder,
    select_few_shot_target_calibration_split,
)


def test_select_few_shot_target_calibration_split_rejects_matrix_indices() -> None:
    labels = np.asarray([0, 0, 1, 1])

    with pytest.raises(ValueError, match="target_indices must be one-dimensional"):
        select_few_shot_target_calibration_split(
            labels,
            target_indices=np.asarray([[0, 1], [2, 3]]),
            per_class=1,
        )


def test_fit_few_shot_target_calibrated_decoder_rejects_matrix_split_indices() -> None:
    features = np.eye(4, dtype=float)
    labels = np.asarray([0, 0, 1, 1])
    split = FewShotTargetCalibrationSplit(
        calibration_indices=np.asarray([[0, 2]]),
        evaluation_indices=np.asarray([1, 3]),
    )

    with pytest.raises(ValueError, match="calibration_indices must be one-dimensional"):
        fit_few_shot_target_calibrated_decoder(
            source_features=features,
            source_labels=labels,
            target_features=features,
            target_labels=labels,
            classes=np.asarray([0, 1]),
            split=split,
        )
