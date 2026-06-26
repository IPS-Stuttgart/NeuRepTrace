from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.few_shot import (
    FewShotTargetCalibrationSplit,
    fit_few_shot_target_calibrated_decoder,
    select_few_shot_target_calibration_split,
)


def test_select_few_shot_target_calibration_split_rejects_duplicate_target_indices() -> None:
    labels = np.asarray([0, 0, 1, 1])

    with pytest.raises(ValueError, match="target_indices.*duplicate"):
        select_few_shot_target_calibration_split(
            labels,
            target_indices=np.asarray([0, 0, 2, 3]),
            per_class=1,
        )


@pytest.mark.parametrize(
    ("split", "match"),
    [
        (
            FewShotTargetCalibrationSplit(
                calibration_indices=np.asarray([0, 0]),
                evaluation_indices=np.asarray([1, 3]),
            ),
            "calibration_indices.*duplicate",
        ),
        (
            FewShotTargetCalibrationSplit(
                calibration_indices=np.asarray([0, 2]),
                evaluation_indices=np.asarray([1, 1]),
            ),
            "evaluation_indices.*duplicate",
        ),
    ],
)
def test_fit_few_shot_target_calibrated_decoder_rejects_duplicate_manual_split_indices(
    split: FewShotTargetCalibrationSplit,
    match: str,
) -> None:
    features = np.eye(4, dtype=float)
    labels = np.asarray([0, 0, 1, 1])

    with pytest.raises(ValueError, match=match):
        fit_few_shot_target_calibrated_decoder(
            source_features=features,
            source_labels=labels,
            target_features=features,
            target_labels=labels,
            classes=np.asarray([0, 1]),
            split=split,
            emission_mode="uncalibrated",
        )
