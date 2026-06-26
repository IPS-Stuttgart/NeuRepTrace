import numpy as np
import pytest

from neureptrace.decoding.few_shot import (
    FewShotTargetCalibrationSplit,
    fit_few_shot_target_calibrated_decoder,
    select_few_shot_target_calibration_split,
)


class RecordingDecoder:
    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        return self

    def predict_proba(self, features):
        probabilities = np.ones((features.shape[0], len(self.classes_)), dtype=float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    "target_indices",
    [
        np.array([0, 1.5, 3], dtype=object),
        np.array([False, 1, 3], dtype=object),
        np.array([0, np.nan, 3], dtype=object),
    ],
)
def test_select_few_shot_target_calibration_split_rejects_lossy_target_indices(target_indices):
    labels = np.array([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="target_indices"):
        select_few_shot_target_calibration_split(labels, target_indices=target_indices)


@pytest.mark.parametrize(
    ("split", "message"),
    [
        (
            FewShotTargetCalibrationSplit(
                calibration_indices=np.array([0.5, 3], dtype=object),
                evaluation_indices=np.array([1, 2, 4, 5]),
            ),
            "calibration_indices",
        ),
        (
            FewShotTargetCalibrationSplit(
                calibration_indices=np.array([0, 3]),
                evaluation_indices=np.array([1, 2, 4.5, 5], dtype=object),
            ),
            "evaluation_indices",
        ),
        (
            FewShotTargetCalibrationSplit(
                calibration_indices=np.array([True, 3], dtype=object),
                evaluation_indices=np.array([1, 2, 4, 5]),
            ),
            "calibration_indices",
        ),
    ],
)
def test_fit_few_shot_target_calibrated_decoder_rejects_lossy_split_indices(monkeypatch, split, message):
    monkeypatch.setattr("neureptrace.decoding.few_shot.make_decoder", lambda *args, **kwargs: RecordingDecoder())

    with pytest.raises(ValueError, match=message):
        fit_few_shot_target_calibrated_decoder(
            source_features=np.array([[0.0], [0.1], [1.0], [0.9]]),
            source_labels=np.array([0, 0, 1, 1]),
            target_features=np.array([[0.0], [0.1], [0.2], [1.0], [0.9], [0.8]]),
            target_labels=np.array([0, 0, 0, 1, 1, 1]),
            classes=np.array([0, 1]),
            split=split,
            emission_mode="uncalibrated",
        )
