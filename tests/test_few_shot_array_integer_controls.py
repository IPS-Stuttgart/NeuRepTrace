import numpy as np
import pytest

from neureptrace.decoding.few_shot import (
    FewShotTargetCalibrationSplit,
    fit_few_shot_target_calibrated_decoder,
    select_few_shot_target_calibration_split,
)


class DummyDecoder:
    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        return self

    def predict_proba(self, features):
        probabilities = np.ones((features.shape[0], len(self.classes_)), dtype=float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"per_class": np.asarray(1)},
            "few_shot_target_calibration_per_class.*array",
        ),
        (
            {"per_class": np.asarray([1])},
            "few_shot_target_calibration_per_class.*array",
        ),
        (
            {"min_evaluation_per_class": np.asarray([1])},
            "few_shot_min_evaluation_per_class.*array",
        ),
        (
            {"seed": np.asarray([13])},
            "few_shot_target_calibration_seed.*array",
        ),
    ],
)
def test_select_few_shot_target_calibration_split_rejects_array_integer_controls(kwargs, match):
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match=match):
        select_few_shot_target_calibration_split(labels, **kwargs)


def test_fit_few_shot_target_calibrated_decoder_rejects_array_target_repeats(monkeypatch):
    monkeypatch.setattr(
        "neureptrace.decoding.few_shot.make_decoder",
        lambda *args, **kwargs: DummyDecoder(),
    )

    with pytest.raises(ValueError, match="few_shot_target_repeats.*array"):
        fit_few_shot_target_calibrated_decoder(
            source_features=np.array([[0.0], [1.0]]),
            source_labels=np.array([0, 1]),
            target_features=np.array([[0.0], [0.1], [1.0], [0.9]]),
            target_labels=np.array([0, 0, 1, 1]),
            split=FewShotTargetCalibrationSplit(
                calibration_indices=np.array([0, 2]),
                evaluation_indices=np.array([1, 3]),
            ),
            target_repeats=np.asarray([2]),
            emission_mode="uncalibrated",
        )
