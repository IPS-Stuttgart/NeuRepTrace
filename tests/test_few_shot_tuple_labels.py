import numpy as np

from neureptrace.decoding.few_shot import (
    FewShotTargetCalibrationSplit,
    fit_few_shot_target_calibrated_decoder,
    select_few_shot_target_calibration_split,
)


class ReversedTupleDecoder:
    def fit(self, features, labels):
        self.classes_ = np.empty(2, dtype=object)
        self.classes_[:] = [("house", "late"), ("face", "early")]
        return self

    def predict_proba(self, features):
        probabilities = np.empty((features.shape[0], 2), dtype=float)
        probabilities[:, 0] = 0.7  # ("house", "late") in model.classes_
        probabilities[:, 1] = 0.3  # ("face", "early") in model.classes_
        return probabilities


def test_select_few_shot_target_calibration_split_handles_tuple_labels():
    labels = [
        ("face", "early"),
        ("face", "early"),
        ("face", "early"),
        ("house", "late"),
        ("house", "late"),
        ("house", "late"),
    ]

    split = select_few_shot_target_calibration_split(labels, per_class=1, seed=11)

    calibration_labels = [labels[index] for index in split.calibration_indices.tolist()]
    evaluation_labels = [labels[index] for index in split.evaluation_indices.tolist()]
    assert calibration_labels.count(("face", "early")) == 1
    assert calibration_labels.count(("house", "late")) == 1
    assert evaluation_labels.count(("face", "early")) == 2
    assert evaluation_labels.count(("house", "late")) == 2


def test_fit_few_shot_target_calibrated_decoder_aligns_tuple_label_columns(monkeypatch):
    source_features = np.array([[0.0], [1.0]])
    source_labels = [("face", "early"), ("house", "late")]
    target_features = np.array([[0.0], [0.1], [1.0], [0.9]])
    target_labels = [("face", "early"), ("face", "early"), ("house", "late"), ("house", "late")]
    split = FewShotTargetCalibrationSplit(
        calibration_indices=np.array([0, 2]),
        evaluation_indices=np.array([1, 3]),
    )
    monkeypatch.setattr("neureptrace.decoding.few_shot.make_decoder", lambda *args, **kwargs: ReversedTupleDecoder())

    result = fit_few_shot_target_calibrated_decoder(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_labels=target_labels,
        classes=[("face", "early"), ("house", "late")],
        split=split,
        emission_mode="uncalibrated",
    )

    np.testing.assert_allclose(result.probabilities, np.array([[0.3, 0.7], [0.3, 0.7]]))
