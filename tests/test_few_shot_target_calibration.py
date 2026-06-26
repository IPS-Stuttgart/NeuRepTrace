import numpy as np
import pytest

from neureptrace.decoding.few_shot import (
    FEW_SHOT_TARGET_CALIBRATION_CATEGORY,
    FEW_SHOT_TARGET_CALIBRATION_PROTOCOL,
    FewShotTargetCalibrationSplit,
    fit_few_shot_target_calibrated_decoder,
    select_few_shot_target_calibration_split,
)


class RecordingDecoder:
    fit_features = []
    fit_labels = []

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        self.fit_features.append(np.asarray(features, dtype=float).copy())
        self.fit_labels.append(np.asarray(labels, dtype=int).copy())
        return self

    def predict_proba(self, features):
        probabilities = np.full((features.shape[0], len(self.classes_)), 0.5)
        if len(self.classes_) == 2:
            probabilities[:, 0] = np.where(features[:, 0] < 0.5, 0.8, 0.2)
            probabilities[:, 1] = 1.0 - probabilities[:, 0]
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def test_select_few_shot_target_calibration_split_is_balanced_and_disjoint():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    split_a = select_few_shot_target_calibration_split(labels, per_class=1, seed=7, context=("fold", 1))
    split_b = select_few_shot_target_calibration_split(labels, per_class=1, seed=7, context=("fold", 1))

    np.testing.assert_array_equal(split_a.calibration_indices, split_b.calibration_indices)
    np.testing.assert_array_equal(split_a.evaluation_indices, split_b.evaluation_indices)
    assert np.intersect1d(split_a.calibration_indices, split_a.evaluation_indices).size == 0
    assert split_a.calibration_indices.size == 3
    assert split_a.evaluation_indices.size == 6
    assert {int(label): int(np.count_nonzero(labels[split_a.calibration_indices] == label)) for label in np.unique(labels)} == {0: 1, 1: 1, 2: 1}
    assert {int(label): int(np.count_nonzero(labels[split_a.evaluation_indices] == label)) for label in np.unique(labels)} == {0: 2, 1: 2, 2: 2}


def test_select_few_shot_target_calibration_split_rejects_invalid_target_indices():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    with pytest.raises(ValueError, match="target_indices.*boolean"):
        select_few_shot_target_calibration_split(labels, target_indices=np.array([True, False, True, False, True, False, True, False, True]))

    with pytest.raises(ValueError, match="target_indices.*integer row indices"):
        select_few_shot_target_calibration_split(labels, target_indices=[0.0, 1.5, 3.0, 4.0, 6.0, 7.0])


@pytest.mark.parametrize(
    ("split", "match"),
    [
        (
            FewShotTargetCalibrationSplit(
                calibration_indices=np.array([True, False, False, False]),
                evaluation_indices=np.array([1, 2]),
            ),
            "calibration_indices.*boolean",
        ),
        (
            FewShotTargetCalibrationSplit(
                calibration_indices=np.array([0]),
                evaluation_indices=np.array([1.5, 2.0]),
            ),
            "evaluation_indices.*integer row indices",
        ),
    ],
)
def test_fit_few_shot_target_calibrated_decoder_rejects_invalid_manual_split_indices(split, match):
    with pytest.raises(ValueError, match=match):
        fit_few_shot_target_calibrated_decoder(
            source_features=np.array([[0.0], [1.0]]),
            source_labels=np.array([0, 1]),
            target_features=np.array([[0.0], [0.1], [1.0], [0.9]]),
            target_labels=np.array([0, 0, 1, 1]),
            split=split,
            emission_mode="uncalibrated",
        )


def test_select_few_shot_target_calibration_split_rejects_if_no_evaluation_rows_remain():
    labels = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="needs at least 3 target rows"):
        select_few_shot_target_calibration_split(labels, per_class=2, seed=13)


def test_fit_few_shot_target_calibrated_decoder_uses_labeled_target_calibration_rows(monkeypatch):
    source_features = np.array([[0.0, 0.1], [0.1, 0.2], [1.0, 0.2], [0.9, 0.3]])
    source_labels = np.array([0, 0, 1, 1])
    target_features = np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [1.0, 1.0], [0.9, 1.1], [0.8, 1.2]])
    target_labels = np.array([0, 0, 0, 1, 1, 1])
    split = FewShotTargetCalibrationSplit(calibration_indices=np.array([0, 3]), evaluation_indices=np.array([1, 2, 4, 5]))
    RecordingDecoder.fit_features = []
    RecordingDecoder.fit_labels = []
    monkeypatch.setattr("neureptrace.decoding.few_shot.make_decoder", lambda *args, **kwargs: RecordingDecoder())

    result = fit_few_shot_target_calibrated_decoder(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_labels=target_labels,
        classes=np.array([0, 1]),
        split=split,
        per_class=1,
        seed=17,
        emission_mode="uncalibrated",
    )

    assert result.probabilities.shape == (4, 2)
    np.testing.assert_array_equal(result.calibration_indices, np.array([0, 3]))
    np.testing.assert_array_equal(result.evaluation_indices, np.array([1, 2, 4, 5]))
    np.testing.assert_array_equal(RecordingDecoder.fit_labels[0], np.array([0, 0, 1, 1, 0, 1]))
    np.testing.assert_allclose(RecordingDecoder.fit_features[0][-2:], target_features[[0, 3]])
    assert result.metadata["few_shot_protocol"] == FEW_SHOT_TARGET_CALIBRATION_PROTOCOL
    assert result.metadata["few_shot_protocol_category"] == FEW_SHOT_TARGET_CALIBRATION_CATEGORY
    assert result.metadata["few_shot_uses_target_features"] is True
    assert result.metadata["few_shot_uses_target_labels"] is True
    assert result.metadata["few_shot_valid_for_strict_source_only"] is False
    assert result.metadata["few_shot_n_target_calibration_rows"] == 2
    assert result.metadata["few_shot_n_target_evaluation_rows"] == 4


def test_fit_few_shot_target_calibrated_decoder_can_repeat_target_calibration_rows(monkeypatch):
    source_features = np.array([[0.0], [1.0]])
    source_labels = np.array([0, 1])
    target_features = np.array([[0.0], [0.1], [1.0], [0.9]])
    target_labels = np.array([0, 0, 1, 1])
    RecordingDecoder.fit_features = []
    RecordingDecoder.fit_labels = []
    monkeypatch.setattr("neureptrace.decoding.few_shot.make_decoder", lambda *args, **kwargs: RecordingDecoder())

    result = fit_few_shot_target_calibrated_decoder(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_labels=target_labels,
        per_class=1,
        target_repeats=2,
        seed=3,
        emission_mode="uncalibrated",
    )

    assert len(RecordingDecoder.fit_labels[0]) == 6
    assert result.metadata["few_shot_target_repeats"] == 2
    assert result.metadata["few_shot_n_target_calibration_rows"] == 2
    assert result.metadata["few_shot_n_target_evaluation_rows"] == 2
