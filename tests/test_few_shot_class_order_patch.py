from __future__ import annotations

import numpy as np

import neureptrace.decoding.few_shot as few_shot


def _object_vector(values):
    vector = np.empty(len(values), dtype=object)
    vector[:] = values
    return vector


class _MixedLabelDecoder:
    def fit(self, features, labels):
        self.fit_features_ = np.asarray(features, dtype=float)
        self.fit_labels_ = _object_vector(list(labels))
        self.classes_ = _object_vector(["face", 1])
        return self


def test_few_shot_infers_class_order_for_mixed_unsortable_labels(monkeypatch):
    decoder = _MixedLabelDecoder()

    def make_decoder(*_args, **_kwargs):
        return decoder

    def predict_emission_probabilities(model, features, *, emission_mode="uncalibrated"):
        del emission_mode
        assert model is decoder
        n_rows = np.asarray(features).shape[0]
        return np.tile(np.asarray([[0.75, 0.25]], dtype=float), (n_rows, 1))

    monkeypatch.setattr(few_shot, "make_decoder", make_decoder)
    monkeypatch.setattr(few_shot, "predict_emission_probabilities", predict_emission_probabilities)

    split = few_shot.FewShotTargetCalibrationSplit(
        evaluation_indices=np.asarray([1, 3], dtype=int),
        calibration_indices=np.asarray([0, 2], dtype=int),
    )

    result = few_shot.fit_few_shot_target_calibrated_decoder(
        source_features=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        ),
        source_labels=_object_vector(["face", 1, "face", 1]),
        target_features=np.asarray(
            [
                [1.1, 0.0],
                [1.0, 0.1],
                [0.0, 1.1],
                [0.1, 1.0],
            ]
        ),
        target_labels=_object_vector(["face", "face", 1, 1]),
        split=split,
        decoder_name="fake",
    )

    assert result.calibration_indices.tolist() == [0, 2]
    assert result.evaluation_indices.tolist() == [1, 3]
    assert result.probabilities.shape == (2, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.model.fit_labels_.tolist() == ["face", 1, "face", 1, "face", 1]
    assert result.model.classes_.tolist() == ["face", 1]
