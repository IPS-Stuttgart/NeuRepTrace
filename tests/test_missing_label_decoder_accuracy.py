from __future__ import annotations

import numpy as np

from neureptrace.decoding.transfer import cross_validate_feature_decoding, evaluate_feature_transfer
from neureptrace.decoding.windowed import score_windowed_decoding


class _MissingLabelSignClassifier:
    def predict(self, features):
        values = np.asarray(features)[:, 0]
        return np.where(values < 0.0, np.float64("nan"), 1.0)

    def decision_function(self, features):
        return np.asarray(features)[:, 0]


def _fit_missing_label_sign_classifier(_features, _labels):
    return _MissingLabelSignClassifier()


def test_windowed_accuracy_treats_equivalent_missing_labels_as_equal() -> None:
    result = score_windowed_decoding(
        train_features=[[-2.0], [-1.0], [1.0], [2.0]],
        train_labels=[float("nan"), np.float64("nan"), 1.0, 1.0],
        validation_features=[[-1.5], [1.5]],
        validation_labels=[float("nan"), 1.0],
        fit_model=_fit_missing_label_sign_classifier,
        components_pca=float("inf"),
    )

    assert result.accuracy == 1.0
    assert result.balanced_accuracy == 1.0


def test_transfer_accuracy_uses_the_same_missing_label_semantics() -> None:
    result = evaluate_feature_transfer(
        train_features=[[-2.0], [-1.0], [1.0], [2.0]],
        train_labels=[float("nan"), np.float64("nan"), 1.0, 1.0],
        validation_features=[[-1.5], [1.5]],
        validation_labels=[np.float64("nan"), 1.0],
        fit_model=_fit_missing_label_sign_classifier,
        components_pca=float("inf"),
    )

    assert result.accuracy == 1.0


def test_cross_validation_accuracy_treats_missing_predictions_as_correct() -> None:
    result = cross_validate_feature_decoding(
        stimulus_features=[[-2.0], [-1.0], [1.0], [2.0]],
        labels=np.asarray([float("nan"), np.float64("nan"), 1.0, 1.0]),
        n_folds=2,
        fit_model=_fit_missing_label_sign_classifier,
        components_pca=float("inf"),
    )

    assert result.accuracy == 1.0
    assert np.isnan(result.predictions[:2]).all()
    np.testing.assert_array_equal(result.predictions[2:], [1.0, 1.0])
