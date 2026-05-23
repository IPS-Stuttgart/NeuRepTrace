import numpy as np
import pytest

from neureptrace.decoding.windowed import score_windowed_decoding


class ConstantClassifier:
    def __init__(self, label=0):
        self.label = label

    def fit(self, features, labels):
        del features, labels
        return self

    def predict(self, features):
        return np.full(np.asarray(features).shape[0], self.label, dtype=int)

    def decision_function(self, features):
        return np.zeros(np.asarray(features).shape[0], dtype=float)


def _fit_constant_zero(features, labels):
    return ConstantClassifier(label=0).fit(features, labels)


def test_windowed_decoding_balanced_accuracy_matches_plain_accuracy_when_balanced():
    result = score_windowed_decoding(
        train_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        validation_features=np.array([[-2.0], [-1.0]]),
        validation_labels=np.array([0, 0]),
        fit_model=_fit_constant_zero,
    )

    assert result.accuracy == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)


def test_windowed_decoding_balanced_accuracy_exposes_imbalanced_failure():
    result = score_windowed_decoding(
        train_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        validation_features=np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        validation_labels=np.array([0, 0, 0, 0, 1]),
        fit_model=_fit_constant_zero,
    )

    assert result.accuracy == pytest.approx(0.8)
    assert result.balanced_accuracy == pytest.approx(0.5)
