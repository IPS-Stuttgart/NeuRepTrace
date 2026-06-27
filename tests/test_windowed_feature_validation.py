import numpy as np
import pytest

from neureptrace.decoding.windowed import fit_window_model, predict_window_model, score_windowed_decoding


class _ConstantClassifier:
    def fit(self, features, labels):
        self.label_ = np.asarray(labels)[0]
        return self

    def predict(self, features):
        return np.full(np.asarray(features).shape[0], self.label_)

    def decision_function(self, features):
        return np.zeros(np.asarray(features).shape[0])


def _fit_constant_classifier(features, labels):
    return _ConstantClassifier().fit(features, labels)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_fit_window_model_rejects_non_finite_train_features(bad_value):
    train_features = np.array([[0.0, 1.0], [bad_value, 2.0]])

    with pytest.raises(ValueError, match="train_features must contain only finite values"):
        fit_window_model(
            train_features,
            np.array([0, 1]),
            fit_model=_fit_constant_classifier,
            components_pca=float("inf"),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_score_windowed_decoding_rejects_non_finite_validation_features(bad_value):
    validation_features = np.array([[bad_value], [1.0]])

    with pytest.raises(ValueError, match="validation_features must contain only finite values"):
        score_windowed_decoding(
            train_features=np.array([[0.0], [1.0]]),
            train_labels=np.array([0, 1]),
            validation_features=validation_features,
            validation_labels=np.array([0, 1]),
            fit_model=_fit_constant_classifier,
            components_pca=float("inf"),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_predict_window_model_rejects_non_finite_features(bad_value):
    bundle = fit_window_model(
        np.array([[0.0], [1.0]]),
        np.array([0, 1]),
        fit_model=_fit_constant_classifier,
        components_pca=float("inf"),
    )

    with pytest.raises(ValueError, match="features must contain only finite values"):
        predict_window_model(bundle, np.array([[bad_value], [1.0]]))
