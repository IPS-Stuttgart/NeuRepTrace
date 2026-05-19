from types import SimpleNamespace

import numpy as np

from neureptrace.decoding import transfer


class _DummyBinaryModel:
    def __init__(self, score: float):
        self.score = score

    def predict(self, features):
        return np.zeros(features.shape[0], dtype=float)


def test_gradient_boosting_one_vs_rest_uses_positive_class_scores(monkeypatch):
    fitted_scores = iter((0.1, 0.9))

    def fake_fit_window_model(*_args, **_kwargs):
        return SimpleNamespace(model=_DummyBinaryModel(next(fitted_scores)))

    def fake_positive_class_score(model, features):
        return np.full(features.shape[0], model.score, dtype=float)

    monkeypatch.setattr(transfer, "fit_window_model", fake_fit_window_model)
    monkeypatch.setattr(
        transfer,
        "transform_window_features",
        lambda _bundle, features: features,
    )
    monkeypatch.setattr(transfer, "positive_class_score", fake_positive_class_score)

    predictions = transfer._one_vs_rest_predictions(
        np.array([[0.0], [1.0]]),
        np.array([1, 2]),
        np.array([[0.5]]),
        np.array([1, 2]),
        classifier="gradient-boosting",
        classifier_param=100,
        components_pca=float("inf"),
        random_state=0,
    )

    assert predictions.tolist() == [2]
