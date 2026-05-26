from __future__ import annotations

import numpy as np

from neureptrace.decoding import make_decoder, predict_emission_probabilities


def test_calibrated_linear_svm_reduces_inner_cv_for_sparse_folds() -> None:
    features = np.array([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
    labels = np.array([0, 0, 1, 1])

    model = make_decoder("linear_svm", max_iter=5000)
    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert model.requested_calibration_cv_ == 3
    assert model.calibration_cv_ == 2
    assert model.used_uncalibrated_fallback_ is False
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_calibrated_linear_svm_falls_back_when_calibration_cv_is_impossible() -> None:
    features = np.array([[-1.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 1])

    model = make_decoder("linear_svm", max_iter=5000)
    model.fit(features, labels)
    probabilities = predict_emission_probabilities(model, features)

    assert model.calibration_cv_ == 0
    assert model.used_uncalibrated_fallback_ is True
    assert probabilities.shape == (2, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
