from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.windowed import fit_window_model, transform_window_features


class _RecordingModel:
    def __init__(self, fitted_features: np.ndarray):
        self.fitted_features = np.asarray(fitted_features, dtype=float).copy()


def test_windowed_pca_centers_extreme_finite_offsets_without_overflow() -> None:
    train_features = np.asarray(
        [
            [1e308, 0.0],
            [1e308, 1.0],
            [1e308, 2.0],
        ]
    )

    model_bundle = fit_window_model(
        train_features,
        np.asarray([0, 0, 1]),
        fit_model=lambda features, _labels: _RecordingModel(features),
        components_pca=1,
    )

    np.testing.assert_allclose(model_bundle.train_features_mean, [1e308, 1.0])
    assert np.all(np.isfinite(model_bundle.pca_coeff))
    assert np.all(np.isfinite(model_bundle.model.fitted_features))
    transformed = transform_window_features(model_bundle, train_features)
    assert np.all(np.isfinite(transformed))
    assert model_bundle.actual_components_pca == 1
    assert model_bundle.explained_variance_percent == pytest.approx(100.0)
    np.testing.assert_allclose(np.sort(np.abs(transformed[:, 0])), [0.0, 1.0, 1.0])
