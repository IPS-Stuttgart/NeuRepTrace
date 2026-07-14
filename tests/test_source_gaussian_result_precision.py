from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_gaussian import fit_source_gaussian_decoder


def test_source_gaussian_preserves_large_finite_result_statistics() -> None:
    source_features = np.asarray([[1.0e40], [1.0e40], [2.0e40], [2.0e40]], dtype=float)
    test_features = np.asarray([[1.0e40], [2.0e40]], dtype=float)

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        result = fit_source_gaussian_decoder(
            source_features=source_features,
            source_labels=[0, 0, 1, 1],
            test_features=test_features,
        )

    assert result.predictions.tolist() == [0, 1]
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.means.dtype == np.float64
    assert result.log_likelihoods.dtype == np.float64
    assert np.all(np.isfinite(result.means))
    assert np.all(np.isfinite(result.log_likelihoods))
    np.testing.assert_allclose(result.means.ravel(), [1.0e40, 2.0e40])


def test_source_gaussian_preserves_tiny_nonzero_variance_floor() -> None:
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        result = fit_source_gaussian_decoder(
            source_features=[[0.0], [0.0], [1.0], [1.0]],
            source_labels=[0, 0, 1, 1],
            test_features=[[0.0], [1.0]],
            config={"variance_floor": 1.0e-100},
        )

    assert result.variances.dtype == np.float64
    assert np.all(result.variances == 1.0e-100)
    assert np.all(np.isfinite(result.variances))
    assert np.all(result.variances > 0.0)
