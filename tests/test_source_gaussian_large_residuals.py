from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_gaussian import gaussian_log_likelihoods


def test_gaussian_log_likelihoods_normalizes_before_squaring_large_residuals() -> None:
    features = np.asarray([[1.0e200, -1.0e200]], dtype=float)
    means = np.asarray([[0.0, 0.0], [1.0e200, 0.0]], dtype=float)
    variances = np.full((2, 2), 1.0e200, dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scores = gaussian_log_likelihoods(
            features,
            means=means,
            variances=variances,
        )

    expected_quadratic = np.asarray([[2.0e200, 1.0e200]], dtype=float)
    expected_log_det = 2.0 * np.log(1.0e200)
    expected = -0.5 * (expected_quadratic + expected_log_det + 2.0 * np.log(2.0 * np.pi))

    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, expected)
    assert scores[0, 1] > scores[0, 0]
