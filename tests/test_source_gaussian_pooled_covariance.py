from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_gaussian import fit_source_gaussian_decoder


def test_tied_gaussian_covariance_uses_pooled_within_class_degrees_of_freedom() -> None:
    source_features = np.asarray([[0.0], [2.0], [10.0], [12.0], [14.0], [16.0]], dtype=float)
    source_labels = np.asarray(["small", "small", "large", "large", "large", "large"], dtype=object)

    result = fit_source_gaussian_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=np.asarray([[1.0], [13.0]], dtype=float),
        config={"covariance_type": "tied_diagonal", "prior": "uniform"},
    )

    # The unbiased class variances are 2 and 20/3 with 1 and 3
    # within-class degrees of freedom, respectively.
    expected = ((1.0 * 2.0) + (3.0 * (20.0 / 3.0))) / 4.0
    np.testing.assert_allclose(result.variances, np.full((2, 1), expected), rtol=1e-6)


def test_singleton_class_does_not_contribute_floor_to_tied_covariance() -> None:
    result = fit_source_gaussian_decoder(
        source_features=np.asarray([[0.0], [10.0], [12.0]], dtype=float),
        source_labels=np.asarray(["singleton", "paired", "paired"], dtype=object),
        test_features=np.asarray([[0.0], [11.0]], dtype=float),
        config={
            "covariance_type": "tied_diagonal",
            "prior": "uniform",
            "variance_floor": 1e-3,
        },
    )

    # The singleton has zero within-class degrees of freedom. The paired class
    # has unbiased variance 2 and supplies the entire pooled estimate.
    np.testing.assert_allclose(result.variances, np.full((2, 1), 2.0), rtol=1e-6)
