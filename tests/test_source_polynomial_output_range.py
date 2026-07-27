from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_polynomial import (
    apply_source_polynomial_transform,
    fit_source_polynomial_reference,
    fit_source_polynomial_transform,
)


def test_polynomial_fit_preserves_outputs_outside_float32_range() -> None:
    source = np.asarray([[1.0e20, 2.0e20]], dtype=float)
    test = np.asarray([[1.0e-30, 2.0e-30]], dtype=float)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = fit_source_polynomial_transform(
            source_features=source,
            test_features=test,
        )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert np.count_nonzero(result.train_features) == result.train_features.size
    assert np.count_nonzero(result.test_features) == result.test_features.size
    np.testing.assert_allclose(
        result.train_features,
        [[1.0e20, 2.0e20, 1.0e40, 4.0e40, 2.0e40]],
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.test_features,
        [[1.0e-30, 2.0e-30, 1.0e-60, 4.0e-60, 2.0e-60]],
        rtol=1e-15,
        atol=0.0,
    )


def test_polynomial_apply_keeps_float32_for_representable_outputs() -> None:
    reference = fit_source_polynomial_reference(2)

    transformed = apply_source_polynomial_transform([[1.0, 2.0]], reference)

    assert transformed.dtype == np.float32
