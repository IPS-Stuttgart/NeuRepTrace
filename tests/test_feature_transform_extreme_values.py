from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.signed_log1p import (
    signed_log1p_transform,
    transform_train_test_signed_log1p,
)
from neureptrace.decoding.source_clip import (
    fit_source_clip,
    fit_source_clip_then_standardize,
)


def test_signed_log1p_remains_finite_for_extreme_finite_ratio() -> None:
    max_float = np.finfo(float).max
    min_positive = np.nextafter(0.0, 1.0)

    transformed = signed_log1p_transform(
        [[-max_float, max_float]],
        scale=min_positive,
    )

    expected_magnitude = np.log(max_float) - np.log(min_positive)
    assert np.all(np.isfinite(transformed))
    assert np.allclose(
        transformed,
        [[-expected_magnitude, expected_magnitude]],
        rtol=1e-15,
        atol=0.0,
    )


def test_signed_log1p_compaction_preserves_subnormal_nonzero_outputs() -> None:
    min_positive = np.nextafter(0.0, 1.0)

    result = transform_train_test_signed_log1p(
        train_features=[[min_positive]],
        test_features=[[-min_positive]],
    )

    assert result.train_features[0, 0] != 0.0
    assert result.test_features[0, 0] != 0.0
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))


@pytest.mark.parametrize("magnitude", [1e40, 1e-50])
def test_source_clip_compaction_preserves_finite_values(magnitude: float) -> None:
    source = np.asarray([[magnitude], [2.0 * magnitude]], dtype=float)
    held_out = np.asarray([[1.5 * magnitude]], dtype=float)

    result = fit_source_clip(
        source_features=source,
        test_features=held_out,
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert np.array_equal(result.train_features, source)
    assert np.array_equal(result.test_features, held_out)
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))


def test_source_clip_standardization_stays_finite_for_large_values() -> None:
    result = fit_source_clip_then_standardize(
        source_features=[[1e40], [2e40]],
        test_features=[[1.5e40]],
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert np.all(np.isfinite(result.center))
    assert np.all(np.isfinite(result.scale))
