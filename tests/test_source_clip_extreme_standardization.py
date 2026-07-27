from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clip import fit_source_clip_bounds, fit_source_clip_then_standardize


@pytest.mark.parametrize("center_mode", ["median", "mean"])
def test_source_clip_center_stays_finite_at_float_limit(center_mode: str) -> None:
    max_float = np.finfo(float).max

    bounds = fit_source_clip_bounds(
        [[max_float], [max_float]],
        config={"lower_quantile": 0.0, "upper_quantile": 1.0, "center": center_mode},
    )

    assert np.isfinite(bounds.center[0])
    assert bounds.center[0] == max_float


def test_clip_standardization_handles_constant_float_limit_values() -> None:
    max_float = np.finfo(float).max

    result = fit_source_clip_then_standardize(
        source_features=[[max_float], [max_float]],
        test_features=[[max_float]],
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert np.all(np.isfinite(result.center))
    assert np.all(np.isfinite(result.scale))
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert result.center[0] == max_float
    assert np.all(result.train_features == 0.0)
    assert np.all(result.test_features == 0.0)


def test_clip_standardization_saturates_unrepresentable_sample_scale() -> None:
    max_float = np.finfo(float).max

    result = fit_source_clip_then_standardize(
        source_features=[[-max_float], [max_float]],
        test_features=[[0.0]],
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert np.all(np.isfinite(result.center))
    assert np.all(np.isfinite(result.scale))
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert result.scale[0] == max_float
    assert result.train_features[:, 0].tolist() == [-1.0, 1.0]
