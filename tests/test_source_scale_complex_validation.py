from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scale import (
    apply_source_feature_scale,
    fit_source_feature_scale,
    fit_source_feature_scale_stats,
    source_feature_scale_config,
)


def test_source_scale_rejects_complex_source_feature_arrays() -> None:
    source = np.asarray([[1.0 + 2.0j], [3.0 + 4.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        fit_source_feature_scale_stats(source)


def test_source_scale_rejects_complex_test_feature_arrays() -> None:
    test = np.asarray([[0.5 + 1.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="test_features must contain real-valued feature values"):
        fit_source_feature_scale(
            source_features=[[0.0], [1.0]],
            test_features=test,
        )


def test_source_scale_apply_rejects_complex_feature_arrays() -> None:
    stats = fit_source_feature_scale_stats([[0.0], [1.0]])
    features = np.asarray([[0.5 + 1.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="features must contain real-valued feature values"):
        apply_source_feature_scale(features, stats)


@pytest.mark.parametrize(
    "epsilon",
    [
        np.complex64(1.0e-8 + 1.0j),
        np.complex128(1.0e-8 + 1.0j),
        np.asarray(1.0e-8 + 1.0j),
    ],
)
def test_source_scale_rejects_complex_epsilon_scalars(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        source_feature_scale_config(epsilon=epsilon)  # type: ignore[arg-type]
