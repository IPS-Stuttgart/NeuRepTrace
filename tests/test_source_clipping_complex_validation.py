from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clipping import (
    apply_feature_clipping,
    fit_source_feature_clipping,
    source_feature_clipping_bounds,
    source_feature_clipping_config,
)


@pytest.mark.parametrize("argument", ["source_features", "test_features"])
def test_source_clipping_rejects_complex_feature_matrices(argument: str) -> None:
    kwargs = {
        "source_features": np.asarray([[0.0], [1.0]], dtype=float),
        "test_features": np.asarray([[0.5]], dtype=float),
        "config": {"lower_quantile": 0.0, "upper_quantile": 1.0},
    }
    kwargs[argument] = np.asarray(kwargs[argument], dtype=np.complex128) + np.complex128(0.25j)

    with pytest.raises(ValueError, match=rf"{argument} must contain real-valued feature values"):
        fit_source_feature_clipping(**kwargs)  # type: ignore[arg-type]


def test_source_clipping_bounds_reject_complex_features() -> None:
    source = np.asarray([[0.0 + 0.5j], [1.0 + 0.5j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        source_feature_clipping_bounds(source)


@pytest.mark.parametrize("argument", ["lower_bounds", "upper_bounds"])
def test_apply_source_clipping_rejects_complex_bounds(argument: str) -> None:
    kwargs = {
        "features": [[0.5]],
        "lower_bounds": [0.0],
        "upper_bounds": [1.0],
    }
    kwargs[argument] = np.asarray(kwargs[argument], dtype=np.complex128) + np.complex128(0.25j)

    with pytest.raises(ValueError, match=rf"{argument} must contain real-valued bounds"):
        apply_feature_clipping(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        np.complex64(0.1 + 0.2j),
        np.complex128(0.1 + 0.2j),
        np.asarray(0.1 + 0.2j),
    ],
)
def test_source_clipping_rejects_complex_quantile_controls(value: object) -> None:
    with pytest.raises(ValueError, match="lower_quantile.*not complex"):
        source_feature_clipping_config(lower_quantile=value, upper_quantile=0.9)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="upper_quantile.*not complex"):
        source_feature_clipping_config(lower_quantile=0.1, upper_quantile=value)  # type: ignore[arg-type]
