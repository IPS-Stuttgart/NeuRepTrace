from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_outlier import (
    SourceOutlierConfig,
    compute_source_outlier_weights,
    source_outlier_config,
)


def test_source_outlier_rejects_complex_numpy_feature_matrix() -> None:
    features = np.asarray([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=complex)

    with pytest.raises(ValueError, match="source_features must contain real-valued features"):
        compute_source_outlier_weights(features, ["a", "b"])


def test_source_outlier_rejects_complex_sequence_feature_matrix() -> None:
    with pytest.raises(ValueError, match="source_features must contain real-valued features"):
        compute_source_outlier_weights([[0.0, 1.0 + 2.0j], [1.0, 0.0]], ["a", "b"])


def test_source_outlier_rejects_complex_generator_rows() -> None:
    rows = (iter(row) for row in ([0.0, 1.0 + 2.0j], [1.0, 0.0]))

    with pytest.raises(ValueError, match="source_features must contain real-valued features"):
        compute_source_outlier_weights(rows, ["a", "b"])


@pytest.mark.parametrize("name", ["quantile", "mad_multiplier", "temperature", "epsilon"])
@pytest.mark.parametrize(
    "value",
    [
        np.complex64(0.5 + 0.25j),
        np.complex128(0.5 + 0.25j),
        np.asarray(0.5 + 0.25j),
    ],
)
def test_source_outlier_config_rejects_complex_numeric_controls(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be finite and real-valued"):
        source_outlier_config(**{name: value})


@pytest.mark.parametrize("name", ["quantile", "mad_multiplier", "temperature", "epsilon"])
def test_source_outlier_direct_config_rejects_complex_numpy_scalars(name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be finite and real-valued"):
        SourceOutlierConfig(**{name: np.complex128(0.5 + 0.25j)})
