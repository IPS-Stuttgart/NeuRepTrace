from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_knn import SourceKNNConfig, SourceKNNReference


def _valid_reference_kwargs() -> dict[str, object]:
    return {
        "features": np.asarray([[0.0], [1.0]], dtype=float),
        "labels": np.asarray(["left", "right"], dtype=object),
        "classes": np.asarray(["left", "right"], dtype=object),
        "mean": np.asarray([0.5], dtype=float),
        "scale": np.asarray([0.5], dtype=float),
        "config": SourceKNNConfig(k=1),
    }


@pytest.mark.parametrize(
    ("field_name", "complex_value"),
    [
        ("features", np.asarray([[0.0 + 1.0j], [1.0 + 0.0j]])),
        ("mean", np.asarray([0.5 + 1.0j], dtype=object)),
        ("scale", np.asarray([0.5 + 1.0j])),
    ],
)
def test_direct_source_knn_reference_rejects_complex_fitted_arrays(
    field_name: str,
    complex_value: np.ndarray,
) -> None:
    kwargs = _valid_reference_kwargs()
    kwargs[field_name] = complex_value

    with pytest.raises(
        ValueError,
        match=rf"SourceKNNReference\.{field_name} must contain real-valued values",
    ):
        SourceKNNReference(**kwargs)


def test_direct_source_knn_reference_accepts_real_fitted_arrays() -> None:
    reference = SourceKNNReference(**_valid_reference_kwargs())

    assert reference.features.dtype == np.float64
    assert reference.mean.dtype == np.float64
    assert reference.scale.dtype == np.float64
