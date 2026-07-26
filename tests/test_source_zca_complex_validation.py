from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_zca import (
    SourceZCAConfig,
    apply_source_zca_transform,
    fit_source_zca_reference,
    fit_source_zca_transform,
    source_zca_config,
)


@pytest.mark.parametrize(
    ("argument", "features"),
    [
        (
            "source_features",
            np.asarray([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128),
        ),
        (
            "test_features",
            np.asarray([[0.5 + 0.25j, 0.5]], dtype=object),
        ),
    ],
)
def test_source_zca_rejects_complex_fit_features(
    argument: str,
    features: np.ndarray,
) -> None:
    kwargs = {
        "source_features": [[0.0, 0.0], [1.0, 1.0]],
        "test_features": [[0.5, 0.5]],
    }
    kwargs[argument] = features

    with pytest.raises(ValueError, match=rf"{argument}.*real-valued.*complex"):
        fit_source_zca_transform(**kwargs)  # type: ignore[arg-type]


def test_source_zca_rejects_complex_one_pass_features() -> None:
    source = (
        (value for value in row)
        for row in ([0.0 + 1.0j, 0.0], [1.0, 1.0])
    )

    with pytest.raises(ValueError, match="source_features.*real-valued.*complex"):
        fit_source_zca_reference(source)


def test_source_zca_rejects_complex_apply_features() -> None:
    reference = fit_source_zca_reference([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="features.*real-valued.*complex"):
        apply_source_zca_transform(
            np.asarray([[0.5 + 0.25j, 0.5]], dtype=np.complex128),
            reference,
        )


@pytest.mark.parametrize(
    "regularization",
    [
        1.0e-4 + 1.0e-5j,
        np.complex64(1.0e-4 + 1.0e-5j),
        np.complex128(1.0e-4 + 1.0e-5j),
        np.asarray(1.0e-4 + 1.0e-5j),
        np.asarray(1.0e-4 + 1.0e-5j, dtype=object),
    ],
)
def test_source_zca_rejects_complex_regularization(regularization: object) -> None:
    with pytest.raises(ValueError, match="regularization"):
        source_zca_config(regularization=regularization)

    with pytest.raises(ValueError, match="regularization"):
        SourceZCAConfig(regularization=regularization)  # type: ignore[arg-type]
