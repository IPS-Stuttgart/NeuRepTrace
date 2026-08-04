from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.source_pca import (
    apply_source_pca_transform,
    fit_source_pca_reference,
    fit_source_pca_transform,
)


def _source() -> np.ndarray:
    return np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)


@pytest.mark.parametrize(
    "operation",
    [
        lambda values: fit_source_pca_reference(values, config={"n_components": 1}),
        lambda values: fit_source_pca_transform(
            source_features=values,
            test_features=[[0.5, 1.5]],
            config={"n_components": 1},
        ),
    ],
)
def test_source_pca_rejects_complex_source_features(operation: Callable[[object], object]) -> None:
    values = np.asarray([[0.0 + 1.0j, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=complex)

    with pytest.raises(ValueError, match="real-valued feature values"):
        operation(values)


def test_source_pca_rejects_complex_heldout_features() -> None:
    test = np.asarray([[0.5 + 0.25j, 1.5]], dtype=complex)

    with pytest.raises(ValueError, match="real-valued feature values"):
        fit_source_pca_transform(
            source_features=_source(),
            test_features=test,
            config={"n_components": 1},
        )


def test_source_pca_apply_rejects_complex_features() -> None:
    reference = fit_source_pca_reference(_source(), config={"n_components": 1})
    test = np.asarray([[0.5 + 0.25j, 1.5]], dtype=complex)

    with pytest.raises(ValueError, match="real-valued feature values"):
        apply_source_pca_transform(test, reference)


def test_source_pca_rejects_complex_values_in_nested_generators() -> None:
    rows = (iter(row) for row in [[0.0 + 1.0j, 1.0], [1.0, 2.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match="real-valued feature values"):
        fit_source_pca_reference(rows, config={"n_components": 1})
