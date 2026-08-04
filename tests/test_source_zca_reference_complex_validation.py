from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_zca import (
    SourceZCAConfig,
    SourceZCAReference,
    fit_source_zca_reference,
)


@pytest.mark.parametrize("field_name", ["mean", "whitening", "coloring", "eigenvalues"])
def test_direct_reference_rejects_complex_fitted_arrays(field_name: str) -> None:
    values: dict[str, object] = {
        "mean": np.asarray([0.0, 0.0]),
        "whitening": np.eye(2),
        "coloring": np.eye(2),
        "eigenvalues": np.asarray([1.0, 1.0]),
        "config": SourceZCAConfig(),
        "n_fit_rows": 2,
    }
    values[field_name] = np.asarray(values[field_name], dtype=complex) + 1j

    with pytest.raises(ValueError, match=rf"{field_name}.*real-valued reference values"):
        SourceZCAReference(**values)  # type: ignore[arg-type]


def test_fitted_reference_remains_valid() -> None:
    reference = fit_source_zca_reference([[0.0, 0.0], [1.0, 1.0]])

    assert np.isrealobj(reference.mean)
    assert np.isrealobj(reference.whitening)
    assert np.isrealobj(reference.coloring)
    assert np.isrealobj(reference.eigenvalues)
