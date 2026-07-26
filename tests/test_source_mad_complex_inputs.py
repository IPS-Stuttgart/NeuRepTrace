from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mad import apply_source_mad_transform, fit_source_mad_reference, fit_source_mad_transform


def test_source_mad_reference_rejects_complex_source_features() -> None:
    features = np.array([[0.0 + 1.0j], [1.0 + 0.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match=r"source_features.*real-valued"):
        fit_source_mad_reference(features)


def test_source_mad_fit_rejects_complex_test_features() -> None:
    source = np.array([[0.0], [1.0]], dtype=float)
    test = np.array([[0.5 + 2.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match=r"test_features.*real-valued"):
        fit_source_mad_transform(source_features=source, test_features=test)


def test_source_mad_apply_rejects_complex_features() -> None:
    reference = fit_source_mad_reference([[0.0], [1.0]])
    features = np.array([[0.5 + 2.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match=r"features.*real-valued"):
        apply_source_mad_transform(features, reference)


def test_source_mad_rejects_complex_values_from_one_pass_iterables() -> None:
    source = ((value for value in row) for row in ((0.0 + 0.0j,), (1.0 + 1.0j,)))

    with pytest.raises(ValueError, match=r"source_features.*real-valued"):
        fit_source_mad_reference(source)
