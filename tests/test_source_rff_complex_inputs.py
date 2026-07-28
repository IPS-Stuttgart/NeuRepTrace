from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_rff import apply_source_rff, fit_source_rff_reference, fit_source_rff_transform


def test_source_rff_rejects_complex_source_and_heldout_matrices() -> None:
    real_source = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    real_test = np.asarray([[4.0, 5.0]], dtype=float)
    complex_source = real_source.astype(np.complex128)
    complex_source[0, 0] += 2.0j
    complex_test = real_test.astype(np.complex64)
    complex_test[0, 1] += np.complex64(3.0j)

    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        fit_source_rff_transform(source_features=complex_source, test_features=real_test)

    with pytest.raises(ValueError, match="test_features must contain real-valued feature values"):
        fit_source_rff_transform(source_features=real_source, test_features=complex_test)


def test_apply_source_rff_rejects_complex_features() -> None:
    source = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    reference = fit_source_rff_reference(source, config={"n_components": 4, "random_state": 3})
    features = np.asarray([[4.0 + 1.0j, 5.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="features must contain real-valued feature values"):
        apply_source_rff(features, reference)


def test_source_rff_rejects_complex_nested_generators_without_exhaustion() -> None:
    source_rows = ((value for value in row) for row in ([0.0, 1.0], [2.0 + 1.0j, 3.0]))

    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        fit_source_rff_reference(source_rows)
