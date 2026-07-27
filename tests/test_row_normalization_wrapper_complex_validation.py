from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_normalization import (
    normalize_rows,
    normalize_source_and_test_rows,
    row_normalization_config,
    row_norms,
)


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_normalize_rows_rejects_complex_feature_arrays(norm: str) -> None:
    features = np.asarray([[1.0 + 2.0j, 2.0], [3.0, 4.0]], dtype=complex)

    with pytest.raises(ValueError, match="complex"):
        normalize_rows(features, config={"norm": norm})


def test_row_norms_rejects_complex_values_in_one_pass_iterables() -> None:
    features = (row for row in ([1.0, 2.0], [3.0 + 1.0j, 4.0]))

    with pytest.raises(ValueError, match="complex"):
        row_norms(features, norm="max")  # type: ignore[arg-type]


def test_source_and_test_wrapper_identifies_complex_test_features() -> None:
    with pytest.raises(ValueError, match="test_features.*complex"):
        normalize_source_and_test_rows(
            source_features=[[1.0, 2.0]],
            test_features=np.asarray([[3.0 + 1.0j, 4.0]], dtype=object),
        )


@pytest.mark.parametrize(
    "epsilon",
    [
        1e-6 + 1e-6j,
        np.complex128(1e-6 + 1e-6j),
        np.asarray(1e-6 + 1e-6j),
    ],
)
def test_row_normalization_config_rejects_complex_epsilon(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        row_normalization_config(epsilon=epsilon)  # type: ignore[arg-type]


def test_valid_real_inputs_are_unchanged() -> None:
    normalized, norms = normalize_rows([[3.0, 4.0], [0.0, 0.0]])

    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0.0, 0.0]])
    np.testing.assert_allclose(norms, [5.0, 0.0])
