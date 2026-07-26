from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.row_l1 import normalize_rows_l1
from neureptrace.decoding.row_linf import normalize_rows_linf


@pytest.mark.parametrize("normalizer", [normalize_rows_l1, normalize_rows_linf])
@pytest.mark.parametrize(
    "features",
    [
        [[1.0 + 2.0j, 3.0]],
        np.asarray([[1.0 + 0.0j, 2.0]], dtype=np.complex128),
        np.asarray([[1.0 + 2.0j, 3.0]], dtype=object),
    ],
)
def test_row_l1_and_linf_reject_complex_feature_values(
    normalizer: Callable[..., tuple[np.ndarray, np.ndarray]],
    features: object,
) -> None:
    with pytest.raises(ValueError, match="complex"):
        normalizer(features)  # type: ignore[arg-type]


@pytest.mark.parametrize("normalizer", [normalize_rows_l1, normalize_rows_linf])
def test_row_l1_and_linf_reject_complex_values_in_one_pass_iterables(
    normalizer: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    rows = ((value for value in row) for row in ([1.0 + 2.0j, 3.0],))

    with pytest.raises(ValueError, match="complex"):
        normalizer(rows)  # type: ignore[arg-type]
