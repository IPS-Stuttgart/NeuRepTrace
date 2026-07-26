from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_l1 import (
    RowL1Config,
    normalize_rows_l1,
    row_l1_config,
)
from neureptrace.decoding.row_linf import (
    RowLinfConfig,
    normalize_rows_linf,
    row_linf_config,
)


@pytest.mark.parametrize("normalizer", [normalize_rows_l1, normalize_rows_linf])
@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[3.0 + 4.0j, -5.0 + 12.0j]], dtype=np.complex128),
        np.asarray([[1.0, np.complex128(2.0 + 1.0j)]], dtype=object),
    ],
)
def test_row_normalizers_reject_complex_feature_arrays(
    normalizer,
    features: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="real-valued.*complex"):
        normalizer(features)


@pytest.mark.parametrize("normalizer", [normalize_rows_l1, normalize_rows_linf])
def test_row_normalizers_reject_complex_one_pass_feature_iterables(normalizer) -> None:
    rows = ((value for value in row) for row in ([1.0 + 2.0j, 3.0], [4.0, 5.0]))

    with pytest.raises(ValueError, match="real-valued.*complex"):
        normalizer(rows)


@pytest.mark.parametrize(
    ("config_type", "config_factory", "normalizer"),
    [
        (RowL1Config, row_l1_config, normalize_rows_l1),
        (RowLinfConfig, row_linf_config, normalize_rows_linf),
    ],
)
@pytest.mark.parametrize(
    "epsilon",
    [
        1.0 + 0.5j,
        np.complex64(1.0 + 0.5j),
        np.complex128(1.0 + 0.5j),
        np.asarray(1.0 + 0.5j),
    ],
)
def test_row_normalizers_reject_complex_epsilon_values(
    config_type,
    config_factory,
    normalizer,
    epsilon: object,
) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        config_type(epsilon=epsilon)

    with pytest.raises(ValueError, match="epsilon"):
        config_factory(epsilon=epsilon)

    with pytest.raises(ValueError, match="epsilon"):
        normalizer([[1.0, 2.0]], epsilon=epsilon)
