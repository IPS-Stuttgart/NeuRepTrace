from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.row_l2 import RowL2Config, normalize_rows_l2, normalize_train_test_rows_l2, row_l2_config


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[3.0 + 4.0j, -5.0 + 12.0j]], dtype=np.complex128),
        np.asarray([[1.0, np.complex128(2.0 + 1.0j)]], dtype=object),
        ((value for value in row) for row in ([np.complex128(1.0 + 2.0j), 3.0], [4.0, 5.0])),
    ],
)
def test_normalize_rows_l2_rejects_complex_features(features: object) -> None:
    with pytest.raises(ValueError, match="real-valued.*complex"):
        normalize_rows_l2(features)  # type: ignore[arg-type]


def test_normalize_train_test_rows_l2_rejects_complex_features() -> None:
    with pytest.raises(ValueError, match="train_features.*real-valued.*complex"):
        normalize_train_test_rows_l2(
            train_features=np.asarray([[3.0 + 4.0j, 1.0]], dtype=np.complex128),
            test_features=[[1.0, 2.0]],
        )

    with pytest.raises(ValueError, match="test_features.*real-valued.*complex"):
        normalize_train_test_rows_l2(
            train_features=[[1.0, 2.0]],
            test_features=np.asarray([[3.0 + 4.0j, 1.0]], dtype=np.complex128),
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
def test_row_l2_rejects_complex_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowL2Config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_l2_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_l2([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]
