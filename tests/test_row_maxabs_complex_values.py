from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.row_maxabs import (
    RowMaxAbsConfig,
    normalize_rows_maxabs,
    normalize_train_score_rows_maxabs,
    row_maxabs_config,
)


def _nested_generator_rows(rows: list[list[object]]) -> object:
    return ((value for value in row) for row in rows)


@pytest.mark.parametrize(
    "features_factory",
    [
        lambda: np.asarray([[1.0 + 2.0j, 3.0 + 0.0j]], dtype=np.complex128),
        lambda: np.asarray([[1.0 + 2.0j, 3.0]], dtype=object),
        lambda: _nested_generator_rows([[1.0 + 2.0j, 3.0]]),
    ],
)
def test_row_maxabs_rejects_complex_feature_values(features_factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="real-valued feature values"):
        normalize_rows_maxabs(features_factory())  # type: ignore[arg-type]


def test_train_score_row_maxabs_rejects_complex_numpy_features() -> None:
    with pytest.raises(ValueError, match="train_features must contain real-valued feature values"):
        normalize_train_score_rows_maxabs(
            train_features=np.asarray([[1.0 + 2.0j, 3.0]], dtype=np.complex128),
            score_features=[[1.0, 2.0]],
        )


@pytest.mark.parametrize(
    "epsilon",
    [
        1.0 + 2.0j,
        np.complex128(1.0 + 2.0j),
        np.asarray(np.complex128(1.0 + 2.0j)),
        np.asarray(np.complex128(1.0 + 2.0j), dtype=object),
    ],
)
def test_row_maxabs_rejects_complex_epsilon_values(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        RowMaxAbsConfig(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        row_maxabs_config(epsilon=epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epsilon"):
        normalize_rows_maxabs([[1.0, 2.0]], epsilon=epsilon)  # type: ignore[arg-type]
