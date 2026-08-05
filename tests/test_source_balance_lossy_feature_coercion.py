from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding.source_balance import resample_source_rows_balanced


@pytest.mark.parametrize(
    "features_factory",
    [
        pytest.param(
            lambda: np.asarray([[1.0 + 2.0j], [3.0 + 4.0j]], dtype=np.complex128),
            id="complex-dtype-array",
        ),
        pytest.param(
            lambda: np.asarray([[1.0 + 2.0j], [3.0]], dtype=object),
            id="complex-object-array",
        ),
        pytest.param(
            lambda: ((value for value in row) for row in [[1.0 + 2.0j], [3.0 + 4.0j]]),
            id="complex-nested-generators",
        ),
        pytest.param(
            lambda: pd.DataFrame([[1.0 + 2.0j], [3.0 + 4.0j]]),
            id="complex-dataframe",
        ),
    ],
)
def test_source_balance_rejects_complex_features_before_float_coercion(
    features_factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="real-valued numeric features"):
        resample_source_rows_balanced(
            features_factory(),
            ["a", "b"],
            config={"strategy": "none"},
        )


def test_source_balance_rejects_boolean_dataframe_before_float_coercion() -> None:
    features = pd.DataFrame([[True, False], [False, True]])

    with pytest.raises(ValueError, match="boolean flags"):
        resample_source_rows_balanced(
            features,
            ["a", "b"],
            config={"strategy": "none"},
        )
