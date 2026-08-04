from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clipping import apply_feature_clipping, fit_source_feature_clipping


def _rows(values: list[list[float]]) -> object:
    return ((value for value in row) for row in values)


def test_fit_source_feature_clipping_accepts_nested_one_pass_rows() -> None:
    result = fit_source_feature_clipping(
        source_features=_rows([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]),
        test_features=_rows([[-5.0, 100.0], [1.5, 11.5]]),
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    np.testing.assert_allclose(
        result.train_features,
        [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]],
    )
    np.testing.assert_allclose(result.test_features, [[0.0, 12.0], [1.5, 11.5]])


def test_apply_feature_clipping_accepts_one_pass_bounds() -> None:
    clipped = apply_feature_clipping(
        _rows([[-1.0, 5.0], [2.0, 9.0]]),
        lower_bounds=(value for value in [0.0, 6.0]),
        upper_bounds=(value for value in [1.0, 8.0]),
    )

    np.testing.assert_allclose(clipped, [[0.0, 6.0], [1.0, 8.0]])


def test_source_feature_clipping_still_rejects_complex_one_pass_values() -> None:
    complex_rows = ((value for value in row) for row in [[1.0 + 1.0j]])

    with pytest.raises(ValueError, match="complex"):
        apply_feature_clipping(
            complex_rows,
            lower_bounds=[0.0],
            upper_bounds=[2.0],
        )
