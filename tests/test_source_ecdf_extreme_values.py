from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_ecdf_uniform import fit_source_ecdf_map, fit_source_ecdf_transform


def test_source_ecdf_quantiles_stay_finite_for_extreme_source_rows() -> None:
    source = np.asarray([[-1e308], [1e308], [1e308], [1e308]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ecdf_map = fit_source_ecdf_map(source, config={"n_quantiles": 4})

    assert np.all(np.isfinite(ecdf_map.quantiles))
    assert ecdf_map.quantiles[0, 0] == pytest.approx(-1e308)
    assert ecdf_map.quantiles[-1, 0] == pytest.approx(1e308)


def test_source_ecdf_interpolates_between_extreme_finite_knots() -> None:
    source = np.asarray([[-1e308], [1e308]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = fit_source_ecdf_transform(
            source_features=source,
            test_features=np.asarray([[0.0]], dtype=float),
            config={"n_quantiles": 2},
        )

    assert result.test_features[0, 0] == pytest.approx(0.5)
