from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_correlation_filter import (
    SOURCE_CORRELATION_FILTER_CATEGORY,
    fit_source_correlation_filter,
    select_uncorrelated_features,
    source_correlation_filter_config,
    source_feature_correlation,
)


def test_source_correlation_filter_removes_redundant_source_features() -> None:
    source = np.asarray(
        [
            [0.0, 0.0, 1.0, 3.0],
            [1.0, 1.0, 1.0, 2.0],
            [2.0, 2.0, 1.0, 1.0],
            [3.0, 3.0, 1.0, 0.0],
        ]
    )
    test = np.asarray([[9.0, 9.0, 8.0, 7.0]])

    result = fit_source_correlation_filter(
        source_features=source,
        test_features=test,
        config={"max_abs_correlation": 0.95},
    )

    assert 0 in result.selected_indices.tolist() or 1 in result.selected_indices.tolist()
    assert not ({0, 1} <= set(result.selected_indices.tolist()))
    assert result.train_features.shape[1] == result.selected_indices.shape[0]
    assert result.test_features.shape[1] == result.selected_indices.shape[0]
    assert result.metadata["source_correlation_filter_protocol_category"] == SOURCE_CORRELATION_FILTER_CATEGORY
    assert result.metadata["source_correlation_filter_uses_test_features_for_fitting"] is False
    assert result.metadata["source_correlation_filter_valid_for_strict_source_only"] is True


def test_source_correlation_filter_respects_max_features() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 10.0], [2.0, 4.0, 20.0], [3.0, 8.0, 25.0]])
    test = np.asarray([[1.0, 2.0, 3.0]])

    result = fit_source_correlation_filter(source_features=source, test_features=test, config={"max_features": 1})

    assert result.selected_indices.shape == (1,)
    assert result.test_features.shape == (1, 1)


def test_source_feature_correlation_handles_constant_features() -> None:
    corr = source_feature_correlation([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])

    assert np.all(np.isfinite(corr))
    assert np.allclose(np.diag(corr), 1.0)
    assert np.isclose(corr[0, 1], 0.0)


def test_select_uncorrelated_features_minimum_fallback() -> None:
    corr = np.ones((3, 3), dtype=float)
    selected = select_uncorrelated_features(corr, importance=[1.0, 3.0, 2.0], max_abs_correlation=0.0, min_features=2)

    assert selected.tolist() == [1, 2]


def test_source_correlation_filter_config_validation() -> None:
    cfg = source_correlation_filter_config(max_abs_correlation="0.5", max_features="2", min_features="1")
    assert cfg.max_abs_correlation == 0.5
    assert cfg.max_features == 2
    assert cfg.min_features == 1

    assert source_correlation_filter_config(max_features=" none ").max_features is None

    with pytest.raises(ValueError):
        source_correlation_filter_config(max_abs_correlation=1.5)

    with pytest.raises(ValueError):
        source_correlation_filter_config(min_features=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_abs_correlation": True},
        {"max_abs_correlation": np.bool_(False)},
        {"max_features": True},
        {"max_features": np.bool_(False)},
        {"min_features": False},
        {"epsilon": np.bool_(True)},
    ],
)
def test_source_correlation_filter_config_rejects_boolean_numeric_values(kwargs) -> None:
    option = next(iter(kwargs))
    with pytest.raises(ValueError, match=option):
        source_correlation_filter_config(**kwargs)


def test_source_correlation_filter_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_correlation_filter(source_features=[[0.0, 1.0]], test_features=[[0.0]])
