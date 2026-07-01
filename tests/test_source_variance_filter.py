from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_variance_filter import (
    SOURCE_VARIANCE_FILTER_CATEGORY,
    SourceVarianceFilterConfig,
    fit_source_variance_filter,
    select_variance_features,
    source_feature_variances,
    source_variance_filter_config,
)


def test_source_variance_filter_selects_source_variance_features() -> None:
    source = np.asarray([[1.0, 0.0, 0.0, 5.0], [1.0, 1.0, 0.0, 7.0], [1.0, 2.0, 0.0, 9.0]])
    test = np.asarray([[99.0, 3.0, 4.0, 11.0]])

    result = fit_source_variance_filter(source_features=source, test_features=test, config={"variance_threshold": 0.0})

    assert result.selected_indices.tolist() == [1, 3]
    assert result.train_features.shape == (3, 2)
    assert result.test_features.tolist() == [[3.0, 11.0]]
    assert result.metadata["source_variance_filter_protocol_category"] == SOURCE_VARIANCE_FILTER_CATEGORY
    assert result.metadata["source_variance_filter_uses_test_features_for_fitting"] is False
    assert result.metadata["source_variance_filter_valid_for_strict_source_only"] is True


def test_source_variance_filter_top_k_limits_selected_features() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 10.0], [2.0, 4.0, 20.0]])
    test = np.asarray([[3.0, 6.0, 30.0]])

    result = fit_source_variance_filter(source_features=source, test_features=test, config={"top_k": 1})

    assert result.selected_indices.tolist() == [2]
    assert result.test_features.tolist() == [[30.0]]


def test_select_variance_features_falls_back_to_highest_variance() -> None:
    assert select_variance_features([0.0, 0.0, 2.0], variance_threshold=10.0).tolist() == [2]


def test_source_feature_variances_uses_ddof_zero_for_too_few_rows() -> None:
    assert np.allclose(source_feature_variances([[1.0, 2.0]], ddof=1), np.asarray([0.0, 0.0]))


def test_source_variance_filter_config_validation() -> None:
    cfg = source_variance_filter_config(variance_threshold="0.5", top_k="2", ddof="0")
    assert cfg.variance_threshold == 0.5
    assert cfg.top_k == 2
    assert cfg.ddof == 0

    with pytest.raises(ValueError):
        source_variance_filter_config(variance_threshold=-0.1)

    with pytest.raises(ValueError):
        source_variance_filter_config(top_k=0)


def test_source_variance_filter_config_normalizes_none_like_top_k() -> None:
    for value in ("", "none", " None ", "null", np.asarray("none", dtype=object)):
        assert source_variance_filter_config(top_k=value).top_k is None


def test_source_variance_filter_dataclass_config_is_revalidated_before_use() -> None:
    source = np.asarray([[1.0, 0.0, 0.0, 5.0], [1.0, 1.0, 0.0, 7.0], [1.0, 2.0, 0.0, 9.0]])
    test = np.asarray([[99.0, 3.0, 4.0, 11.0]])
    config = SourceVarianceFilterConfig(variance_threshold="0.0", top_k=" none ", ddof="0")

    result = fit_source_variance_filter(source_features=source, test_features=test, config=config)

    assert result.selected_indices.tolist() == [1, 3]
    assert result.metadata["source_variance_filter_top_k"] == ""
    assert result.metadata["source_variance_filter_ddof"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance_threshold": True},
        {"variance_threshold": np.bool_(False)},
        {"top_k": True},
        {"top_k": np.bool_(True)},
        {"ddof": False},
        {"ddof": np.bool_(False)},
    ],
)
def test_source_variance_filter_config_rejects_boolean_numeric_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        source_variance_filter_config(**kwargs)


def test_source_variance_filter_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_variance_filter(source_features=[[0.0, 1.0]], test_features=[[0.0]])
