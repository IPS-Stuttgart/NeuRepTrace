from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clipping import (
    SOURCE_CLIPPING_CATEGORY,
    apply_feature_clipping,
    fit_source_feature_clipping,
    source_feature_clipping_bounds,
    source_feature_clipping_config,
)


def test_source_feature_clipping_fits_bounds_from_source_only() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [100.0, 13.0]], dtype=float)
    test = np.asarray([[-10.0, 9.0], [50.0, 99.0]], dtype=float)

    result = fit_source_feature_clipping(
        source_features=source,
        test_features=test,
        config={"lower_quantile": 0.25, "upper_quantile": 0.75},
    )

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert np.all(result.train_features >= result.lower_bounds)
    assert np.all(result.train_features <= result.upper_bounds)
    assert np.all(result.test_features >= result.lower_bounds)
    assert np.all(result.test_features <= result.upper_bounds)
    assert result.metadata["source_feature_clipping_protocol_category"] == SOURCE_CLIPPING_CATEGORY
    assert result.metadata["source_feature_clipping_uses_source_features"] is True
    assert result.metadata["source_feature_clipping_uses_test_features_for_fitting"] is False
    assert result.metadata["source_feature_clipping_uses_test_labels"] is False
    assert result.metadata["source_feature_clipping_valid_for_strict_source_only"] is True


def test_source_feature_clipping_bounds_match_numpy_quantiles() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)

    lower, upper = source_feature_clipping_bounds(source, lower_quantile=0.25, upper_quantile=0.75)

    assert np.allclose(lower, np.quantile(source, 0.25, axis=0))
    assert np.allclose(upper, np.quantile(source, 0.75, axis=0))


def test_apply_feature_clipping_can_modify_in_place() -> None:
    features = np.asarray([[-2.0, 0.0], [2.0, 4.0]], dtype=float)

    clipped = apply_feature_clipping(features, lower_bounds=[0.0, 1.0], upper_bounds=[1.0, 3.0], copy=False)

    assert clipped is features
    assert np.allclose(features, np.asarray([[0.0, 1.0], [1.0, 3.0]]))


def test_source_feature_clipping_config_aliases_and_validation() -> None:
    cfg = source_feature_clipping_config(lower_quantile="0.1", upper_quantile="0.9", copy="false")

    assert cfg.lower_quantile == 0.1
    assert cfg.upper_quantile == 0.9
    assert cfg.copy is False

    with pytest.raises(ValueError, match="lower_quantile"):
        source_feature_clipping_config(lower_quantile=0.9, upper_quantile=0.1)

    with pytest.raises(ValueError, match="copy"):
        source_feature_clipping_config(copy="maybe")


def test_source_feature_clipping_rejects_bool_quantiles() -> None:
    with pytest.raises(ValueError, match="lower_quantile"):
        source_feature_clipping_config(lower_quantile=False, upper_quantile=0.9)

    with pytest.raises(ValueError, match="upper_quantile"):
        source_feature_clipping_config(lower_quantile=0.1, upper_quantile=True)


def test_source_feature_clipping_rejects_array_quantiles() -> None:
    source = np.asarray([[0.0], [1.0]], dtype=float)
    bad_quantiles = [np.asarray([0.1]), np.asarray(0.1)]

    for bad_quantile in bad_quantiles:
        with pytest.raises(ValueError, match="lower_quantile.*scalar"):
            source_feature_clipping_config(lower_quantile=bad_quantile, upper_quantile=0.9)

        with pytest.raises(ValueError, match="upper_quantile.*scalar"):
            source_feature_clipping_bounds(source, lower_quantile=0.1, upper_quantile=bad_quantile)


def test_source_feature_clipping_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_feature_clipping(
            source_features=[[0.0, 1.0]],
            test_features=[[0.0]],
        )


def test_source_feature_clipping_has_no_target_label_api() -> None:
    with pytest.raises(TypeError):
        fit_source_feature_clipping(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            target_labels=[0],  # type: ignore[call-arg]
        )
