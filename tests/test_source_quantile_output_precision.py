from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_quantile import source_quantile_clip, source_quantile_rank


def test_source_quantile_clip_preserves_tiny_nonzero_features() -> None:
    source = np.asarray([[1.0e-50], [2.0e-50], [3.0e-50], [4.0e-50]])
    test = np.asarray([[1.5e-50], [3.5e-50]])

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = source_quantile_clip(
            source_features=source,
            test_features=test,
            lower=0.0,
            upper=1.0,
        )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    np.testing.assert_array_equal(result.train_features, source)
    np.testing.assert_array_equal(result.test_features, test)


def test_source_quantile_rank_preserves_tiny_positive_epsilon() -> None:
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = source_quantile_rank(
            source_features=[[0.0], [1.0]],
            test_features=[[-1.0]],
            epsilon=1.0e-50,
        )

    assert result.test_features.dtype == np.float64
    assert result.test_features[0, 0] == 1.0e-50


def test_source_quantile_outputs_keep_float32_for_representable_values() -> None:
    clipped = source_quantile_clip(
        source_features=[[0.0], [1.0]],
        test_features=[[0.5]],
        lower=0.0,
        upper=1.0,
    )
    ranked = source_quantile_rank(
        source_features=[[0.0], [1.0]],
        test_features=[[0.5]],
        epsilon=0.125,
    )

    assert clipped.train_features.dtype == np.float32
    assert clipped.test_features.dtype == np.float32
    assert ranked.train_features.dtype == np.float32
    assert ranked.test_features.dtype == np.float32
