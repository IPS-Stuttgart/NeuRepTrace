from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from neureptrace.decoding.source_clipping import SourceFeatureClippingConfig, fit_source_feature_clipping, source_feature_clipping_config


def test_source_clipping_revalidates_direct_dataclass_config() -> None:
    cfg = SourceFeatureClippingConfig(
        lower_quantile=cast(float, "0.0"),
        upper_quantile=cast(float, "1.0"),
        copy=cast(bool, "false"),
    )

    source = [[0.0], [1.0], [2.0]]
    test = [[-1.0], [3.0]]

    result = fit_source_feature_clipping(source_features=source, test_features=test, config=cfg)

    assert result.metadata["source_feature_clipping_lower_quantile"] == 0.0
    assert result.metadata["source_feature_clipping_upper_quantile"] == 1.0
    assert result.test_features[0, 0] == 0.0
    assert result.test_features[1, 0] == 2.0


def test_source_clipping_accepts_scalar_numpy_config_values() -> None:
    cfg = source_feature_clipping_config(
        lower_quantile=np.asarray(0.0),
        upper_quantile=np.float64(1.0),
        copy=np.asarray(False),
    )

    assert cfg.lower_quantile == 0.0
    assert cfg.upper_quantile == 1.0
    assert cfg.copy is False


@pytest.mark.parametrize("bad_quantile", [np.asarray([0.1]), np.asarray(True), np.bool_(False)])
def test_source_clipping_rejects_ambiguous_numpy_quantiles(bad_quantile) -> None:
    with pytest.raises(ValueError, match="lower_quantile"):
        source_feature_clipping_config(lower_quantile=bad_quantile)


def test_source_clipping_rejects_vector_numpy_copy() -> None:
    with pytest.raises(ValueError, match="copy"):
        source_feature_clipping_config(copy=np.asarray([False]))
