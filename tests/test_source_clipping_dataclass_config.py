from __future__ import annotations

from typing import cast

from neureptrace.decoding.source_clipping import SourceFeatureClippingConfig, fit_source_feature_clipping


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
