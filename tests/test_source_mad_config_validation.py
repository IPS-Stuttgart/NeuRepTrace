from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mad import SourceMADConfig, fit_source_mad_transform


def test_source_mad_dataclass_normalizes_string_booleans_before_use() -> None:
    source_features = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
        ],
        dtype=float,
    )
    test_features = np.asarray([[7.0, 22.0]], dtype=float)
    config = SourceMADConfig(
        center="false",
        scale="false",
        normal_consistency="false",
        epsilon="1e-6",
    )

    result = fit_source_mad_transform(
        source_features=source_features,
        test_features=test_features,
        config=config,
    )

    np.testing.assert_allclose(result.train_features, source_features.astype(np.float32))
    np.testing.assert_allclose(result.test_features, test_features.astype(np.float32))
    assert result.reference.config.center is False
    assert result.reference.config.scale is False
    assert result.reference.config.normal_consistency is False
    assert result.reference.config.epsilon == 1e-6
    assert result.metadata["source_mad_center"] is False
    assert result.metadata["source_mad_scale"] is False
    assert result.metadata["source_mad_normal_consistency"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center": "sometimes"},
        {"scale": 2},
        {"normal_consistency": object()},
        {"epsilon": True},
        {"epsilon": np.asarray([1.0])},
        {"epsilon": 0.0},
    ],
)
def test_source_mad_dataclass_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SourceMADConfig(**kwargs)
