from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.geodesic_flow import geodesic_flow_config, sample_geodesic_bases, transform_with_geodesic_bases


def test_geodesic_flow_config_normalizes_boolean_like_strings() -> None:
    cfg = geodesic_flow_config(
        center="false",
        scale="0",
        include_endpoints="off",
        normalize_blocks="no",
    )

    assert cfg.center is False
    assert cfg.scale is False
    assert cfg.include_endpoints is False
    assert cfg.normalize_blocks is False


def test_geodesic_flow_config_rejects_ambiguous_boolean_strings() -> None:
    with pytest.raises(ValueError, match="center must be a boolean value"):
        geodesic_flow_config(center="maybe")


def test_geodesic_flow_runtime_boolean_options_normalize_strings() -> None:
    bases = sample_geodesic_bases(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        n_steps=2,
        include_endpoints="false",
    )
    assert [basis.position for basis in bases] == pytest.approx([1 / 3, 2 / 3])

    features = np.asarray([[1.0, 2.0]], dtype=float)
    transformed = transform_with_geodesic_bases(features, bases, normalize_blocks="false")
    assert np.allclose(transformed, np.asarray([[1.0, 2.0, 1.0, 2.0]]))
