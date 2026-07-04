from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clip import SourceClipConfig, source_clip_config


def test_source_clip_config_direct_construction_normalizes_fields() -> None:
    cfg = SourceClipConfig(lower_quantile="0.05", upper_quantile=np.asarray(0.95), symmetric="yes", center="avg")

    assert cfg.lower_quantile == pytest.approx(0.05)
    assert cfg.upper_quantile == pytest.approx(0.95)
    assert cfg.symmetric is True
    assert cfg.center == "mean"

    default_center = SourceClipConfig(center=None)
    assert default_center.center == "median"


def test_source_clip_config_accepts_scalar_array_booleans() -> None:
    yes_cfg = SourceClipConfig(symmetric=np.asarray("yes"))
    no_cfg = source_clip_config(symmetric=np.asarray("off"))

    assert yes_cfg.symmetric is True
    assert no_cfg.symmetric is False


def test_source_clip_config_factory_matches_direct_validation() -> None:
    direct = SourceClipConfig(lower_quantile="0.10", upper_quantile="0.90", symmetric=1, center="none")
    via_factory = source_clip_config(lower_quantile="0.10", upper_quantile="0.90", symmetric=1, center="none")

    assert direct == via_factory
    assert direct.center == "zero"
    assert direct.symmetric is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lower_quantile": -0.1}, "lower_quantile"),
        ({"upper_quantile": 1.1}, "upper_quantile"),
        ({"lower_quantile": 0.9, "upper_quantile": 0.1}, "lower_quantile must be smaller"),
        ({"lower_quantile": np.nan}, "lower_quantile"),
        ({"symmetric": "sometimes"}, "symmetric"),
        ({"symmetric": np.asarray([True, False])}, "symmetric"),
        ({"center": "middle"}, "Unknown center mode"),
    ],
)
def test_source_clip_config_direct_construction_rejects_invalid_fields(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SourceClipConfig(**kwargs)
