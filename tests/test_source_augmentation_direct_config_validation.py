from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_jitter import SourceFeatureJitterConfig
from neureptrace.decoding.source_masking import SourceFeatureMaskingConfig


def test_source_feature_jitter_config_normalizes_direct_dataclass_values() -> None:
    cfg = SourceFeatureJitterConfig(
        synthetic_per_class="2",  # type: ignore[arg-type]
        noise_scale="0.25",  # type: ignore[arg-type]
        scale_mode="classwise",
        preserve_original="0",  # type: ignore[arg-type]
        random_state=np.asarray("none"),
        epsilon="0.001",  # type: ignore[arg-type]
    )

    assert cfg.synthetic_per_class == 2
    assert isinstance(cfg.synthetic_per_class, int)
    assert cfg.noise_scale == pytest.approx(0.25)
    assert cfg.scale_mode == "class"
    assert cfg.preserve_original is False
    assert cfg.random_state is None
    assert cfg.epsilon == pytest.approx(0.001)
    assert cfg.enabled is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"synthetic_per_class": True}, "synthetic_per_class must be an integer"),
        ({"noise_scale": -0.1}, "noise_scale must be non-negative and finite"),
        ({"scale_mode": "bad"}, "Unknown scale_mode"),
        ({"preserve_original": np.asarray([True])}, "preserve_original must be a boolean value"),
        ({"epsilon": 0}, "epsilon must be positive and finite"),
    ],
)
def test_source_feature_jitter_config_rejects_invalid_direct_dataclass_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SourceFeatureJitterConfig(**kwargs)  # type: ignore[arg-type]


def test_source_feature_masking_config_normalizes_direct_dataclass_values() -> None:
    cfg = SourceFeatureMaskingConfig(
        synthetic_per_class="3",  # type: ignore[arg-type]
        mask_fraction="0.5",  # type: ignore[arg-type]
        mask_mode="contiguous",
        block_size=np.asarray("none"),
        fill_mode="mean",
        noise_std="0.1",  # type: ignore[arg-type]
        preserve_original="false",  # type: ignore[arg-type]
        random_state=np.asarray("none"),
    )

    assert cfg.synthetic_per_class == 3
    assert isinstance(cfg.synthetic_per_class, int)
    assert cfg.mask_fraction == pytest.approx(0.5)
    assert cfg.mask_mode == "block"
    assert cfg.block_size is None
    assert cfg.fill_mode == "feature_mean"
    assert cfg.noise_std == pytest.approx(0.1)
    assert cfg.preserve_original is False
    assert cfg.random_state is None
    assert cfg.enabled is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"synthetic_per_class": True}, "synthetic_per_class must be an integer"),
        ({"mask_fraction": 1.1}, r"mask_fraction must be in \[0, 1\]"),
        ({"mask_mode": "bad"}, "Unknown mask_mode"),
        ({"block_size": 0}, "block_size must be a positive integer or None"),
        ({"fill_mode": "bad"}, "Unknown fill_mode"),
        ({"preserve_original": np.asarray([False])}, "preserve_original must be a boolean value"),
        ({"random_state": -1}, "random_state must be a non-negative integer or None"),
    ],
)
def test_source_feature_masking_config_rejects_invalid_direct_dataclass_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SourceFeatureMaskingConfig(**kwargs)  # type: ignore[arg-type]
