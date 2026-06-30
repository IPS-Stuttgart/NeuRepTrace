from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_jitter import SourceFeatureJitterConfig, augment_source_with_feature_jitter
from neureptrace.decoding.source_masking import SourceFeatureMaskingConfig, augment_source_with_feature_masking


def _small_source_data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=float),
        np.asarray(["a", "a"], dtype=object),
    )


@pytest.mark.parametrize(
    ("field", "config"),
    [
        ("synthetic_per_class", SourceFeatureJitterConfig(synthetic_per_class=np.asarray([1]))),
        ("noise_scale", SourceFeatureJitterConfig(noise_scale=np.asarray([0.05]))),
        ("epsilon", SourceFeatureJitterConfig(epsilon=np.asarray([1e-8]))),
    ],
)
def test_source_jitter_revalidates_dataclass_config(field: str, config: SourceFeatureJitterConfig) -> None:
    features, labels = _small_source_data()

    with pytest.raises(ValueError, match=field):
        augment_source_with_feature_jitter(features, labels, config=config)


@pytest.mark.parametrize(
    ("field", "config"),
    [
        ("synthetic_per_class", SourceFeatureMaskingConfig(synthetic_per_class=np.asarray([1]))),
        ("mask_fraction", SourceFeatureMaskingConfig(mask_fraction=np.asarray([0.25]))),
        ("block_size", SourceFeatureMaskingConfig(block_size=np.asarray([1]))),
        ("noise_std", SourceFeatureMaskingConfig(noise_std=np.asarray([0.01]))),
    ],
)
def test_source_masking_revalidates_dataclass_config(field: str, config: SourceFeatureMaskingConfig) -> None:
    features, labels = _small_source_data()

    with pytest.raises(ValueError, match=field):
        augment_source_with_feature_masking(features, labels, config=config)
