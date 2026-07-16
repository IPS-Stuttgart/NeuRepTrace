from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scaling import SourceFeatureScalingConfig, source_feature_scaling_config


def test_source_feature_scaling_config_preserves_large_exact_integers() -> None:
    large_value = 2**53 + 1

    assert source_feature_scaling_config(random_state=large_value).random_state == large_value
    assert source_feature_scaling_config(random_state=np.uint64(large_value)).random_state == large_value
    assert source_feature_scaling_config(random_state=str(large_value)).random_state == large_value
    assert source_feature_scaling_config(synthetic_per_class=str(large_value)).synthetic_per_class == large_value
    assert source_feature_scaling_config(synthetic_per_class="1e3").synthetic_per_class == 1000


def test_direct_source_feature_scaling_config_preserves_large_exact_integers() -> None:
    large_value = 2**53 + 1

    config = SourceFeatureScalingConfig(
        synthetic_per_class="1e3",  # type: ignore[arg-type]
        random_state=np.uint64(large_value),  # type: ignore[arg-type]
    )

    assert config.synthetic_per_class == 1000
    assert config.random_state == large_value


@pytest.mark.parametrize("value", ["1.5", "9007199254740993.5"])
def test_source_feature_scaling_config_still_rejects_fractional_integer_strings(value: str) -> None:
    with pytest.raises(ValueError, match="random_state"):
        source_feature_scaling_config(random_state=value)
