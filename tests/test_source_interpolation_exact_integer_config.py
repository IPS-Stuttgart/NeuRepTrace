from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mixup import SourceMixUpConfig, augment_source_with_mixup, source_mixup_config
from neureptrace.decoding.source_smote import SourceSmoteConfig, source_smote_config


@pytest.mark.parametrize("factory", [source_mixup_config, source_smote_config])
def test_source_interpolation_configs_preserve_large_exact_integers(factory) -> None:
    large_value = 2**53 + 1

    for value in (large_value, np.uint64(large_value), str(large_value)):
        config = factory(synthetic_per_class=value, random_state=value)

        assert config.synthetic_per_class == large_value
        assert config.random_state == large_value

    assert factory(synthetic_per_class="1e3").synthetic_per_class == 1000


def test_direct_source_smote_config_preserves_large_exact_integers() -> None:
    large_value = 2**53 + 1

    config = SourceSmoteConfig(
        synthetic_per_class=np.uint64(large_value),  # type: ignore[arg-type]
        random_state=str(large_value),
    )

    assert config.synthetic_per_class == large_value
    assert config.random_state == large_value


def test_direct_source_mixup_config_is_revalidated_without_precision_loss() -> None:
    large_value = 2**53 + 1
    config = SourceMixUpConfig(
        synthetic_per_class=0,
        random_state=np.uint64(large_value),  # type: ignore[arg-type]
    )

    result = augment_source_with_mixup(
        np.asarray([[0.0, 1.0]], dtype=float),
        np.asarray(["a"]),
        config=config,
    )

    assert result.metadata["source_mixup_random_state"] == large_value


@pytest.mark.parametrize("factory", [source_mixup_config, source_smote_config])
def test_source_interpolation_configs_reject_large_fractional_integer_strings(factory) -> None:
    with pytest.raises(ValueError, match="synthetic_per_class must be an integer"):
        factory(synthetic_per_class="9007199254740993.5")

    with pytest.raises(ValueError, match="random_state"):
        factory(random_state="9007199254740993.5")
