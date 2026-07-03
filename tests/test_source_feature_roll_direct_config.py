from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_roll import (
    SourceFeatureRollConfig,
    augment_source_with_feature_roll,
)


def test_source_feature_roll_direct_config_normalizes_scalars() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]], dtype=float)
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)

    config = SourceFeatureRollConfig(
        synthetic_per_class="1",
        max_shift=np.asarray(1),
        roll_mode="zero-pad",
        fill_value="-2.5",
        include_zero_shift="true",
        preserve_original="false",
        random_state="none",
    )

    assert config.synthetic_per_class == 1
    assert config.max_shift == 1
    assert config.roll_mode == "constant"
    assert config.fill_value == -2.5
    assert config.include_zero_shift is True
    assert config.preserve_original is False
    assert config.random_state is None

    result = augment_source_with_feature_roll(features, labels, config=config)

    assert result.features.shape == (2, 2)
    assert result.synthetic_mask.tolist() == [True, True]
    assert result.metadata["source_feature_roll_n_output_rows"] == 2
    assert result.metadata["source_feature_roll_random_state"] == ""


def test_source_feature_roll_direct_config_rejects_vector_scalars() -> None:
    with pytest.raises(ValueError, match="synthetic_per_class"):
        SourceFeatureRollConfig(synthetic_per_class=np.asarray([1, 2]))
