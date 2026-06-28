from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scaling import (
    augment_source_with_feature_scaling,
    source_feature_scaling_config,
)


def test_source_feature_scaling_parses_quoted_false_preserve_original() -> None:
    features = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    labels = np.asarray(["x", "x"], dtype=object)

    result = augment_source_with_feature_scaling(
        features,
        labels,
        config={"synthetic_per_class": 3, "preserve_original": "false", "random_state": 3},
    )

    assert result.features.shape == (3, 2)
    assert result.synthetic_mask.tolist() == [True, True, True]
    assert result.metadata["source_feature_scaling_preserve_original"] is False
    assert result.metadata["source_feature_scaling_n_output_rows"] == 3


@pytest.mark.parametrize("value", ["", "maybe", 2, -1, 0.5])
def test_source_feature_scaling_rejects_ambiguous_preserve_original(value: object) -> None:
    with pytest.raises(ValueError, match="preserve_original"):
        source_feature_scaling_config(preserve_original=value)
