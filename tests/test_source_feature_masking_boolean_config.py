from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_masking import (
    augment_source_with_feature_masking,
    source_feature_masking_config,
)


def test_source_feature_masking_parses_quoted_false_preserve_original() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    labels = np.asarray(["x", "x"], dtype=object)

    result = augment_source_with_feature_masking(
        features,
        labels,
        config={"synthetic_per_class": 2, "mask_fraction": 1.0, "preserve_original": "false", "random_state": 3},
    )

    assert result.features.shape == (2, 2)
    assert result.synthetic_mask.tolist() == [True, True]
    assert result.metadata["source_feature_masking_preserve_original"] is False
    assert result.metadata["source_feature_masking_n_output_rows"] == 2


@pytest.mark.parametrize("value", ["", "maybe", 2, -1, 0.5])
def test_source_feature_masking_rejects_ambiguous_preserve_original(value: object) -> None:
    with pytest.raises(ValueError, match="preserve_original"):
        source_feature_masking_config(preserve_original=value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"synthetic_per_class": np.asarray(1)}, "synthetic_per_class"),
        ({"synthetic_per_class": np.array([1])}, "synthetic_per_class"),
        ({"mask_fraction": np.asarray(0.25)}, "mask_fraction"),
        ({"mask_fraction": np.array([0.25])}, "mask_fraction"),
        ({"noise_std": np.asarray(0.01)}, "noise_std"),
        ({"noise_std": np.array([0.01])}, "noise_std"),
    ],
)
def test_source_feature_masking_rejects_array_valued_numeric_config(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source_feature_masking_config(**kwargs)
