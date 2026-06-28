from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_smote import augment_source_with_smote, source_smote_config


def test_source_smote_config_parses_string_boolean_tokens() -> None:
    cfg = source_smote_config(cross_domain_partner="false", preserve_original="0")

    assert cfg.cross_domain_partner is False
    assert cfg.preserve_original is False


def test_source_smote_config_rejects_invalid_boolean_tokens() -> None:
    with pytest.raises(ValueError, match="cross_domain_partner must be a boolean"):
        source_smote_config(cross_domain_partner="maybe")

    with pytest.raises(ValueError, match="preserve_original must be a boolean"):
        source_smote_config(preserve_original="maybe")


def test_source_smote_mapping_config_honors_preserve_original_false() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "b", "b"])

    result = augment_source_with_smote(
        features,
        labels,
        config={
            "synthetic_per_class": 1,
            "cross_domain_partner": "false",
            "preserve_original": "false",
            "random_state": 0,
        },
    )

    assert result.features.shape == (2, 2)
    assert result.labels.tolist() == ["a", "b"]
    assert result.synthetic_mask.tolist() == [True, True]
    assert result.metadata["source_smote_preserve_original"] is False
    assert result.metadata["source_smote_n_output_rows"] == 2
