from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_balancing import balance_source_classes, source_class_balancing_config


def test_undersample_mode_defaults_to_minority_class_count() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0], [12.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1, 1], dtype=object)

    cfg = source_class_balancing_config(mode="undersample")
    result = balance_source_classes(features, labels, config={"mode": "undersample", "random_state": 3})

    assert cfg.target_count == "min"
    assert result.features.shape == (4, 1)
    assert result.class_counts_after == {0: 2, 1: 2}
    assert not np.any(result.synthetic_mask)
    assert result.metadata["source_class_balancing_target_count"] == 2


def test_oversample_and_weight_modes_keep_majority_default() -> None:
    assert source_class_balancing_config(mode="oversample").target_count == "max"
    assert source_class_balancing_config(mode="weights").target_count == "max"
