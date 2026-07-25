from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_balancing import SourceClassBalancingConfig, balance_source_classes


def test_direct_source_class_balancing_config_is_normalized_before_use() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0], [12.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1, 1], dtype=object)

    result = balance_source_classes(
        features,
        labels,
        config=SourceClassBalancingConfig(
            mode="under",
            target_count="min",
            random_state="7",  # type: ignore[arg-type]
            preserve_order="false",  # type: ignore[arg-type]
        ),
    )

    assert result.class_counts_after == {0: 2, 1: 2}
    assert result.metadata["source_class_balancing_mode"] == "undersample"
    assert result.metadata["source_class_balancing_random_state"] == 7
    assert result.metadata["source_class_balancing_preserve_order"] is False
    assert not np.any(result.synthetic_mask)
