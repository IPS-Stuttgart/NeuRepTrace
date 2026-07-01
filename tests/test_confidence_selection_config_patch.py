from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence_selection import ConfidenceSelectionConfig, select_confident_probability_rows


def test_direct_confidence_selection_config_is_normalized_like_mapping_config() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.30],
            [0.99, 0.01],
            [0.60, 0.40],
            [0.20, 0.80],
        ]
    )
    config = ConfidenceSelectionConfig(mode="topk", threshold="0.2", top_k="2", min_margin="0", epsilon="1e-6")

    result = select_confident_probability_rows(probabilities, config=config)

    assert result.selected_indices.tolist() == [1, 3]
    assert result.metadata["confidence_selection_mode"] == "top_k"
    assert result.metadata["confidence_selection_threshold"] == 0.2
    assert result.metadata["confidence_selection_min_margin"] == 0.0
