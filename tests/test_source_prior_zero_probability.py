from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior


def test_source_prior_preserves_zero_probability_columns_for_source_target() -> None:
    probabilities = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=[0, 0, 1],
        classes=[0, 1],
        config={"target_prior": "source", "epsilon": 1e-3},
    )

    assert np.array_equal(result.probabilities, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
