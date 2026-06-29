from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence_filter import confidence_filter


def test_confidence_filter_uses_lowest_index_for_tied_probabilities() -> None:
    result = confidence_filter(np.asarray([[0.5, 0.5, 0.0], [0.2, 0.8, 0.8]], dtype=float))

    assert result.predicted_index.tolist() == [0, 1]
    assert np.allclose(result.margin, [0.0, 0.0])
