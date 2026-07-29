from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import rank_class_scores


@pytest.mark.parametrize("top_k", [None, 2, np.array(2, dtype=int)])
def test_rank_class_scores_rejects_non_iterable_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a sequence of integers"):
        rank_class_scores(
            [[0.8, 0.2], [0.3, 0.7]],
            ["left", "right"],
            ["left", "right"],
            top_k=top_k,
        )
