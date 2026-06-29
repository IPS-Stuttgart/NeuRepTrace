from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.ranking import rank_class_scores


@pytest.mark.parametrize(
    "bad_class_column",
    ["", np.array(["class"]), ["class"], None, 1],
)
def test_rank_class_scores_rejects_invalid_class_column_names(bad_class_column: object) -> None:
    with pytest.raises(ValueError, match="class_column must be a non-empty string"):
        rank_class_scores(
            [[0.8, 0.2]],
            ["target", "distractor"],
            ["target"],
            top_k=(1,),
            row_top_k=1,
            class_column=bad_class_column,
        )
