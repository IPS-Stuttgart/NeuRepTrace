from __future__ import annotations

import numpy as np

from neureptrace._object_label_utils import values_equal
from neureptrace.decoding.source_knn import fit_source_knn_decoder


def test_source_knn_groups_and_predicts_missing_labels() -> None:
    result = fit_source_knn_decoder(
        source_features=[[0.0], [1.0], [10.0], [11.0]],
        source_labels=[float("nan"), np.float64("nan"), "right", "right"],
        test_features=[[0.1], [10.9]],
        config={"k": 1, "standardize": False},
    )

    assert result.classes.shape == (2,)
    assert values_equal(result.classes[0], np.nan)
    assert result.classes[1] == "right"
    assert values_equal(result.predictions[0], np.nan)
    assert result.predictions[1] == "right"
    np.testing.assert_allclose(result.probabilities, [[1.0, 0.0], [0.0, 1.0]])
