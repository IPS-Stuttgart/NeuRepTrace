from __future__ import annotations

import pandas as pd

from neureptrace.decoding.source_knn import fit_source_knn_reference


def test_source_knn_treats_indeterminate_label_equality_as_distinct() -> None:
    reference = fit_source_knn_reference(
        source_features=[[0.0], [1.0], [2.0]],
        source_labels=["left", pd.NA, "right"],
        config={"standardize": False},
    )

    assert reference.classes.shape[0] == 3
    assert reference.labels.tolist()[1] is pd.NA
