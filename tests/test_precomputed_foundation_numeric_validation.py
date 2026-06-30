from __future__ import annotations

import pytest

from neureptrace.decoding.precomputed_foundation import (
    fit_precomputed_foundation_probe,
    make_precomputed_foundation_feature_table,
)


def test_precomputed_foundation_probe_rejects_list_classifier_strength() -> None:
    table = make_precomputed_foundation_feature_table(
        [[-2.0], [-1.0], [1.0], [2.0], [0.0]],
        row_ids=["a", "b", "c", "d", "target"],
    )

    with pytest.raises(ValueError, match="classifier_C"):
        fit_precomputed_foundation_probe(
            feature_table=table,
            train_row_ids=["a", "b", "c", "d"],
            train_labels=[0, 0, 1, 1],
            test_row_ids=["target"],
            classifier_C=[1.0],
        )
