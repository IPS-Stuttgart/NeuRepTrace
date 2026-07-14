from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.bushmeg_artifact_diff import per_class_recall_frame


def test_per_class_recall_preserves_numeric_equality_after_abstention_coercion() -> None:
    predictions = pd.DataFrame(
        {
            "outer_test_subject": ["s1", "s1", "s1"],
            "true_label": [0, 1, 1],
            # A missing prediction makes pandas represent the whole numeric column
            # as floating point after a CSV round trip.
            "predicted_label": [0.0, 1.0, np.nan],
        }
    )

    recall = per_class_recall_frame(predictions)
    rows = {int(row["true_class"]): row for row in recall.to_dict("records")}

    assert rows[0]["n_correct"] == 1
    assert np.isclose(rows[0]["recall"], 1.0)
    assert rows[1]["n_correct"] == 1
    assert np.isclose(rows[1]["recall"], 0.5)
