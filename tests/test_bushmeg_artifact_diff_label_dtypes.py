from __future__ import annotations

import pandas as pd

from neureptrace.bushmeg_artifact_diff import compare_prediction_frames


def test_compare_prediction_frames_accepts_equivalent_numeric_label_dtypes() -> None:
    reference = pd.DataFrame(
        {
            "outer_test_subject": ["s1", "s1"],
            "trial_index": [0, 1],
            "true_label": [0, 1],
            "predicted_label": [0, 1],
        }
    )
    candidate = pd.DataFrame(
        {
            "outer_test_subject": ["s1", "s1"],
            "trial_index": [0, 1],
            "true_label": [0.0, 1.0],
            "predicted_label": [0.0, 1.0],
        }
    )

    diagnostics, _ = compare_prediction_frames(reference, candidate)
    values = dict(zip(diagnostics["diagnostic"], diagnostics["value"], strict=True))

    assert values["matched_prediction_rows"] == 2
    assert values["true_label_mismatch_rows"] == 0
