from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.bushmeg_artifact_diff import compare_prediction_frames, compare_summary_frames


def test_compare_summary_frames_reports_fold_and_mean_deltas():
    reference = pd.DataFrame({"outer_test_subject": ["s1", "s2"], "balanced_accuracy": [0.10, 0.20], "top2_accuracy": [0.20, 0.30]})
    candidate = pd.DataFrame({"outer_test_subject": ["s1", "s2"], "balanced_accuracy": [0.15, 0.18], "top2_accuracy": [0.22, 0.31]})

    diff = compare_summary_frames(reference, candidate, metrics=("balanced_accuracy", "top2_accuracy"))

    mean_ba = diff[(diff["outer_test_subject"] == "__mean__") & (diff["metric"] == "balanced_accuracy")].iloc[0]
    assert np.isclose(mean_ba["delta_candidate_minus_reference"], 0.015)


def test_compare_prediction_frames_reports_label_mismatches_and_recall_deltas():
    reference = pd.DataFrame(
        {
            "outer_test_subject": ["s1", "s1", "s1", "s1"],
            "trial_index": [0, 1, 2, 3],
            "true_label": [0, 0, 1, 1],
            "predicted_label": [0, 1, 1, 0],
        }
    )
    candidate = pd.DataFrame(
        {
            "outer_test_subject": ["s1", "s1", "s1", "s1"],
            "trial_index": [0, 1, 2, 3],
            "true_label": [0, 0, 1, 1],
            "predicted_label": [0, 0, 1, 1],
        }
    )

    diagnostics, per_class = compare_prediction_frames(reference, candidate)

    assert int(diagnostics.loc[diagnostics["diagnostic"] == "true_label_mismatch_rows", "value"].iloc[0]) == 0
    class0 = per_class[(per_class["outer_test_subject"] == "s1") & (per_class["true_class"] == 0)].iloc[0]
    class1 = per_class[(per_class["outer_test_subject"] == "s1") & (per_class["true_class"] == 1)].iloc[0]
    assert np.isclose(class0["delta_candidate_minus_reference"], 0.5)
    assert np.isclose(class1["delta_candidate_minus_reference"], 0.5)
