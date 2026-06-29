from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.bushmeg_artifact_diff import compare_prediction_frames, compare_summary_frames, main


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


def test_compare_prediction_frames_aligns_recall_classes_after_csv_dtype_roundtrip():
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
            "true_label": ["0", "1"],
            "predicted_label": ["0", "0"],
        }
    )

    _, per_class = compare_prediction_frames(reference, candidate)
    class_rows = {str(row["true_class"]): row for row in per_class.to_dict("records")}

    assert set(class_rows) == {"0", "1"}
    assert not per_class[["reference_recall", "candidate_recall"]].isna().any().any()
    assert np.isclose(class_rows["0"]["reference_recall"], 1.0)
    assert np.isclose(class_rows["0"]["candidate_recall"], 1.0)
    assert np.isclose(class_rows["1"]["reference_recall"], 1.0)
    assert np.isclose(class_rows["1"]["candidate_recall"], 0.0)


def test_compare_prediction_frames_does_not_count_unmatched_rows_as_true_label_mismatches():
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
            "trial_index": [0, 2],
            "true_label": [0, 1],
            "predicted_label": [0, 1],
        }
    )

    diagnostics, _ = compare_prediction_frames(reference, candidate)
    diagnostic_values = dict(zip(diagnostics["diagnostic"], diagnostics["value"], strict=True))

    assert diagnostic_values["matched_prediction_rows"] == 1
    assert diagnostic_values["reference_only_rows"] == 1
    assert diagnostic_values["candidate_only_rows"] == 1
    assert diagnostic_values["true_label_mismatch_rows"] == 0


def test_compare_prediction_frames_uses_only_common_recall_group_columns():
    reference = pd.DataFrame(
        {
            "outer_test_subject": ["s1", "s1", "s2", "s2"],
            "trial_index": [0, 1, 2, 3],
            "true_label": [0, 1, 0, 1],
            "predicted_label": [0, 1, 1, 1],
        }
    )
    candidate = pd.DataFrame(
        {
            "trial_index": [0, 1, 2, 3],
            "true_label": [0, 1, 0, 1],
            "predicted_label": [0, 0, 0, 1],
        }
    )

    _, per_class = compare_prediction_frames(reference, candidate, group_columns=("outer_test_subject",))

    assert "outer_test_subject" not in per_class.columns
    assert set(per_class["true_class"].tolist()) == {0, 1}
    class0 = per_class.loc[per_class["true_class"] == 0].iloc[0]
    assert np.isclose(class0["reference_recall"], 0.5)
    assert np.isclose(class0["candidate_recall"], 1.0)


def test_artifact_diff_cli_creates_prediction_output_parents(tmp_path: Path):
    reference_summary = tmp_path / "reference_summary.csv"
    candidate_summary = tmp_path / "candidate_summary.csv"
    reference_predictions = tmp_path / "reference_predictions.csv"
    candidate_predictions = tmp_path / "candidate_predictions.csv"

    pd.DataFrame(
        {
            "outer_test_subject": ["s1"],
            "balanced_accuracy": [0.10],
            "accuracy": [0.10],
            "top2_accuracy": [0.20],
            "top3_accuracy": [0.30],
            "log_loss": [1.5],
        }
    ).to_csv(reference_summary, index=False)
    pd.DataFrame(
        {
            "outer_test_subject": ["s1"],
            "balanced_accuracy": [0.20],
            "accuracy": [0.20],
            "top2_accuracy": [0.30],
            "top3_accuracy": [0.40],
            "log_loss": [1.0],
        }
    ).to_csv(candidate_summary, index=False)
    pd.DataFrame(
        {
            "outer_test_subject": ["s1"],
            "trial_index": [0],
            "true_label": [0],
            "predicted_label": [0],
        }
    ).to_csv(reference_predictions, index=False)
    pd.DataFrame(
        {
            "outer_test_subject": ["s1"],
            "trial_index": [0],
            "true_label": [0],
            "predicted_label": [0],
        }
    ).to_csv(candidate_predictions, index=False)

    summary_out = tmp_path / "nested" / "summary" / "diff.csv"
    diagnostics_out = tmp_path / "nested" / "predictions" / "diagnostics.csv"
    per_class_out = tmp_path / "nested" / "classes" / "recall.csv"

    assert main(
        [
            str(reference_summary),
            str(candidate_summary),
            "--reference-predictions",
            str(reference_predictions),
            "--candidate-predictions",
            str(candidate_predictions),
            "--summary-out",
            str(summary_out),
            "--prediction-diagnostics-out",
            str(diagnostics_out),
            "--per-class-out",
            str(per_class_out),
        ]
    ) == 0

    assert summary_out.is_file()
    assert diagnostics_out.is_file()
    assert per_class_out.is_file()
