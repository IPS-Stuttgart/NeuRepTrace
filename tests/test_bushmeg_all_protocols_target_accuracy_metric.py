from __future__ import annotations

import pandas as pd

from neureptrace.bushmeg_all_protocols_audit import build_audit_markdown


def test_target_accuracy_metric_column_does_not_fail_audit(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "method": "calibrated_method",
                "protocol_category": 3,
                "uses_target_data": True,
                "uses_target_labels_for_fitting": True,
                "calibration_rows_disjoint_from_evaluation": True,
                "valid_for_zero_calibration": False,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
                "outer_test_subject": "1",
                "fold_index": 1,
                "k_per_class": 1,
                "target_calibration_per_class": 1,
                "n_target_calibration_trials": 2,
                "n_target_evaluation_trials": 4,
                "n_classes": 2,
                "balanced_accuracy": 0.55,
                "target_accuracy": 0.73,
                "target_calibration_indices": "0|2",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "method": "calibrated_method",
                "protocol_category": 3,
                "outer_test_subject": "1",
                "fold_index": 1,
                "target_calibration_per_class": 1,
                "n_target_calibration_trials": 2,
                "n_target_evaluation_trials": 4,
                "target_row_index": 1,
                "trial_index": 1,
                "is_calibration_row": False,
            }
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "method": "calibrated_method",
                "protocol_category": 3,
                "status": "runnable",
            }
        ]
    )
    summary.to_csv(tmp_path / "summary.csv", index=False)
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    metadata.to_csv(tmp_path / "method_metadata.csv", index=False)

    audit_path = build_audit_markdown(results_dir=tmp_path, out_path=tmp_path / "audit.md", include_calibrated=True)
    text = audit_path.read_text(encoding="utf-8")

    assert "[PASS] No method selected hyperparameters using held-out target accuracy" in text
    assert "Target-accuracy-related columns inspected: target_accuracy" in text
