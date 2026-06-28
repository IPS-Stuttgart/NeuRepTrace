from __future__ import annotations

import pandas as pd

import neureptrace.bushmeg_all_protocols_audit as audit


def _summary_with_list_values() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "source_plus_target_calibration_logistic_k2",
                "method_family": "source_plus_target_calibration",
                "protocol_category": 3,
                "uses_target_data": [True, True],
                "uses_target_labels_for_fitting": [True, True],
                "calibration_rows_disjoint_from_evaluation": [True, True],
                "valid_for_zero_calibration": [False, False],
                "valid_for_strict_source_only": [False, False],
                "debug_upper_bound": [False, False],
                "outer_test_subject": "1",
                "fold_index": 1,
                "k_per_class": [2, 2],
                "target_calibration_per_class": [2, 2],
                "n_target_calibration_trials": [6, 6],
                "n_target_evaluation_trials": [12, 12],
                "n_classes": [3, 3],
                "balanced_accuracy": 0.55,
                "accuracy": 0.56,
                "target_calibration_indices": [0, 2, 4, 6, 8, 10],
            }
        ]
    )


def _predictions_with_list_values(*, target_row_index: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "source_plus_target_calibration_logistic_k2",
                "protocol_category": 3,
                "outer_test_subject": "1",
                "fold_index": 1,
                "target_calibration_per_class": [2, 2],
                "n_target_calibration_trials": [6, 6],
                "n_target_evaluation_trials": [12, 12],
                "target_row_index": target_row_index,
                "trial_index": target_row_index,
                "is_calibration_row": [False, False],
                "true_label": 0,
                "predicted_label": 0,
            }
        ]
    )


def test_protocol3_audit_accepts_list_valued_metadata() -> None:
    summary = _summary_with_list_values()
    predictions = _predictions_with_list_values(target_row_index=[1, 1])

    assert audit._protocol3_summary_failures(summary) == []
    assert audit._protocol3_calibration_count_failures(summary) == []
    assert audit._protocol3_prediction_failures(summary, predictions) == []


def test_protocol3_audit_detects_list_valued_calibration_overlap() -> None:
    summary = _summary_with_list_values()
    predictions = _predictions_with_list_values(target_row_index=[2, 2])

    failures = audit._protocol3_prediction_failures(summary, predictions)

    assert any("Protocol 3 prediction uses calibration row 2" in failure for failure in failures)
