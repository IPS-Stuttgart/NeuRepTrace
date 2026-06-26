from __future__ import annotations

import pandas as pd

from neureptrace.bushmeg_all_protocols_audit import build_audit_markdown


def _valid_protocol3_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "source_plus_target_calibration_logistic_k2",
                "method_family": "source_plus_target_calibration",
                "protocol_category": 3,
                "uses_target_data": True,
                "uses_target_labels_for_fitting": True,
                "calibration_rows_disjoint_from_evaluation": True,
                "valid_for_zero_calibration": False,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
                "outer_test_subject": "1",
                "fold_index": 1,
                "k_per_class": 2,
                "target_calibration_per_class": 2,
                "n_target_calibration_trials": 6,
                "n_target_evaluation_trials": 12,
                "n_classes": 3,
                "balanced_accuracy": 0.55,
                "accuracy": 0.56,
                "target_calibration_indices": "0|2|4|6|8|10",
            }
        ]
    )


def _valid_protocol3_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "source_plus_target_calibration_logistic_k2",
                "protocol_category": 3,
                "outer_test_subject": "1",
                "fold_index": 1,
                "target_calibration_per_class": 2,
                "n_target_calibration_trials": 6,
                "n_target_evaluation_trials": 12,
                "target_row_index": 1,
                "trial_index": 1,
                "is_calibration_row": False,
                "true_label": 0,
                "predicted_label": 0,
            },
            {
                "method": "source_plus_target_calibration_logistic_k2",
                "protocol_category": 3,
                "outer_test_subject": "1",
                "fold_index": 1,
                "target_calibration_per_class": 2,
                "n_target_calibration_trials": 6,
                "n_target_evaluation_trials": 12,
                "target_row_index": 3,
                "trial_index": 3,
                "is_calibration_row": False,
                "true_label": 1,
                "predicted_label": 1,
            },
        ]
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "source_plus_target_calibration_logistic_k2",
                "protocol_category": 3,
                "method_family": "source_plus_target_calibration",
                "status": "runnable",
                "uses_target_data": True,
                "uses_target_labels_for_fitting": True,
                "calibration_rows_disjoint_from_evaluation": True,
                "valid_for_zero_calibration": False,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
            }
        ]
    )


def _write_artifacts(
    tmp_path,
    *,
    summary: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    leaderboard: pd.DataFrame | None = None,
) -> None:
    (summary if summary is not None else _valid_protocol3_summary()).to_csv(tmp_path / "summary.csv", index=False)
    (predictions if predictions is not None else _valid_protocol3_predictions()).to_csv(tmp_path / "predictions.csv", index=False)
    (metadata if metadata is not None else _metadata()).to_csv(tmp_path / "method_metadata.csv", index=False)
    if leaderboard is not None:
        leaderboard.to_csv(tmp_path / "leaderboard.csv", index=False)


def test_protocol3_audit_passes_for_valid_synthetic_tables(tmp_path) -> None:
    _write_artifacts(tmp_path)

    audit_path = build_audit_markdown(results_dir=tmp_path, out_path=tmp_path / "audit.md", include_calibrated=True)
    text = audit_path.read_text(encoding="utf-8")

    assert "[PASS] Protocol 3 summary rows declare calibrated target use" in text
    assert "[PASS] Protocol 3 prediction rows exclude calibration rows" in text
    assert "[PASS] Protocol 3 calibration counts equal `k * n_classes`" in text
    assert "[FAIL] Protocol 3" not in text


def test_protocol3_audit_fails_for_bad_summary_predictions_and_counts(tmp_path) -> None:
    summary = _valid_protocol3_summary()
    summary.loc[0, "uses_target_labels_for_fitting"] = False
    summary.loc[0, "valid_for_zero_calibration"] = True
    summary.loc[0, "n_target_calibration_trials"] = 5
    summary.loc[0, "target_calibration_indices"] = "1|4|8"

    predictions = _valid_protocol3_predictions().drop(columns=["n_target_evaluation_trials"])
    predictions.loc[0, "is_calibration_row"] = True
    predictions.loc[0, "target_row_index"] = 1

    _write_artifacts(tmp_path, summary=summary, predictions=predictions)

    audit_path = build_audit_markdown(results_dir=tmp_path, out_path=tmp_path / "audit.md", include_calibrated=True)
    text = audit_path.read_text(encoding="utf-8")

    assert "[FAIL] Protocol 3 summary rows declare calibrated target use" in text
    assert "`uses_target_labels_for_fitting` must be true" in text
    assert "`valid_for_zero_calibration` must be false" in text
    assert "[FAIL] Protocol 3 prediction rows exclude calibration rows" in text
    assert "missing required column `n_target_evaluation_trials`" in text
    assert "include rows marked as calibration rows" in text
    assert "prediction uses calibration row 1" in text
    assert "[FAIL] Protocol 3 calibration counts equal `k * n_classes`" in text
    assert "n_target_calibration_trials=5 but k*n_classes=6" in text


def test_protocol3_audit_allows_explicitly_skipped_calibration_count(tmp_path) -> None:
    summary = _valid_protocol3_summary()
    summary.loc[0, "n_target_calibration_trials"] = 0
    summary.loc[0, "target_calibration_skipped"] = True
    summary.loc[0, "target_calibration_skip_reason"] = "insufficient rows"
    _write_artifacts(tmp_path, summary=summary)

    audit_path = build_audit_markdown(results_dir=tmp_path, out_path=tmp_path / "audit.md", include_calibrated=True)
    text = audit_path.read_text(encoding="utf-8")

    assert "[PASS] Protocol 3 calibration counts equal `k * n_classes`" in text


def test_protocol3_leaderboard_requires_explicit_calibrated_flag(tmp_path) -> None:
    _write_artifacts(
        tmp_path,
        leaderboard=pd.DataFrame(
            [
                {
                    "protocol_category": 3,
                    "method": "source_plus_target_calibration_logistic_k2",
                    "mean_balanced_accuracy": 0.55,
                    "n_rows": 1,
                }
            ]
        ),
    )

    audit_path = build_audit_markdown(results_dir=tmp_path, out_path=tmp_path / "audit_without_flag.md")
    text = audit_path.read_text(encoding="utf-8")
    assert "[FAIL] Protocol 3 calibrated rows are absent from default Protocol 1/2 leaderboards" in text
    assert "not marked with `--include-calibrated`" in text

    audit_path = build_audit_markdown(results_dir=tmp_path, out_path=tmp_path / "audit_with_flag.md", include_calibrated=True)
    text = audit_path.read_text(encoding="utf-8")
    assert "[PASS] Protocol 3 calibrated rows are absent from default Protocol 1/2 leaderboards" in text
