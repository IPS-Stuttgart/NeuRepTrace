from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.bushmeg_all_protocols_report import LEADERBOARD_COLUMNS, build_bushmeg_all_protocols_report


def test_bushmeg_all_protocols_report_writes_expected_outputs(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "protocol_category": 1,
                "method": "source_a",
                "method_family": "source",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.50,
                "accuracy": 0.52,
                "log_loss": 1.1,
                "brier": 0.22,
                "ece": 0.08,
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": True,
                "debug_upper_bound": False,
            },
            {
                "protocol_category": 1,
                "method": "source_a",
                "method_family": "source",
                "outer_test_subject": "2",
                "balanced_accuracy": 0.70,
                "accuracy": 0.68,
                "log_loss": 0.9,
                "brier": 0.18,
                "ece": 0.05,
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": True,
                "debug_upper_bound": False,
            },
            {
                "protocol_category": 2,
                "method": "coral",
                "method_family": "unlabeled_alignment",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.62,
                "accuracy": 0.60,
                "log_loss": 1.0,
                "brier": 0.20,
                "ece": 0.07,
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
            },
        ]
    )
    method_metadata = pd.DataFrame(
        [
            {
                "protocol_category": 1,
                "method": "source_a",
                "method_family": "source",
                "status": "runnable",
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": True,
                "debug_upper_bound": False,
            },
            {
                "protocol_category": 1,
                "method": "blocked_source",
                "method_family": "source",
                "status": "skipped",
                "skip_reason": "missing optional adapter",
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": True,
                "debug_upper_bound": False,
            },
            {
                "protocol_category": 2,
                "method": "coral",
                "method_family": "unlabeled_alignment",
                "status": "runnable",
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
            },
        ]
    )
    summary_csv = tmp_path / "summary.csv"
    metadata_csv = tmp_path / "method_metadata.csv"
    summary.to_csv(summary_csv, index=False)
    method_metadata.to_csv(metadata_csv, index=False)

    result = build_bushmeg_all_protocols_report(summary_csv=summary_csv, method_metadata_csv=metadata_csv, out_dir=tmp_path)

    assert list(result.leaderboard.columns) == LEADERBOARD_COLUMNS
    source_row = result.leaderboard.loc[result.leaderboard["method"] == "source_a"].iloc[0]
    assert source_row["mean_balanced_accuracy"] == 0.60
    assert source_row["median_balanced_accuracy"] == 0.60
    assert source_row["n_subjects"] == 2
    assert source_row["n_rows"] == 2

    skipped_row = result.leaderboard.loc[result.leaderboard["method"] == "blocked_source"].iloc[0]
    assert skipped_row["n_skipped"] == 1
    assert skipped_row["n_rows"] == 0
    assert result.skipped_methods["method"].tolist() == ["blocked_source"]

    protocol1 = result.protocol_summary.loc[result.protocol_summary["protocol_category"] == 1].iloc[0]
    assert protocol1["n_methods"] == 2
    assert protocol1["n_runnable_methods"] == 1
    assert protocol1["n_skipped_methods"] == 1

    assert set(result.subject_summary["outer_test_subject"]) == {"1", "2"}
    for path in (
        result.leaderboard_csv,
        result.protocol_summary_csv,
        result.subject_summary_csv,
        result.skipped_methods_csv,
        result.balanced_accuracy_by_method_png,
        result.balanced_accuracy_by_protocol_png,
        result.report_md,
    ):
        assert path.exists()
        assert path.stat().st_size > 0

    assert "BUSH-MEG All-Protocols Report" in result.report_md.read_text(encoding="utf-8")


def test_protocol3_kshot_tables_and_plots_are_written(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "protocol_category": 1,
                "method": "source_loso_logistic",
                "method_family": "source",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.50,
                "accuracy": 0.50,
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": True,
                "debug_upper_bound": False,
            },
            {
                "protocol_category": 1,
                "method": "better_source",
                "method_family": "source",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.55,
                "accuracy": 0.55,
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": True,
                "debug_upper_bound": False,
            },
            {
                "protocol_category": 3,
                "method": "few_shot_target_calibrated_decoder_k2",
                "method_family": "few_shot_target_calibration",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.65,
                "accuracy": 0.64,
                "k_per_class": 2,
                "n_target_evaluation_trials": 90,
                "n_target_calibration_trials": 6,
                "valid_for_zero_calibration": False,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
            },
            {
                "protocol_category": 3,
                "method": "few_shot_target_calibrated_decoder_k4",
                "method_family": "few_shot_target_calibration",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.70,
                "accuracy": 0.69,
                "k_per_class": 4,
                "n_target_evaluation_trials": 84,
                "n_target_calibration_trials": 12,
                "valid_for_zero_calibration": False,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {"method": "source_loso_logistic", "protocol_category": 1, "method_family": "source", "status": "runnable"},
            {"method": "better_source", "protocol_category": 1, "method_family": "source", "status": "runnable"},
            {
                "method": "few_shot_target_calibrated_decoder_k2",
                "protocol_category": 3,
                "method_family": "few_shot_target_calibration",
                "status": "runnable",
            },
            {
                "method": "few_shot_target_calibrated_decoder_k4",
                "protocol_category": 3,
                "method_family": "few_shot_target_calibration",
                "status": "runnable",
            },
        ]
    )
    summary_csv = tmp_path / "summary.csv"
    metadata_csv = tmp_path / "method_metadata.csv"
    summary.to_csv(summary_csv, index=False)
    metadata.to_csv(metadata_csv, index=False)

    result = build_bushmeg_all_protocols_report(summary_csv=summary_csv, method_metadata_csv=metadata_csv, out_dir=tmp_path)

    assert result.protocol3_kshot_leaderboard_csv.exists()
    assert result.protocol3_by_k_csv.exists()
    assert result.protocol3_delta_vs_source_only_csv.exists()
    assert result.protocol3_accuracy_by_k_png.exists()
    assert result.protocol3_delta_by_k_png.exists()

    k2 = result.protocol3_kshot_leaderboard.loc[result.protocol3_kshot_leaderboard["k_per_class"] == 2].iloc[0]
    assert k2["method_base"] == "few_shot_target_calibrated_decoder"
    assert k2["mean_balanced_accuracy"] == 0.65
    assert k2["mean_delta_vs_source_loso_logistic"] == pytest.approx(0.15)
    assert round(float(k2["mean_delta_vs_best_protocol1"]), 10) == 0.10
    assert int(k2["n_eval_trials"]) == 90
    assert int(k2["n_calibration_trials"]) == 6

    assert set(result.protocol3_by_k["k_per_class"].tolist()) == {2, 4}
    assert "Protocol 3 K-Shot Summary" in result.report_md.read_text(encoding="utf-8")
