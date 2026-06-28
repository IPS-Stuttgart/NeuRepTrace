from __future__ import annotations

import pandas as pd

from neureptrace.bushmeg_all_protocols_report import build_bushmeg_all_protocols_report


def test_bushmeg_all_protocols_report_preserves_fractional_protocol_categories(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "protocol_category": 2.5,
                "method": "mid_protocol_method",
                "method_family": "unlabeled_target_selection",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.61,
                "accuracy": 0.60,
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
            }
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "protocol_category": 2.5,
                "method": "mid_protocol_method",
                "method_family": "unlabeled_target_selection",
                "status": "runnable",
                "valid_for_zero_calibration": True,
                "valid_for_strict_source_only": False,
                "debug_upper_bound": False,
            }
        ]
    )
    summary_csv = tmp_path / "summary.csv"
    metadata_csv = tmp_path / "method_metadata.csv"
    summary.to_csv(summary_csv, index=False)
    metadata.to_csv(metadata_csv, index=False)

    result = build_bushmeg_all_protocols_report(summary_csv=summary_csv, method_metadata_csv=metadata_csv, out_dir=tmp_path)

    assert result.protocol_summary["protocol_category"].tolist() == [2.5]
    protocol25 = result.protocol_summary.loc[result.protocol_summary["protocol_category"] == 2.5].iloc[0]
    assert protocol25["n_rows"] == 1
    assert protocol25["n_methods"] == 1
    assert protocol25["n_runnable_methods"] == 1
    assert protocol25["mean_balanced_accuracy"] == 0.61

    report_text = result.report_md.read_text(encoding="utf-8")
    assert "P2.5 `mid_protocol_method`" in report_text
    assert "- P2.5: 61.00% mean BA" in report_text
