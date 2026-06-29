from __future__ import annotations

from neureptrace.bushmeg_all_protocols_report import build_bushmeg_all_protocols_report


def test_bushmeg_all_protocols_report_accepts_existing_empty_csv_inputs(tmp_path) -> None:
    summary_csv = tmp_path / "summary.csv"
    metadata_csv = tmp_path / "method_metadata.csv"
    summary_csv.write_text("", encoding="utf-8")
    metadata_csv.write_text("", encoding="utf-8")

    result = build_bushmeg_all_protocols_report(summary_csv=summary_csv, method_metadata_csv=metadata_csv, out_dir=tmp_path)

    assert result.leaderboard.empty
    assert result.protocol_summary.empty
    assert result.report_md.exists()
    assert "No runnable result rows were found." in result.report_md.read_text(encoding="utf-8")
