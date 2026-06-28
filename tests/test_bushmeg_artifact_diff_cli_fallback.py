from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.bushmeg_artifact_diff import main


def test_artifact_diff_cli_prints_mean_when_group_column_is_absent(tmp_path: Path):
    reference_summary = tmp_path / "reference_summary.csv"
    candidate_summary = tmp_path / "candidate_summary.csv"
    summary_out = tmp_path / "summary_diff.csv"
    pd.DataFrame({"balanced_accuracy": [0.10, 0.20], "accuracy": [0.20, 0.30]}).to_csv(reference_summary, index=False)
    pd.DataFrame({"balanced_accuracy": [0.15, 0.25], "accuracy": [0.25, 0.35]}).to_csv(candidate_summary, index=False)

    assert main([str(reference_summary), str(candidate_summary), "--summary-out", str(summary_out)]) == 0

    diff = pd.read_csv(summary_out)
    assert "row_index" in diff.columns
    mean_row = diff[(diff["row_index"] == "__mean__") & (diff["metric"] == "balanced_accuracy")].iloc[0]
    assert np.isclose(mean_row["delta_candidate_minus_reference"], 0.05)
