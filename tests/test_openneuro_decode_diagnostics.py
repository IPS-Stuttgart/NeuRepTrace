from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neureptrace.openneuro_decode_diagnostics import best_metric_rows, main, write_decode_diagnostics


def test_best_metric_rows_selects_maximized_and_minimized_metrics():
    summary = pd.DataFrame(
        {
            "time": [0.1, 0.1, 0.2, 0.2],
            "accuracy": [0.6, 0.4, 0.7, 0.9],
            "balanced_accuracy": [0.55, 0.45, 0.6, 0.7],
            "log_loss": [0.8, 0.9, 0.5, 0.7],
            "ece": [0.20, 0.10, 0.15, 0.05],
        }
    )

    best = best_metric_rows(summary).set_index("selection_metric")

    assert best.loc["accuracy", "time"] == 0.2
    assert best.loc["accuracy", "selection_value"] == 0.8
    assert best.loc["balanced_accuracy", "time"] == 0.2
    assert best.loc["balanced_accuracy", "selection_value"] == 0.65
    assert best.loc["log_loss", "time"] == 0.2
    assert best.loc["log_loss", "selection_value"] == 0.6
    assert best.loc["ece", "time"] == 0.2
    assert best.loc["ece", "selection_value"] == 0.1


def test_write_decode_diagnostics_handles_missing_decode_summary(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "openneuro_ds006629_full"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset_id": ["ds006629", "ds006629"],
            "subject": ["sub-01", "sub-02"],
            "epochs_path": ["sub-01_epo.fif", "sub-02_epo.fif"],
            "n_trials": [10, 20],
            "labels": ["a|b", "a|b"],
            "runs": ["0", "0"],
        }
    ).to_csv(output_dir / "stage_summary.csv", index=False)

    diagnostics, best = write_decode_diagnostics(output_dir)

    assert diagnostics["decode_summary"] == {"exists": False}
    assert "Missing decode summary" in diagnostics["warning"]
    assert diagnostics["stage_summary"]["n_subjects"] == 2
    assert diagnostics["stage_summary"]["total_trials"] == 30
    assert diagnostics["stage_summary"]["labels"] == ["a", "b"]
    assert best.empty

    diagnostics_path = output_dir / "decode_diagnostics.json"
    best_path = output_dir / "decode_best_metrics.csv"
    assert diagnostics_path.is_file()
    assert best_path.is_file()
    loaded = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert loaded["stage_summary"]["subjects"] == ["sub-01", "sub-02"]


def test_write_decode_diagnostics_writes_best_metric_table(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "openneuro_ds006629_smoke"
    decode_dir = output_dir / "decode"
    decode_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": [0.1, 0.2],
            "accuracy": [0.5, 0.75],
            "log_loss": [0.9, 0.4],
        }
    ).to_csv(decode_dir / "time_decode_summary.csv", index=False)

    diagnostics, best = write_decode_diagnostics(output_dir)

    assert diagnostics["decode_summary"]["exists"] is True
    assert diagnostics["decode_summary"]["rows"] == 2
    assert diagnostics["decode_summary"]["n_times"] == 2
    assert set(best["selection_metric"]) == {"accuracy", "log_loss"}
    assert (output_dir / "decode_best_metrics.csv").is_file()


def test_main_strict_reports_missing_decode_summary(tmp_path: Path):
    output_dir = tmp_path / "missing-summary"

    assert main([str(output_dir)]) == 0
    assert main([str(output_dir), "--strict"]) == 1
