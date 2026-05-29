from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
    assert best.loc["accuracy", "selection_value"] == pytest.approx(0.8)
    assert best.loc["balanced_accuracy", "time"] == 0.2
    assert best.loc["balanced_accuracy", "selection_value"] == pytest.approx(0.65)
    assert best.loc["log_loss", "time"] == 0.2
    assert best.loc["log_loss", "selection_value"] == pytest.approx(0.6)
    assert best.loc["ece", "time"] == 0.2
    assert best.loc["ece", "selection_value"] == pytest.approx(0.1)


def test_write_decode_diagnostics_handles_missing_decode_summary(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "openneuro_ds006629_full"
    output_dir.mkdir(parents=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "ds006629",
                "mode": "full",
                "artifact_name": "openneuro-meg-ds006629-full",
                "label_shuffle_control": "false",
            }
        ),
        encoding="utf-8",
    )
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
    quality_path = output_dir / "workflow_quality_summary.csv"
    assert diagnostics_path.is_file()
    assert best_path.is_file()
    assert quality_path.is_file()
    loaded = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert loaded["stage_summary"]["subjects"] == ["sub-01", "sub-02"]
    quality = pd.read_csv(quality_path)
    assert quality.loc[0, "dataset"] == "ds006629"
    assert quality.loc[0, "result_variant"] == "raw"
    assert quality.loc[0, "artifact_name"] == "openneuro-meg-ds006629-full"
    assert not bool(quality.loc[0, "decode_summary_exists"])
    assert quality.loc[0, "quality_decision"] == "missing_decode_summary"


def test_write_decode_diagnostics_writes_best_metric_table(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "openneuro_ds006629_smoke"
    decode_dir = output_dir / "decode"
    diagnostics_dir = decode_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "ds006629",
                "mode": "smoke",
                "artifact_name": "openneuro-meg-ds006629-smoke-label-shuffle-seed-13",
                "label_shuffle_control": "true",
                "label_shuffle_seed": "13",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time": [0.1, 0.2],
            "accuracy": [0.5, 0.75],
            "log_loss": [0.9, 0.4],
        }
    ).to_csv(decode_dir / "time_decode_summary.csv", index=False)
    pd.DataFrame(
        {
            "n_classes": [3],
            "chance_accuracy": [1 / 3],
            "top2_chance": [2 / 3],
            "top3_chance": [1.0],
            "top3_interpretation": ["automatic_ceiling"],
            "fixed_time": [0.2],
            "fixed_balanced_accuracy": [0.34],
            "fixed_balanced_minus_chance": [0.0067],
            "fixed_top2_accuracy": [0.8],
            "subjects_fixed_above_chance": [14],
        }
    ).to_csv(diagnostics_dir / "quality_summary.csv", index=False)

    diagnostics, best = write_decode_diagnostics(output_dir)

    assert diagnostics["decode_summary"]["exists"] is True
    assert diagnostics["decode_summary"]["rows"] == 2
    assert diagnostics["decode_summary"]["n_times"] == 2
    assert set(best["selection_metric"]) == {"accuracy", "log_loss"}
    assert (output_dir / "decode_best_metrics.csv").is_file()
    quality = pd.read_csv(output_dir / "workflow_quality_summary.csv")
    assert bool(quality.loc[0, "label_shuffle_control"])
    assert quality.loc[0, "n_classes"] == 3
    assert quality.loc[0, "quality_decision"] == "null_near_chance"
    assert quality.loc[0, "null_chance_tolerance"] == pytest.approx(0.03)
    assert quality.loc[0, "top3_interpretation"] == "automatic_ceiling"
    assert quality.loc[0, "fixed_balanced_accuracy"] == pytest.approx(0.34)
    assert quality.loc[0, "best_selection_metric"] == "accuracy"
    assert quality.loc[0, "best_selection_value"] == pytest.approx(0.75)


def test_write_decode_diagnostics_adds_temporal_smoothing_quality_row(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "openneuro_ds006629_full"
    decode_dir = output_dir / "decode"
    diagnostics_dir = decode_dir / "diagnostics"
    smoothed_dir = decode_dir / "temporal_smoothing"
    smoothed_diagnostics_dir = smoothed_dir / "diagnostics"
    smoothed_diagnostics_dir.mkdir(parents=True)
    diagnostics_dir.mkdir(parents=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "ds006629",
                "mode": "full",
                "artifact_name": "openneuro-meg-ds006629-full",
                "label_shuffle_control": "false",
                "temporal_smoothing": "true",
                "temporal_smoothing_fit_window": "0.10,0.30",
                "temporal_smoothing_stay_grid_size": "200",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "dataset_id": ["ds006629", "ds006629"],
            "subject": ["sub-01", "sub-02"],
            "epochs_path": ["sub-01_epo.fif", "sub-02_epo.fif"],
            "n_trials": [10, 20],
            "labels": ["a|b|c", "a|b|c"],
            "runs": ["0", "0"],
        }
    ).to_csv(output_dir / "stage_summary.csv", index=False)
    pd.DataFrame({"time": [0.184], "balanced_accuracy": [0.48], "top2_accuracy": [0.80]}).to_csv(
        decode_dir / "time_decode_summary.csv",
        index=False,
    )
    pd.DataFrame({"time": [0.184], "balanced_accuracy": [0.52], "top2_accuracy": [0.82]}).to_csv(
        smoothed_dir / "time_decode_summary.csv",
        index=False,
    )
    for path, balanced in (
        (diagnostics_dir / "quality_summary.csv", 0.48),
        (smoothed_diagnostics_dir / "quality_summary.csv", 0.52),
    ):
        pd.DataFrame(
            {
                "n_classes": [3],
                "chance_accuracy": [1 / 3],
                "top2_chance": [2 / 3],
                "top3_chance": [1.0],
                "top3_interpretation": ["automatic_ceiling"],
                "fixed_time": [0.184],
                "fixed_balanced_accuracy": [balanced],
                "fixed_balanced_minus_chance": [balanced - 1 / 3],
                "fixed_top2_accuracy": [0.8],
                "subjects_fixed_above_chance": [2],
            }
        ).to_csv(path, index=False)

    diagnostics, _best = write_decode_diagnostics(output_dir)

    assert diagnostics["temporal_smoothing_summary"]["exists"] is True
    quality = pd.read_csv(output_dir / "workflow_quality_summary.csv")
    assert quality["result_variant"].tolist() == ["raw", "temporal_smoothing"]
    smoothed = quality.set_index("result_variant").loc["temporal_smoothing"]
    assert bool(smoothed["temporal_smoothing"])
    assert smoothed["temporal_smoothing_fit_window"] == "0.10,0.30"
    assert smoothed["fixed_balanced_accuracy"] == pytest.approx(0.52)
    assert smoothed["best_selection_value"] == pytest.approx(0.52)


def test_main_strict_reports_missing_decode_summary(tmp_path: Path):
    output_dir = tmp_path / "missing-summary"

    assert main([str(output_dir)]) == 0
    assert main([str(output_dir), "--strict"]) == 1
