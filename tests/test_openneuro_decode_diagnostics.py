from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from neureptrace.openneuro_decode_diagnostics import aggregate_workflow_outputs, best_metric_rows, main, write_decode_diagnostics


def _toy_observations(subject: str) -> pd.DataFrame:
    rows = []
    for time in (0.10, 0.184):
        for sample_index, true_label in enumerate((0, 1, 2)):
            predicted = true_label
            if subject == "sub-02" and time == 0.184 and true_label == 2:
                predicted = 1
            probabilities = [0.1, 0.1, 0.1]
            probabilities[predicted] = 0.8
            rows.append(
                {
                    "group": subject,
                    "time": time,
                    "sample_index": f"{subject}-{sample_index}",
                    "true_label": true_label,
                    "true_class": f"class_{true_label}",
                    "predicted_label": predicted,
                    "predicted_class": f"class_{predicted}",
                    "prob_class_0": probabilities[0],
                    "prob_class_1": probabilities[1],
                    "prob_class_2": probabilities[2],
                }
            )
    return pd.DataFrame(rows)


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
                "ensemble_weights": "0.5,0.3,0.2",
                "ensemble_source_decoders": "multinomial-logistic-weighted,linear_svm,shrinkage_lda",
                "ensemble_source_temperatures": "[1.25,1.0,0.8]",
                "ensemble_score_mode": "probability",
                "ensemble_source_baseline_debiasing": "true",
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
            "top2_interpretation": ["informative"],
            "top3_interpretation": ["automatic_ceiling"],
            "fixed_time": [0.2],
            "fixed_balanced_accuracy": [0.34],
            "fixed_balanced_minus_chance": [0.0067],
            "fixed_top2_accuracy": [0.8],
            "fixed_top2_minus_chance": [0.8 - 2 / 3],
            "fixed_top3_accuracy": [1.0],
            "fixed_top3_minus_chance": [0.0],
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
    assert quality.loc[0, "top2_evidence_role"] == "chance_adjusted_supporting"
    assert quality.loc[0, "top3_evidence_role"] == "uninformative_automatic_ceiling"
    assert quality.loc[0, "ensemble_weights"] == "0.5,0.3,0.2"
    assert quality.loc[0, "ensemble_source_decoders"] == "multinomial-logistic-weighted,linear_svm,shrinkage_lda"
    assert quality.loc[0, "ensemble_source_temperatures"] == "[1.25,1.0,0.8]"
    assert quality.loc[0, "ensemble_score_mode"] == "probability"
    assert bool(quality.loc[0, "ensemble_source_baseline_debiasing"])
    assert quality.loc[0, "fixed_balanced_accuracy"] == pytest.approx(0.34)
    assert quality.loc[0, "fixed_balanced_minus_chance_pct"] == pytest.approx(0.67)
    assert quality.loc[0, "fixed_top2_minus_chance_pct"] == pytest.approx((0.8 - 2 / 3) * 100.0)
    assert quality.loc[0, "fixed_top3_minus_chance_pct"] == pytest.approx(0.0)
    assert quality.loc[0, "best_selection_metric"] == "accuracy"
    assert quality.loc[0, "best_selection_value"] == pytest.approx(0.75)


def test_write_decode_diagnostics_recovers_ensemble_provenance_from_summary(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "openneuro_ds006629_full"
    decode_dir = output_dir / "decode"
    diagnostics_dir = decode_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
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
            "time": [0.184],
            "balanced_accuracy": [0.48],
            "source_decoders": ["multinomial-logistic-weighted|linear_svm|shrinkage_lda"],
            "ensemble_weights": ["0.5|0.3|0.2"],
            "ensemble_source_temperatures": ["1.25|1|0.8"],
            "ensemble_score_mode": ["probability"],
            "ensemble_source_baseline_debiasing": [True],
            "ensemble_baseline_window_start": [-0.2],
            "ensemble_baseline_window_stop": [0.0],
        }
    ).to_csv(decode_dir / "time_decode_summary.csv", index=False)
    pd.DataFrame(
        {
            "n_classes": [3],
            "fixed_time": [0.184],
            "fixed_balanced_accuracy": [0.48],
            "fixed_balanced_minus_chance": [0.48 - 1 / 3],
            "subjects_fixed_above_chance": [12],
        }
    ).to_csv(diagnostics_dir / "quality_summary.csv", index=False)

    write_decode_diagnostics(output_dir)

    quality = pd.read_csv(output_dir / "workflow_quality_summary.csv")
    assert quality.loc[0, "ensemble_source_decoders"] == "multinomial-logistic-weighted|linear_svm|shrinkage_lda"
    assert quality.loc[0, "ensemble_weights"] == "0.5|0.3|0.2"
    assert quality.loc[0, "ensemble_source_temperatures"] == "1.25|1|0.8"
    assert quality.loc[0, "ensemble_score_mode"] == "probability"
    assert bool(quality.loc[0, "ensemble_source_baseline_debiasing"])
    assert quality.loc[0, "ensemble_baseline_window_start"] == pytest.approx(-0.2)
    assert quality.loc[0, "ensemble_baseline_window_stop"] == pytest.approx(0.0)


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


def test_aggregate_workflow_outputs_combines_sharded_loso_artifacts(tmp_path: Path):
    source_dirs = []
    for subject in ("sub-01", "sub-02"):
        output_dir = tmp_path / f"shard-{subject}"
        decode_dir = output_dir / "decode"
        decode_dir.mkdir(parents=True)
        source_dirs.append(output_dir)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "ds006629",
                    "mode": "full",
                    "artifact_name": "openneuro-meg-ds006629-full",
                    "label_shuffle_control": "false",
                    "outer_test_groups": subject,
                    "diagnostics_best_time": "0.184",
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "dataset_id": ["ds006629"],
                "subject": [subject],
                "epochs_path": [f"{subject}_epo.fif"],
                "n_trials": [3],
                "labels": ["class_0|class_1|class_2"],
                "runs": ["0"],
            }
        ).to_csv(output_dir / "stage_summary.csv", index=False)
        _toy_observations(subject).to_csv(decode_dir / "observations.csv", index=False)
        pd.DataFrame(
            {
                "time": [0.10, 0.184],
                "balanced_accuracy": [1.0, 1.0 if subject == "sub-01" else 2 / 3],
                "accuracy": [1.0, 1.0 if subject == "sub-01" else 2 / 3],
            }
        ).to_csv(decode_dir / "time_decode_summary.csv", index=False)
        pd.DataFrame(
            {
                "dataset": ["openneuro_ds006629_singsing"],
                "test_subject": [subject],
                "alignment_method": ["mcca"],
                "sample_mode": ["class_mean"],
                "n_source_subjects": [1],
                "n_classes": [3],
                "n_alignment_rows": [3],
                "n_repetitions_per_class": [""],
                "requested_components": [64],
                "actual_components": [2],
                "feature_dim": [64],
                "decode_feature_dim": [2],
                "alignment_window_center": [0.184],
                "alignment_window_size": [0.1],
                "decode_window_center": [0.184],
                "decode_window_size": [0.1],
                "uses_channel_projection_collapse": [False],
                "anchor_row_correlation_before": [0.2],
                "anchor_row_correlation_after": [0.6],
                "source_inner_decoding_before_alignment": [0.5],
                "source_inner_decoding_after_alignment": [0.55],
                "target_transform_type": ["source_group_projection"],
            }
        ).to_csv(decode_dir / "alignment_diagnostics.csv", index=False)

    aggregate_dir = tmp_path / "aggregate"
    diagnostics, best = aggregate_workflow_outputs(source_dirs, out_dir=aggregate_dir)

    assert diagnostics["decode_summary"]["exists"] is True
    assert diagnostics["decode_summary"]["rows"] == 4
    assert diagnostics["alignment_diagnostics"]["exists"] is True
    assert diagnostics["alignment_diagnostics"]["rows"] == 2
    assert best.set_index("selection_metric").loc["balanced_accuracy", "selection_value"] == pytest.approx(1.0)
    assert pd.read_csv(aggregate_dir / "stage_summary.csv")["subject"].tolist() == ["sub-01", "sub-02"]
    assert len(pd.read_csv(aggregate_dir / "decode" / "observations.csv")) == 12
    alignment = pd.read_csv(aggregate_dir / "decode" / "alignment_diagnostics.csv")
    assert alignment["test_subject"].tolist() == ["sub-01", "sub-02"]
    assert alignment["actual_components"].tolist() == [2, 2]
    assert alignment["target_transform_type"].tolist() == ["source_group_projection", "source_group_projection"]
    manifest = json.loads((aggregate_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_name"] == "openneuro-meg-ds006629-full-shard-aggregate"
    assert manifest["outer_test_groups"] == "sub-01|sub-02"
    quality = pd.read_csv(aggregate_dir / "workflow_quality_summary.csv")
    assert quality.loc[0, "shard_count"] == 2
    assert quality.loc[0, "aggregate_outer_test_groups"] == "sub-01|sub-02"
    assert quality.loc[0, "source_artifacts"] == "openneuro-meg-ds006629-full|openneuro-meg-ds006629-full"
    assert quality.loc[0, "quality_decision"] == "promising_above_chance_consistent"
    assert quality.loc[0, "fixed_balanced_accuracy"] == pytest.approx(5 / 6)
    assert quality.loc[0, "fixed_balanced_minus_chance"] == pytest.approx(5 / 6 - 1 / 3)
    assert quality.loc[0, "fixed_balanced_minus_chance_pct"] == pytest.approx(50.0)


def test_main_strict_reports_missing_decode_summary(tmp_path: Path):
    output_dir = tmp_path / "missing-summary"

    assert main([str(output_dir)]) == 0
    assert main([str(output_dir), "--strict"]) == 1
