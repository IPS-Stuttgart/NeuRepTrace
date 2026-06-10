from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from neureptrace.openneuro_alignment_compare import (
    build_anchor_comparison,
    build_oracle_comparison,
    build_raw_alignment_comparison,
    build_target_calibrated_comparison,
    build_variant_summary,
    discover_output_dirs,
    run_alignment_comparison,
)


def _write_alignment_artifact(
    root: Path,
    name: str,
    *,
    dataset: str = "ds000117",
    method: str = "mcca",
    anchor_mode: str,
    target_projection: str,
    fixed_value: float,
    write_diagnostics: bool = True,
) -> Path:
    output = root / name / f"openneuro_{dataset}_smoke"
    decode = output / "decode"
    decode.mkdir(parents=True)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": name,
                "dataset": dataset,
                "mode": "smoke",
                "github_run_id": f"run-{name}",
                "subjects": "1-4",
                "runs": "01,02",
                "n_subjects": 4,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time": [0.136, 0.184],
            "balanced_accuracy": [fixed_value - 0.02, fixed_value],
            "alignment_method": [method, method],
            "alignment_anchor_mode": [anchor_mode, anchor_mode],
            "alignment_anchor_column": ["", ""],
            "alignment_target_projection": [target_projection, target_projection],
        }
    ).to_csv(decode / "time_decode_summary.csv", index=False)
    if write_diagnostics:
        pd.DataFrame(
            {
                "dataset": [f"openneuro_{dataset}"],
                "test_subject": ["sub-01"],
                "alignment_method": [method],
                "sample_mode": [anchor_mode],
                "n_source_subjects": [3],
                "n_classes": [3],
                "n_alignment_rows": [48 if anchor_mode.endswith("repetition") else 3],
                "n_repetitions_per_class": [16 if anchor_mode.endswith("repetition") else ""],
                "requested_components": [64],
                "actual_components": [48 if anchor_mode.endswith("repetition") else 2],
                "feature_dim": [7650],
                "decode_feature_dim": [48 if anchor_mode.endswith("repetition") else 2],
                "alignment_window_center": [0.184],
                "alignment_window_size": [0.1],
                "decode_window_center": [0.184],
                "decode_window_size": [0.1],
                "uses_channel_projection_collapse": [False],
                "alignment_dimensionality_reduction": [True],
                "anchor_row_correlation_before": [0.1],
                "anchor_row_correlation_after": [0.8],
                "source_inner_decoding_before_alignment": [0.4],
                "source_inner_decoding_after_alignment": [0.45],
                "target_transform_type": [
                    "template_ridge_least_squares"
                    if target_projection == "oracle_target_calibrated_alignment"
                    else "target_calibrated_template_ridge_least_squares"
                    if target_projection == "target_calibrated_alignment"
                    else "source_group_projection"
                ],
            }
        ).to_csv(decode / "alignment_diagnostics.csv", index=False)
    return output


def test_alignment_compare_writes_variant_and_debug_decision_tables(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    _write_alignment_artifact(
        artifacts_root,
        "raw",
        method="none",
        anchor_mode="class_mean",
        target_projection="group_projection",
        fixed_value=0.45,
    )
    _write_alignment_artifact(
        artifacts_root,
        "class-repetition-strict",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.50,
    )
    _write_alignment_artifact(
        artifacts_root,
        "stimulus-id-strict",
        anchor_mode="stimulus_id_mean",
        target_projection="group_projection",
        fixed_value=0.62,
    )
    _write_alignment_artifact(
        artifacts_root,
        "class-repetition-oracle",
        anchor_mode="class_repetition",
        target_projection="oracle_target_calibrated_alignment",
        fixed_value=0.70,
    )
    _write_alignment_artifact(
        artifacts_root,
        "class-repetition-target-calibrated",
        anchor_mode="class_repetition",
        target_projection="target_calibrated_alignment",
        fixed_value=0.56,
    )

    discovered = discover_output_dirs([artifacts_root])
    assert len(discovered) == 5

    written = run_alignment_comparison(
        [artifacts_root],
        out_dir=tmp_path / "comparison",
        fixed_time=0.184,
        min_delta=0.01,
    )

    variants = pd.read_csv(written["variant_summary"])
    assert variants["artifact_name"].tolist() == [
        "class-repetition-oracle",
        "class-repetition-strict",
        "class-repetition-target-calibrated",
        "raw",
        "stimulus-id-strict",
    ]
    assert variants["diagnostic_actual_components_median"].tolist() == [48.0, 48.0, 48.0, 2.0, 2.0]
    assert variants["diagnostic_channel_projection_collapse_fraction"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert variants["diagnostic_dimensionality_reduction_fraction"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert variants["diagnostic_target_transform_type"].tolist() == [
        "template_ridge_least_squares",
        "source_group_projection",
        "target_calibrated_template_ridge_least_squares",
        "source_group_projection",
        "source_group_projection",
    ]
    validity = dict(zip(variants["artifact_name"], variants["alignment_valid_for_benchmark"], strict=True))
    assert bool(validity["class-repetition-oracle"]) is False
    assert bool(validity["class-repetition-target-calibrated"]) is False
    assert bool(validity["class-repetition-strict"]) is True

    raw = pd.read_csv(written["raw_comparison"])
    assert raw.loc[0, "decision"] == "alignment_improves_raw"
    assert raw.loc[0, "best_alignment_artifact"] == "stimulus-id-strict"
    assert round(float(raw.loc[0, "score_delta_alignment_minus_raw"]), 2) == 0.17

    anchors = pd.read_csv(written["anchor_comparison"])
    assert anchors.loc[0, "decision"] == "true_identity_anchor_better_than_class_repetition"
    assert anchors.loc[0, "interpretation"] == "anchor_semantics_likely_issue"
    assert anchors.loc[0, "best_identity_anchor_mode"] == "stimulus_id_mean"

    oracle = pd.read_csv(written["oracle_comparison"])
    assert oracle.loc[0, "decision"] == "oracle_target_calibration_helps"
    assert oracle.loc[0, "interpretation"] == "strict_source_only_target_projection_likely_bottleneck"

    target = pd.read_csv(written["target_calibrated_comparison"])
    assert target.loc[0, "decision"] == "target_calibrated_beats_strict_source_only"
    assert target.loc[0, "interpretation"] == "small_target_calibration_can_help_this_alignment"
    assert round(float(target.loc[0, "score_delta_target_calibrated_minus_strict"]), 2) == 0.06

    note = written["note"].read_text(encoding="utf-8")
    assert "Anchor Semantics" in note
    assert "Oracle Target Calibration" in note
    assert "Disjoint Target Calibration" in note


def test_alignment_compare_tolerates_older_artifacts_without_alignment_diagnostics(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    _write_alignment_artifact(
        artifacts_root,
        "raw",
        method="none",
        anchor_mode="class_mean",
        target_projection="group_projection",
        fixed_value=0.45,
        write_diagnostics=False,
    )
    _write_alignment_artifact(
        artifacts_root,
        "event-code-strict",
        anchor_mode="event_code_mean",
        target_projection="group_projection",
        fixed_value=0.50,
        write_diagnostics=False,
    )

    written = run_alignment_comparison(
        [artifacts_root],
        out_dir=tmp_path / "comparison",
        fixed_time=0.184,
        min_delta=0.01,
    )

    variants = pd.read_csv(written["variant_summary"])
    assert variants["alignment_diagnostics_present"].tolist() == [False, False]
    raw = pd.read_csv(written["raw_comparison"])
    assert raw.loc[0, "decision"] == "alignment_improves_raw"
    note = written["note"].read_text(encoding="utf-8")
    assert "Artifacts with alignment diagnostics: `0/2`" in note


def test_alignment_compare_selects_metric_from_observation_diagnostics_when_present(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    output = _write_alignment_artifact(
        artifacts_root,
        "aggregate-strict",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.90,
    )
    diagnostics_dir = output / "decode" / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": [0.136, 0.184],
            "balanced_accuracy": [0.62, 0.48],
            "accuracy": [0.62, 0.48],
        }
    ).to_csv(diagnostics_dir / "time_course_summary.csv", index=False)

    written = run_alignment_comparison([artifacts_root], out_dir=tmp_path / "comparison", fixed_time=None)

    variants = pd.read_csv(written["variant_summary"])
    assert variants.loc[0, "selection_source"] == "diagnostics_time_course"
    assert variants.loc[0, "selection_time"] == pytest.approx(0.136)
    assert variants.loc[0, "selection_value"] == pytest.approx(0.62)


def test_alignment_compare_rejects_mixed_alignment_metadata(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    output = _write_alignment_artifact(
        artifacts_root,
        "mixed-projection",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.50,
    )
    summary_path = output / "decode" / "time_decode_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[summary["time"] == 0.184, "alignment_target_projection"] = "oracle_target_calibrated_alignment"
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="inconsistent 'alignment_target_projection'"):
        run_alignment_comparison([artifacts_root], out_dir=tmp_path / "comparison", fixed_time=0.184)


def test_alignment_compare_respects_explicit_invalid_benchmark_flag(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    output = _write_alignment_artifact(
        artifacts_root,
        "invalid-strict-looking",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.80,
    )
    summary_path = output / "decode" / "time_decode_summary.csv"
    summary = pd.read_csv(summary_path)
    summary["alignment_valid_for_benchmark"] = False
    summary.to_csv(summary_path, index=False)

    written = run_alignment_comparison([artifacts_root], out_dir=tmp_path / "comparison", fixed_time=0.184)

    variants = pd.read_csv(written["variant_summary"])
    assert variants["alignment_target_projection"].unique().tolist() == ["group_projection"]
    assert variants["alignment_valid_for_benchmark"].unique().tolist() == [False]


def test_alignment_compare_ignores_invalid_strict_rows_for_debug_baselines(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    invalid_strict = _write_alignment_artifact(
        artifacts_root,
        "invalid-strict-looking",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.80,
    )
    summary_path = invalid_strict / "decode" / "time_decode_summary.csv"
    summary = pd.read_csv(summary_path)
    summary["alignment_valid_for_benchmark"] = False
    summary.to_csv(summary_path, index=False)
    oracle = _write_alignment_artifact(
        artifacts_root,
        "oracle-debug",
        anchor_mode="class_repetition",
        target_projection="oracle_target_calibrated_alignment",
        fixed_value=0.90,
    )
    target_calibrated = _write_alignment_artifact(
        artifacts_root,
        "target-calibrated-debug",
        anchor_mode="class_repetition",
        target_projection="target_calibrated_alignment",
        fixed_value=0.85,
    )

    variants = build_variant_summary(
        [invalid_strict, oracle, target_calibrated],
        fixed_time=0.184,
    )

    assert build_oracle_comparison(variants).empty
    target_comparison = build_target_calibrated_comparison(variants)
    assert target_comparison.loc[0, "decision"] == "target_calibrated_without_strict_pair"
    assert target_comparison.loc[0, "strict_artifact"] == ""


def test_alignment_compare_ignores_invalid_anchor_semantics_rows(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    class_row = _write_alignment_artifact(
        artifacts_root,
        "invalid-class-repetition",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.50,
    )
    identity_row = _write_alignment_artifact(
        artifacts_root,
        "invalid-stimulus-id",
        anchor_mode="stimulus_id_mean",
        target_projection="group_projection",
        fixed_value=0.70,
    )
    for output in (class_row, identity_row):
        summary_path = output / "decode" / "time_decode_summary.csv"
        summary = pd.read_csv(summary_path)
        summary["alignment_valid_for_benchmark"] = False
        summary.to_csv(summary_path, index=False)

    variants = build_variant_summary([class_row, identity_row], fixed_time=0.184)

    assert build_anchor_comparison(variants).empty


def test_alignment_compare_ignores_invalid_raw_baseline(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    raw = _write_alignment_artifact(
        artifacts_root,
        "invalid-raw",
        method="none",
        anchor_mode="class_mean",
        target_projection="group_projection",
        fixed_value=0.40,
    )
    raw_summary_path = raw / "decode" / "time_decode_summary.csv"
    raw_summary = pd.read_csv(raw_summary_path)
    raw_summary["alignment_valid_for_benchmark"] = False
    raw_summary.to_csv(raw_summary_path, index=False)
    aligned = _write_alignment_artifact(
        artifacts_root,
        "strict-aligned",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.60,
    )

    variants = build_variant_summary([raw, aligned], fixed_time=0.184)

    assert build_raw_alignment_comparison(variants).empty


def test_alignment_compare_writes_readable_empty_comparison_tables(tmp_path: Path):
    artifacts_root = tmp_path / "artifacts"
    _write_alignment_artifact(
        artifacts_root,
        "strict-only",
        anchor_mode="class_repetition",
        target_projection="group_projection",
        fixed_value=0.55,
    )

    written = run_alignment_comparison(
        [artifacts_root],
        out_dir=tmp_path / "comparison",
        fixed_time=0.184,
    )

    for key in ("raw_comparison", "anchor_comparison", "oracle_comparison", "target_calibrated_comparison"):
        table = pd.read_csv(written[key])
        assert table.empty
        assert len(table.columns) > 0
