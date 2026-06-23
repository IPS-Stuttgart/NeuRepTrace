from pathlib import Path

import pandas as pd
import pytest

from neureptrace import bushmeg_protocol4_oracle as oracle


def test_protocol4_oracle_requires_explicit_gate(tmp_path: Path):
    with pytest.raises(ValueError, match="include_oracle"):
        oracle.run_bushmeg_protocol4_oracle(
            tmp_path / "config.yml",
            include_oracle=False,
            out_dir=tmp_path / "out",
        )


def test_protocol4_method_normalization_and_overrides():
    assert oracle.normalize_oracle_alignment_method("m-cca") == "mcca"
    overrides = oracle.protocol4_oracle_overrides("hyper-alignment", components=12, repetition_cap=None)

    assert "source_loso.alignment_method=hyperalignment" in overrides
    assert "source_loso.alignment_target_projection=oracle_target_calibrated_alignment" in overrides
    assert "source_loso.alignment_anchor_mode=class_mean" in overrides
    assert "source_loso.alignment_components=12" in overrides
    assert not any(item.startswith("source_loso.alignment_repetition_cap=") for item in overrides)


def test_protocol4_metadata_marks_debug_upper_bound():
    metadata = oracle.protocol4_metadata("procrustes")

    assert metadata["protocol_category"] == 4
    assert metadata["uses_target_data"] is True
    assert metadata["uses_target_labels_for_fitting"] is True
    assert metadata["uses_target_labels_for_scoring_only"] is False
    assert metadata["valid_for_zero_calibration"] is False
    assert metadata["valid_for_strict_source_only"] is False
    assert metadata["debug_upper_bound"] is True


def test_protocol4_runner_writes_enriched_outputs(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run_bushmeg_source_loso(config_path, *, overrides, out_path, inner_cv_out_path, predictions_out_path):
        calls.append(
            {
                "config_path": config_path,
                "overrides": tuple(overrides),
                "out_path": Path(out_path),
                "inner_cv_out_path": Path(inner_cv_out_path),
                "predictions_out_path": Path(predictions_out_path),
            }
        )
        summary = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "candidate": "candidate-a",
                    "balanced_accuracy": 0.25,
                    "accuracy": 0.25,
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "trial_index": 0,
                    "true_label": 0,
                    "predicted_label": 0,
                }
            ]
        )
        inner = pd.DataFrame(
            [
                {
                    "outer_test_subject": "1",
                    "inner_test_subject": "2",
                    "balanced_accuracy": 0.3,
                }
            ]
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        predictions.to_csv(predictions_out_path, index=False)
        inner.to_csv(inner_cv_out_path, index=False)
        return summary

    monkeypatch.setattr(oracle, "run_bushmeg_source_loso", fake_run_bushmeg_source_loso)

    outputs = oracle.run_bushmeg_protocol4_oracle(
        tmp_path / "config.yml",
        include_oracle=True,
        methods="procrustes,mcca",
        out_dir=tmp_path / "oracle",
        overrides=["decoding.max_iter=5"],
    )

    assert len(calls) == 2
    assert all("source_loso.alignment_target_projection=oracle_target_calibrated_alignment" in call["overrides"] for call in calls)
    assert calls[0]["overrides"][0] == "decoding.max_iter=5"
    assert outputs["summary"]["protocol_category"].tolist() == [4, 4]
    assert outputs["summary"]["debug_upper_bound"].tolist() == [True, True]
    assert outputs["method_metadata"]["status"].tolist() == ["evaluated", "evaluated"]

    summary_csv = pd.read_csv(tmp_path / "oracle" / "summary.csv")
    metadata_csv = pd.read_csv(tmp_path / "oracle" / "method_metadata.csv")
    predictions_csv = pd.read_csv(tmp_path / "oracle" / "predictions.csv")

    assert summary_csv["valid_for_zero_calibration"].tolist() == [False, False]
    assert metadata_csv["uses_target_labels_for_fitting"].tolist() == [True, True]
    assert predictions_csv["protocol_name"].unique().tolist() == ["oracle_target_calibrated_alignment"]
