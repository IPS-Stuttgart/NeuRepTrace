from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neureptrace.loso_observation_diagnostics import write_loso_observation_diagnostics


def _toy_observations() -> pd.DataFrame:
    rows = []
    for subject, true_labels in {"sub-01": [0, 1, 2], "sub-02": [0, 1, 2]}.items():
        for time in (0.10, 0.184):
            for sample_index, true_label in enumerate(true_labels):
                probabilities = [0.1, 0.1, 0.1]
                if subject == "sub-01":
                    predicted = true_label
                elif time == 0.184 and true_label == 2:
                    predicted = 1
                else:
                    predicted = true_label
                probabilities[predicted] = 0.8
                confidence = 0.45 if subject == "sub-02" and time == 0.184 and true_label == 2 else 0.8
                rows.append(
                    {
                        "group": subject,
                        "time": time,
                        "sample_index": f"{subject}-{sample_index}",
                        "true_label": true_label,
                        "true_class": f"class_{true_label}",
                        "predicted_label": predicted,
                        "predicted_class": f"class_{predicted}",
                        "confidence": confidence,
                        "prob_class_0": probabilities[0],
                        "prob_class_1": probabilities[1],
                        "prob_class_2": probabilities[2],
                    }
                )
    return pd.DataFrame(rows)


def test_loso_observation_diagnostics_writes_subject_confusion_and_class_tables(tmp_path: Path):
    observations_csv = tmp_path / "observations.csv"
    stage_summary_csv = tmp_path / "stage_summary.csv"
    out_dir = tmp_path / "diagnostics"
    _toy_observations().to_csv(observations_csv, index=False)
    pd.DataFrame({"subject": ["sub-01", "sub-02"], "n_trials": [3, 3]}).to_csv(stage_summary_csv, index=False)

    paths = write_loso_observation_diagnostics(
        observations_csv,
        out_dir=out_dir,
        stage_summary_csv=stage_summary_csv,
        best_time=0.183,
    )

    assert set(paths) == {
        "time_course",
        "per_subject",
        "confusion_matrix",
        "class_counts",
        "selective_coverage",
        "quality_summary",
    }
    per_subject = pd.read_csv(out_dir / "per_subject.csv")
    confusion = pd.read_csv(out_dir / "confusion_matrix.csv")
    class_counts = pd.read_csv(out_dir / "class_counts.csv")
    selective = pd.read_csv(out_dir / "selective_coverage.csv")
    time_course = pd.read_csv(out_dir / "time_course_summary.csv")
    quality = pd.read_csv(out_dir / "quality_summary.csv")

    assert per_subject["subject"].tolist() == ["sub-01", "sub-02"]
    assert per_subject["top2_interpretation"].unique().tolist() == ["informative"]
    assert per_subject["top3_interpretation"].unique().tolist() == ["automatic_ceiling"]
    assert per_subject["fixed_time"].round(3).unique().tolist() == [0.184]
    assert per_subject["staged_n_trials"].tolist() == [3, 3]
    assert int(confusion.loc[(confusion["true_class"] == "class_2") & (confusion["predicted_class"] == "class_1"), "count"].iloc[0]) == 1
    assert class_counts.groupby("subject")["n_trials"].sum().to_dict() == {"sub-01": 3, "sub-02": 3}
    assert selective["coverage_target"].tolist() == [1.0, 0.9, 0.8, 0.7]
    assert selective.loc[selective["coverage_target"] == 1.0, "balanced_accuracy"].iloc[0].round(6) == round(5 / 6, 6)
    assert selective.loc[selective["coverage_target"] == 0.8, "balanced_accuracy"].iloc[0].round(6) == 1.0
    assert selective.loc[selective["coverage_target"] == 0.8, "selective_risk"].iloc[0].round(6) == 0.0
    assert selective.loc[selective["coverage_target"] == 0.8, "all_classes_present"].iloc[0]
    assert time_course["top2_interpretation"].unique().tolist() == ["informative"]
    assert time_course["top3_interpretation"].unique().tolist() == ["automatic_ceiling"]
    assert quality.loc[0, "n_subjects"] == 2
    assert quality.loc[0, "n_classes"] == 3
    assert quality.loc[0, "fixed_time"].round(3) == 0.184
    assert quality.loc[0, "fixed_balanced_accuracy"].round(6) == round(5 / 6, 6)
    assert quality.loc[0, "fixed_balanced_minus_chance"].round(6) == round(5 / 6 - 1 / 3, 6)
    assert quality.loc[0, "subjects_fixed_above_chance"] == 2
    assert quality.loc[0, "top3_interpretation"] == "automatic_ceiling"


def test_loso_observation_diagnostics_tolerates_missing_stage_summary(tmp_path: Path):
    observations_csv = tmp_path / "observations.csv"
    missing_stage_summary_csv = tmp_path / "stage_summary.csv"
    out_dir = tmp_path / "diagnostics"
    _toy_observations().to_csv(observations_csv, index=False)

    paths = write_loso_observation_diagnostics(
        observations_csv,
        out_dir=out_dir,
        stage_summary_csv=missing_stage_summary_csv,
        best_time=0.184,
    )

    assert paths["quality_summary"].exists()
    per_subject = pd.read_csv(paths["per_subject"])
    assert per_subject["subject"].tolist() == ["sub-01", "sub-02"]
    assert per_subject["staged_n_trials"].isna().all()


def test_loso_observation_diagnostics_rejects_fractional_true_labels(tmp_path: Path):
    observations = _toy_observations()
    observations["true_label"] = observations["true_label"].astype(float)
    observations.loc[0, "true_label"] = 0.5
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="true_label values must be integer-valued"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_fractional_predicted_labels(tmp_path: Path):
    observations = _toy_observations()
    observations["predicted_label"] = observations["predicted_label"].astype(float)
    observations.loc[0, "predicted_label"] = 0.5
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="predicted_label values must be integer-valued"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_nonfinite_probabilities(tmp_path: Path):
    observations = _toy_observations()
    observations.loc[0, "prob_class_0"] = float("nan")
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="prob_class_\\* values must be finite"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_negative_probabilities(tmp_path: Path):
    observations = _toy_observations()
    observations.loc[0, "prob_class_0"] = -0.1
    observations.loc[0, "prob_class_1"] = 0.3
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="prob_class_\\* values must be non-negative"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_unnormalized_probabilities(tmp_path: Path):
    observations = _toy_observations()
    observations.loc[0, "prob_class_0"] = 0.7
    observations.loc[0, "prob_class_1"] = 0.1
    observations.loc[0, "prob_class_2"] = 0.1
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="prob_class_\\* rows must sum to 1"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_nonfinite_confidence(tmp_path: Path):
    observations = _toy_observations()
    fixed_index = observations.index[observations["time"].eq(0.184)][0]
    observations.loc[fixed_index, "confidence"] = float("nan")
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="confidence values must be finite"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_out_of_range_confidence(tmp_path: Path):
    observations = _toy_observations()
    fixed_index = observations.index[observations["time"].eq(0.184)][0]
    observations.loc[fixed_index, "confidence"] = 1.2
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="confidence values must lie"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_mixed_decoder_provenance(tmp_path: Path):
    observations = _toy_observations()
    observations["decoder"] = "multinomial-logistic"
    observations.loc[0, "decoder"] = "linear-svm"
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="mixes 'decoder' provenance"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_mixed_shuffle_provenance(tmp_path: Path):
    observations = _toy_observations()
    observations["label_shuffle_control"] = False
    observations.loc[0, "label_shuffle_control"] = True
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="mixes 'label_shuffle_control' provenance"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_missing_seed_mixed_with_seed(tmp_path: Path):
    observations = _toy_observations()
    observations["label_shuffle_control"] = True
    observations["label_shuffle_seed"] = "13"
    observations.loc[0, "label_shuffle_seed"] = ""
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="missing 'label_shuffle_seed' provenance"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)


def test_loso_observation_diagnostics_rejects_duplicate_observation_rows(tmp_path: Path):
    observations = pd.concat([_toy_observations(), _toy_observations().iloc[[0]]], ignore_index=True)
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="duplicate rows"):
        write_loso_observation_diagnostics(observations_csv, out_dir=tmp_path / "diagnostics", best_time=0.184)
