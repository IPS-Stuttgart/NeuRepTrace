from __future__ import annotations

from pathlib import Path

import pandas as pd

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

    assert set(paths) == {"time_course", "per_subject", "confusion_matrix", "class_counts", "quality_summary"}
    per_subject = pd.read_csv(out_dir / "per_subject.csv")
    confusion = pd.read_csv(out_dir / "confusion_matrix.csv")
    class_counts = pd.read_csv(out_dir / "class_counts.csv")
    time_course = pd.read_csv(out_dir / "time_course_summary.csv")
    quality = pd.read_csv(out_dir / "quality_summary.csv")

    assert per_subject["subject"].tolist() == ["sub-01", "sub-02"]
    assert per_subject["top2_interpretation"].unique().tolist() == ["informative"]
    assert per_subject["top3_interpretation"].unique().tolist() == ["automatic_ceiling"]
    assert per_subject["fixed_time"].round(3).unique().tolist() == [0.184]
    assert per_subject["staged_n_trials"].tolist() == [3, 3]
    assert int(confusion.loc[(confusion["true_class"] == "class_2") & (confusion["predicted_class"] == "class_1"), "count"].iloc[0]) == 1
    assert class_counts.groupby("subject")["n_trials"].sum().to_dict() == {"sub-01": 3, "sub-02": 3}
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
