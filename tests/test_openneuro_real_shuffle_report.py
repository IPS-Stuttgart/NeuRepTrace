from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neureptrace.loso_observation_diagnostics import write_loso_observation_diagnostics
from neureptrace.openneuro_real_shuffle_report import write_real_shuffle_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _observation_rows(*, shuffle: bool, subjects: tuple[str, ...] = ("sub-01", "sub-02")) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        for time in (-0.056, 0.184, 0.232):
            for sample_index, true_label in enumerate((0, 1, 2)):
                if shuffle:
                    predicted = {0: 0, 1: 0, 2: 1}[true_label]
                elif time == 0.184 and subject == "sub-02" and true_label == 2:
                    predicted = 1
                else:
                    predicted = true_label
                probabilities = [0.1, 0.1, 0.1]
                probabilities[predicted] = 0.8
                rows.append(
                    {
                        "group": subject,
                        "time": time,
                        "sample_index": f"{subject}-{time}-{sample_index}",
                        "true_label": true_label,
                        "true_class": f"class_{true_label}",
                        "predicted_label": predicted,
                        "predicted_class": f"class_{predicted}",
                        "label_shuffle_control": bool(shuffle),
                        "label_shuffle_seed": 13 if shuffle else "",
                        "prob_class_0": probabilities[0],
                        "prob_class_1": probabilities[1],
                        "prob_class_2": probabilities[2],
                    }
                )
    return pd.DataFrame(rows)


def _write_artifact(
    root: Path,
    *,
    shuffle: bool,
    subjects: tuple[str, ...] = ("sub-01", "sub-02"),
    diagnostics_best_time: float = 0.184,
) -> None:
    decode = root / "decode"
    decode.mkdir(parents=True)
    observations = decode / "observations.csv"
    _observation_rows(shuffle=shuffle, subjects=subjects).to_csv(observations, index=False)
    write_loso_observation_diagnostics(
        observations,
        out_dir=decode / "diagnostics",
        best_time=diagnostics_best_time,
    )


def test_real_shuffle_report_writes_auditable_outputs(tmp_path: Path) -> None:
    real = tmp_path / "real"
    shuffle = tmp_path / "shuffle"
    out = tmp_path / "report"
    _write_artifact(real, shuffle=False)
    _write_artifact(shuffle, shuffle=True)

    paths = write_real_shuffle_report(real_dir=real, shuffle_dir=shuffle, out_dir=out, fixed_time=0.184)

    assert paths["summary"].name == "ds006629_real_vs_shuffle_summary.csv"
    assert paths["per_subject"].name == "ds006629_real_vs_shuffle_per_subject.csv"
    assert paths["markdown"].name == "ds006629_real_vs_shuffle.md"
    summary = pd.read_csv(paths["summary"])
    per_subject = pd.read_csv(paths["per_subject"])
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert summary.loc[0, "fixed_balanced_accuracy_real"] > summary.loc[0, "fixed_balanced_accuracy_shuffle"]
    assert bool(summary.loc[0, "real_label_shuffle_control"]) is False
    assert bool(summary.loc[0, "shuffle_label_shuffle_control"]) is True
    assert summary.loc[0, "shuffle_label_shuffle_seed"] == 13
    assert summary.loc[0, "top3_interpretation"] == "automatic_ceiling"
    assert per_subject["fixed_balanced_accuracy_delta"].notna().all()
    assert "## Fixed-Time Real vs Shuffle" in markdown
    assert "## Best-Time Real vs Shuffle (Exploratory)" in markdown
    assert "## Pre-Stimulus Sanity Check" in markdown
    assert "## Confusion Matrix at Fixed Time" in markdown
    assert "## Classwise Balanced Recalls" in markdown
    assert "Top-2 is informative" in markdown
    assert "Top-3 is automatic ceiling" in markdown


def test_real_shuffle_report_rejects_swapped_artifacts(tmp_path: Path) -> None:
    real = tmp_path / "real"
    shuffle = tmp_path / "shuffle"
    out = tmp_path / "report"
    _write_artifact(real, shuffle=False)
    _write_artifact(shuffle, shuffle=True)

    with pytest.raises(ValueError, match="real artifact is marked label_shuffle_control=true"):
        write_real_shuffle_report(real_dir=shuffle, shuffle_dir=real, out_dir=out, fixed_time=0.184)


def test_real_shuffle_report_rejects_nonoverlapping_subjects(tmp_path: Path) -> None:
    real = tmp_path / "real"
    shuffle = tmp_path / "shuffle"
    out = tmp_path / "report"
    _write_artifact(real, shuffle=False, subjects=("sub-01", "sub-02"))
    _write_artifact(shuffle, shuffle=True, subjects=("sub-03", "sub-04"))

    with pytest.raises(ValueError, match="no overlapping subjects"):
        write_real_shuffle_report(real_dir=real, shuffle_dir=shuffle, out_dir=out, fixed_time=0.184)


def test_real_shuffle_report_fallback_fixed_time_keeps_quality_fields(tmp_path: Path) -> None:
    real = tmp_path / "real"
    shuffle = tmp_path / "shuffle"
    out = tmp_path / "report"
    _write_artifact(real, shuffle=False, diagnostics_best_time=0.232)
    _write_artifact(shuffle, shuffle=True, diagnostics_best_time=0.232)

    paths = write_real_shuffle_report(real_dir=real, shuffle_dir=shuffle, out_dir=out, fixed_time=0.184)

    summary = pd.read_csv(paths["summary"])
    assert summary.loc[0, "fixed_time_real"] == 0.184
    assert summary.loc[0, "n_subjects_real"] == 2
    assert summary.loc[0, "n_classes"] == 3


def test_openneuro_real_vs_shuffle_workflow_uses_locked_defaults() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "openneuro-real-vs-shuffle-report.yml").read_text(encoding="utf-8")

    assert "26745113419" in workflow
    assert "26765349067" in workflow
    assert "openneuro-meg-ds006629-full-shard-aggregate" in workflow
    assert "openneuro-meg-ds006629-full-label-shuffle-seed-13-shard-aggregate" in workflow
    assert "ds006629_real_vs_shuffle_summary.csv" in workflow
    assert "ds006629_real_vs_shuffle_per_subject.csv" in workflow
    assert "ds006629_real_vs_shuffle.md" in workflow
    assert "gh run download" in workflow
    assert "python -m neureptrace.openneuro_real_shuffle_report" in workflow
    assert "$GITHUB_STEP_SUMMARY" in workflow
