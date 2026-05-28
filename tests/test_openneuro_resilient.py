from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.openneuro_meg import StageResult
from neureptrace.openneuro_resilient import (
    StageFailure,
    participant_override_token,
    stage_dataset_resilient,
    successful_subjects_override,
    write_stage_failure_summary,
)


def test_participant_override_token_normalizes_bids_subject_labels() -> None:
    assert participant_override_token("sub-06") == "6"
    assert participant_override_token("sub-010") == "10"
    assert participant_override_token("sub-control") == "sub-control"
    assert participant_override_token(14) == "14"


def test_successful_subjects_override_keeps_numeric_decode_template_tokens() -> None:
    assert successful_subjects_override((1, "sub-06", 14)) == "1,6,14"


def test_write_stage_failure_summary_writes_machine_readable_csv(tmp_path: Path) -> None:
    out_path = tmp_path / "failures.csv"

    write_stage_failure_summary(
        [
            StageFailure(
                dataset_id="ds006629",
                subject="sub-06",
                error_type="RuntimeError",
                error="Number of channels is not defined",
            )
        ],
        out_path,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "dataset_id,subject,error_type,error" in text
    assert "ds006629,sub-06,RuntimeError,Number of channels is not defined" in text


def test_stage_dataset_resilient_skips_failed_subjects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int | str] = []

    def fake_stage_subject(*_args, subject, **_kwargs):
        calls.append(subject)
        if subject == 2:
            raise RuntimeError("corrupt raw file")
        return StageResult(
            dataset_id="ds006629",
            subject=f"sub-{int(subject):02d}",
            epochs_path=tmp_path / f"sub-{int(subject):02d}-epo.fif",
            n_trials=12,
            labels=("a", "b"),
            runs=("0",),
        )

    monkeypatch.setattr("neureptrace.openneuro_resilient.stage_subject", fake_stage_subject)

    results, failures, successful_subjects = stage_dataset_resilient(
        "ds006629",
        bids_root=tmp_path / "raw",
        staged_dir=tmp_path / "staged",
        subjects="1-3",
        runs="0",
        skip_failed_subjects=True,
    )

    assert calls == [1, 2, 3]
    assert [result.subject for result in results] == ["sub-01", "sub-03"]
    assert successful_subjects == (1, 3)
    assert failures == [
        StageFailure(
            dataset_id="ds006629",
            subject="sub-02",
            error_type="RuntimeError",
            error="corrupt raw file",
        )
    ]


def test_stage_dataset_resilient_reraises_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_stage_subject(*_args, **_kwargs):
        raise RuntimeError("corrupt raw file")

    monkeypatch.setattr("neureptrace.openneuro_resilient.stage_subject", fake_stage_subject)

    with pytest.raises(RuntimeError, match="corrupt raw file"):
        stage_dataset_resilient(
            "ds006629",
            bids_root=tmp_path / "raw",
            staged_dir=tmp_path / "staged",
            subjects="1",
            runs="0",
        )
