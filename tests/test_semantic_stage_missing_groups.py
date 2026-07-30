import pandas as pd

from neureptrace.semantic_stages import (
    build_stage_report,
    detect_stable_stages,
    summarize_category_timecourse,
    summarize_dominant_timecourse,
)


def _missing_group_trace() -> pd.DataFrame:
    rows = []
    for subject in ("sub-01", "sub-02"):
        for time in (0.0, 0.2):
            rows.append(
                {
                    "subject": subject,
                    "sequence_id": "trial-1",
                    "decoder": pd.NA,
                    "emission_mode": pd.NA,
                    "time": time,
                    "true_class": "animate",
                    "viterbi_class": "animate",
                    "viterbi_posterior": 0.9,
                    "state_0": "animate",
                    "state_1": "inanimate",
                    "posterior_state_0": 0.9,
                    "posterior_state_1": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_semantic_stage_pipeline_preserves_missing_optional_groups():
    summary, _ = summarize_category_timecourse(_missing_group_trace())

    assert len(summary) == 2
    assert summary["decoder"].isna().all()
    assert summary["emission_mode"].isna().all()

    stages = detect_stable_stages(
        summary,
        posterior_threshold=0.8,
        match_threshold=0.8,
        min_duration=0.1,
    )

    assert len(stages) == 1
    assert stages["decoder"].isna().all()
    assert stages["emission_mode"].isna().all()

    report = build_stage_report(
        summary,
        stages,
        posterior_threshold=0.8,
        match_threshold=0.8,
        min_duration=0.1,
    )
    assert "No stable semantic stages" not in report
    assert "|  |  | animate |" in report


def test_dominant_semantic_summary_preserves_missing_optional_groups():
    traces = _missing_group_trace().drop(columns="true_class")

    summary = summarize_dominant_timecourse(traces)

    assert len(summary) == 2
    assert summary["decoder"].isna().all()
    assert summary["emission_mode"].isna().all()
