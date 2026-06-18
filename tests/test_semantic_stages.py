from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.semantic_stages import analyze_semantic_stages, detect_stable_stages, posterior_columns, read_state_traces


def _state_trace_frame() -> pd.DataFrame:
    rows = []
    times = [-0.05, 0.10, 0.20, 0.30, 0.50]
    for sequence_id in range(12):
        for semantic_class in ("animate", "inanimate"):
            for time in times:
                if time < 0.0:
                    posterior_true = 0.52
                elif time <= 0.30:
                    posterior_true = 0.82
                else:
                    posterior_true = 0.57
                if semantic_class == "animate":
                    posterior_state_0 = posterior_true
                    posterior_state_1 = 1.0 - posterior_true
                    viterbi_class = "animate" if posterior_true >= 0.6 else "inanimate"
                    viterbi_state = 0 if viterbi_class == "animate" else 1
                else:
                    posterior_state_0 = 1.0 - posterior_true
                    posterior_state_1 = posterior_true
                    viterbi_class = "inanimate" if posterior_true >= 0.6 else "animate"
                    viterbi_state = 1 if viterbi_class == "inanimate" else 0
                rows.append(
                    {
                        "subject": "sub-01",
                        "fold": sequence_id % 2,
                        "sequence_id": f"{semantic_class}-{sequence_id}",
                        "decoder": "logistic",
                        "time": time,
                        "sample_index": sequence_id,
                        "true_class": semantic_class,
                        "predicted_class": semantic_class,
                        "viterbi_state": viterbi_state,
                        "viterbi_class": viterbi_class,
                        "viterbi_posterior": max(posterior_state_0, posterior_state_1),
                        "state_0": "animate",
                        "state_1": "inanimate",
                        "posterior_state_0": posterior_state_0,
                        "posterior_state_1": posterior_state_1,
                    }
                )
    return pd.DataFrame(rows)


def test_read_state_traces_detects_posterior_columns(tmp_path: Path):
    state_csv = tmp_path / "state_trace.csv"
    _state_trace_frame().to_csv(state_csv, index=False)

    traces = read_state_traces([state_csv])

    assert posterior_columns(traces) == ["posterior_state_0", "posterior_state_1"]
    assert traces["source_file"].unique().tolist() == ["state_trace.csv"]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("time", "late", "time values must be numeric"),
        ("time", float("inf"), "time values must be finite"),
        ("posterior_state_0", "high", "posterior_state_0 values must be numeric"),
        ("posterior_state_0", float("nan"), "posterior_state_0 values must be finite"),
        ("posterior_state_0", -0.1, "non-negative"),
        ("posterior_state_0", 1.2, "must not exceed 1.0"),
        ("viterbi_posterior", "high", "viterbi_posterior values must be numeric"),
        ("viterbi_posterior", float("nan"), "viterbi_posterior values must be finite"),
        ("viterbi_posterior", 1.2, "viterbi_posterior values must be between 0 and 1"),
    ],
)
def test_read_state_traces_rejects_malformed_numeric_values(tmp_path: Path, column: str, value, message: str):
    state_csv = tmp_path / "state_trace.csv"
    frame = _state_trace_frame()
    if isinstance(value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    frame.to_csv(state_csv, index=False)

    with pytest.raises(ValueError, match=message):
        read_state_traces([state_csv])


def test_read_state_traces_rejects_unnormalized_posterior_rows(tmp_path: Path):
    state_csv = tmp_path / "state_trace.csv"
    frame = _state_trace_frame()
    frame.loc[0, ["posterior_state_0", "posterior_state_1"]] = [0.2, 0.2]
    frame.to_csv(state_csv, index=False)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        read_state_traces([state_csv])


def test_analyze_semantic_stages_detects_category_conditioned_segments(tmp_path: Path):
    state_csv = tmp_path / "state_trace.csv"
    time_csv = tmp_path / "stage_time.csv"
    stages_csv = tmp_path / "stages.csv"
    report_md = tmp_path / "stage_report.md"
    _state_trace_frame().to_csv(state_csv, index=False)

    time_summary, stages, report = analyze_semantic_stages(
        [state_csv],
        posterior_threshold=0.7,
        match_threshold=0.7,
        min_duration=0.15,
        out_time=time_csv,
        out_stages=stages_csv,
        out_report=report_md,
    )

    assert time_csv.exists()
    assert stages_csv.exists()
    assert report_md.exists()
    assert report is not None and "do semantic categories unfold" in report
    assert sorted(stages["semantic_class"].tolist()) == ["animate", "inanimate"]
    assert stages["start_time"].round(3).tolist() == [0.1, 0.1]
    assert stages["stop_time"].round(3).tolist() == [0.3, 0.3]
    assert stages["mean_posterior_true_class"].round(3).tolist() == [0.82, 0.82]
    assert set(time_summary["true_class"]) == {"animate", "inanimate"}


def test_analyze_semantic_stages_keeps_emission_modes_distinct(tmp_path: Path):
    state_csv = tmp_path / "state_trace.csv"
    calibrated = _state_trace_frame().assign(emission_mode="calibrated")
    uncalibrated = _state_trace_frame().assign(emission_mode="uncalibrated")
    pd.concat([calibrated, uncalibrated], ignore_index=True).to_csv(state_csv, index=False)

    time_summary, stages, _ = analyze_semantic_stages(
        [state_csv],
        posterior_threshold=0.7,
        match_threshold=0.7,
        min_duration=0.15,
    )

    assert set(time_summary["emission_mode"]) == {"calibrated", "uncalibrated"}
    assert set(stages["emission_mode"]) == {"calibrated", "uncalibrated"}
    assert len(stages) == 4


def test_analyze_semantic_stages_rejects_unmapped_true_classes(tmp_path: Path):
    state_csv = tmp_path / "state_trace.csv"
    frame = _state_trace_frame()
    frame.loc[frame["true_class"] == "animate", "true_class"] = "animal"
    frame.to_csv(state_csv, index=False)

    with pytest.raises(ValueError, match="true_class values are not represented"):
        analyze_semantic_stages([state_csv])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"posterior_threshold": True}, "posterior_threshold"),
        ({"posterior_threshold": 1.5}, "posterior_threshold"),
        ({"match_threshold": np.nan}, "match_threshold"),
        ({"match_threshold": -0.1}, "match_threshold"),
        ({"min_duration": True}, "min_duration"),
        ({"min_duration": -0.01}, "min_duration"),
    ],
)
def test_analyze_semantic_stages_rejects_malformed_controls(tmp_path: Path, kwargs: dict, message: str):
    state_csv = tmp_path / "state_trace.csv"
    _state_trace_frame().to_csv(state_csv, index=False)

    with pytest.raises(ValueError, match=message):
        analyze_semantic_stages([state_csv], **kwargs)


def test_detect_stable_stages_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="time_summary is missing required columns"):
        detect_stable_stages(pd.DataFrame({"time": [0.1]}))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("time", "late", "time values must be numeric"),
        ("posterior_true_class_mean", float("inf"), "posterior_true_class_mean values must be finite"),
        ("viterbi_match_rate", np.nan, "viterbi_match_rate values must be finite"),
    ],
)
def test_detect_stable_stages_rejects_malformed_time_summary_values(column: str, value, message: str):
    time_summary = pd.DataFrame(
        {
            "decoder": ["logistic"],
            "emission_mode": ["calibrated"],
            "true_class": ["animate"],
            "time": [0.1],
            "posterior_true_class_mean": [0.8],
            "viterbi_match_rate": [0.9],
            "n_sequences": [3],
        }
    )
    if isinstance(value, str):
        time_summary[column] = time_summary[column].astype(object)
    time_summary.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        detect_stable_stages(time_summary)
