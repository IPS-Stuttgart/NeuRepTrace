from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.stimulus_detection import (
    detect_stimulus_events,
    fit_stimulus_detection_thresholds,
    summarize_stimulus_events,
)


def _missing_identifier_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [np.nan] * 4,
            "stream_id": [np.nan] * 4,
            "time": [-0.2, -0.1, 0.1, 0.2],
            "prob_class_0": [0.9, 0.8, 0.2, 0.1],
            "prob_class_1": [0.1, 0.2, 0.8, 0.9],
        }
    )


def test_stimulus_detection_preserves_missing_group_and_stream_identifiers():
    observations = _missing_identifier_observations()

    thresholds = fit_stimulus_detection_thresholds(
        observations,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=0.5,
        target_classes=[1],
        group_columns=["subject"],
        stream_columns=["stream_id"],
    )

    assert len(thresholds) == 1
    assert pd.isna(thresholds.loc[0, "subject"])
    assert thresholds.loc[0, "score_threshold"] == pytest.approx(0.15)

    events = detect_stimulus_events(
        observations,
        thresholds=thresholds,
        target_classes=[1],
        group_columns=["subject"],
        stream_columns=["stream_id"],
        detection_window=(0.0, 1.0),
        min_consecutive=2,
    )

    assert len(events) == 1
    assert pd.isna(events.loc[0, "subject"])
    assert pd.isna(events.loc[0, "stream_id"])
    assert events.loc[0, "event_index"] == 0
    assert events.loc[0, "peak_score"] == pytest.approx(0.9)


def test_stimulus_detection_preserves_missing_constrained_identifier_dtypes():
    observations = _missing_identifier_observations()
    observations["subject"] = pd.Series(pd.Categorical([None] * len(observations), categories=["known"]))
    observations["stream_id"] = pd.Series([pd.NA] * len(observations), dtype="Int64")

    thresholds = fit_stimulus_detection_thresholds(
        observations,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=0.5,
        target_classes=[1],
        group_columns=["subject"],
        stream_columns=["stream_id"],
    )

    assert len(thresholds) == 1
    assert pd.isna(thresholds.loc[0, "subject"])
    assert pd.isna(thresholds.loc[0, "stream_id"])

    events = detect_stimulus_events(
        observations,
        thresholds=thresholds,
        target_classes=[1],
        group_columns=["subject"],
        stream_columns=["stream_id"],
        detection_window=(0.0, 1.0),
        min_consecutive=2,
    )

    assert len(events) == 1
    assert pd.isna(events.loc[0, "subject"])
    assert pd.isna(events.loc[0, "stream_id"])
    assert events.loc[0, "event_index"] == 0


def test_stimulus_summary_counts_duration_for_missing_stream_identifier():
    observations = _missing_identifier_observations()
    events = pd.DataFrame(
        {
            "subject": [np.nan],
            "stream_id": [np.nan],
            "stimulus_class": ["1"],
            "onset_time": [0.1],
            "is_true_positive": [False],
        }
    )

    summary = summarize_stimulus_events(
        events,
        observations=observations,
        group_columns=["subject"],
        stream_columns=["stream_id"],
    )

    assert len(summary) == 1
    assert pd.isna(summary.loc[0, "subject"])
    assert summary.loc[0, "false_positive_count"] == 1
    assert summary.loc[0, "false_alarms_per_minute"] == pytest.approx(150.0)