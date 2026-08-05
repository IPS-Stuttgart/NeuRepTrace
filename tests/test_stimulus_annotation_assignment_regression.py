from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.stimulus_detection import match_stimulus_annotations


def _event(onset_time: float) -> dict[str, object]:
    return {
        "stream_id": "run-1",
        "stimulus_class": "A",
        "onset_time": onset_time,
    }


def _annotation(annotation_id: int, onset_time: float) -> dict[str, object]:
    return {
        "stream_id": "run-1",
        "annotation_id": annotation_id,
        "stimulus_class": "A",
        "onset_time": onset_time,
    }


def test_annotation_matching_maximizes_one_to_one_match_count() -> None:
    events = pd.DataFrame([_event(1.0), _event(1.4)])
    annotations = pd.DataFrame([_annotation(1, 0.0), _annotation(2, 1.5)])

    matched = match_stimulus_annotations(
        events,
        annotations,
        stream_columns=("stream_id",),
        match_tolerance=1.0,
    )

    assert matched["candidate_annotation_id"].tolist() == [2, 2]
    assert matched["matched_annotation_id"].tolist() == [1, 2]
    assert matched["is_true_positive"].tolist() == [True, True]
    assert matched["is_duplicate_detection"].tolist() == [False, False]


def test_annotation_matching_minimizes_total_latency_after_cardinality() -> None:
    events = pd.DataFrame([_event(0.9), _event(1.1)])
    annotations = pd.DataFrame([_annotation(1, 0.0), _annotation(2, 1.0)])

    matched = match_stimulus_annotations(
        events,
        annotations,
        stream_columns=("stream_id",),
        match_tolerance=2.0,
    )

    assert matched["matched_annotation_id"].tolist() == [1, 2]
    assert matched["latency"].tolist() == pytest.approx([0.9, 0.1])
