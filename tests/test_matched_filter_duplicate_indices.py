from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.matched_filter_detection import fit_stimulus_event_templates, score_stimulus_event_templates


def _stream(stream_id: str, *, event_onset: float | None) -> pd.DataFrame:
    rows = []
    bump = {0.0: 0.40, 0.1: 0.60, 0.2: 0.30}
    for time in np.round(np.arange(-0.3, 0.81, 0.1), 1):
        probability = 0.20
        if event_onset is not None:
            probability += bump.get(round(float(time - event_onset), 1), 0.0)
        rows.append(
            {
                "stream_id": stream_id,
                "time": float(time),
                "class_0": "A",
                "class_1": "B",
                "prob_class_0": probability,
                "prob_class_1": 1.0 - probability,
            }
        )
    return pd.DataFrame(rows)


def test_matched_filter_preserves_row_alignment_with_duplicate_indices() -> None:
    train_observations = pd.concat(
        [
            _stream("annotated", event_onset=0.0),
            _stream("distractor", event_onset=None),
        ]
    )
    assert not train_observations.index.is_unique

    templates = fit_stimulus_event_templates(
        train_observations,
        pd.DataFrame([{"stream_id": "annotated", "stimulus_class": "A", "onset_time": 0.0}]),
        template_window=(0.0, 0.2),
        template_step=0.1,
        target_classes=("A",),
        group_columns=(),
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    scan_observations = pd.concat(
        [
            _stream("event", event_onset=0.3),
            _stream("baseline", event_onset=None),
        ]
    )
    assert not scan_observations.index.is_unique

    scores = score_stimulus_event_templates(
        scan_observations,
        templates,
        group_columns=(),
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert set(scores["stream_id"]) == {"event", "baseline"}
    event_peak = scores.loc[scores["stream_id"].eq("event")].sort_values(
        "matched_filter_score", ascending=False
    ).iloc[0]
    assert event_peak["time"] == pytest.approx(0.3)
