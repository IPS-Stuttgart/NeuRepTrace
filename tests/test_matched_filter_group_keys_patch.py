from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.matched_filter_detection import _filter_by_values, fit_stimulus_event_templates, score_stimulus_event_templates


def _observations() -> pd.DataFrame:
    probabilities = [0.2, 0.8, 0.2]
    return pd.DataFrame(
        {
            "subject": [np.nan, np.nan, np.nan],
            "stream_id": [("session", 1), ("session", 1), ("session", 1)],
            "time": [0.0, 0.1, 0.2],
            "class_0": ["A", "A", "A"],
            "class_1": ["B", "B", "B"],
            "prob_class_0": probabilities,
            "prob_class_1": [1.0 - value for value in probabilities],
        }
    )


def test_matched_filter_preserves_missing_groups_and_tuple_stream_ids() -> None:
    observations = _observations()
    annotations = pd.DataFrame([{"subject": np.nan, "stimulus_class": "A", "onset_time": 0.0}])

    templates = fit_stimulus_event_templates(
        observations,
        annotations,
        template_window=(0.0, 0.1),
        template_step=0.1,
        target_classes=("A",),
        group_columns=("subject",),
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )
    scores = score_stimulus_event_templates(
        observations,
        templates,
        group_columns=("subject",),
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert not templates.empty
    assert not scores.empty
    assert scores["stream_id"].tolist() == [("session", 1)] * len(scores)


def test_matched_filter_group_filter_does_not_conflate_stringified_ids() -> None:
    frame = pd.DataFrame({"group": pd.Series([1, "1"], dtype=object), "kind": ["numeric", "text"]})

    selected = _filter_by_values(frame, {"group": 1})

    assert selected["kind"].tolist() == ["numeric"]
