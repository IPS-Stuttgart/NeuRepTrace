from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.matched_filter_detection import fit_stimulus_event_templates


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "stream_id": ["stream", "stream"],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "time": [0.0, 0.1],
            "class_0": ["A", "A"],
            "class_1": ["B", "B"],
            "prob_class_0": [0.7, 0.9],
            "prob_class_1": [0.3, 0.1],
        }
    )


@pytest.mark.parametrize(
    "onset_time",
    [None, np.nan, np.inf, -np.inf, True, 0.0 + 0.0j, "not-a-time", np.asarray([0.0])],
    ids=["none", "nan", "positive-infinity", "negative-infinity", "boolean", "complex", "text", "vector"],
)
def test_matched_filter_rejects_invalid_template_annotation_onset_times(onset_time: object) -> None:
    annotations = pd.DataFrame(
        [{"stream_id": "stream", "stimulus_class": "A", "onset_time": onset_time}],
        index=[17],
    )

    with pytest.raises(
        ValueError,
        match=r"onset_time values must be finite real numbers; invalid row\(s\): \[17\]",
    ):
        fit_stimulus_event_templates(
            _observations(),
            annotations,
            template_window=(0.0, 0.1),
            template_step=0.1,
            target_classes=["A"],
            stream_columns=("stream_id",),
            min_template_coverage=1.0,
        )


def test_matched_filter_accepts_numeric_string_template_annotation_onset_times() -> None:
    annotations = pd.DataFrame(
        [{"stream_id": "stream", "stimulus_class": "A", "onset_time": "0.0"}],
    )

    templates = fit_stimulus_event_templates(
        _observations(),
        annotations,
        template_window=(0.0, 0.1),
        template_step=0.1,
        target_classes=["A"],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert templates["template_time"].tolist() == [0.0, 0.1]
    assert templates["n_template_events"].tolist() == [1, 1]
