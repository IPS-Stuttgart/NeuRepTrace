from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.matched_filter_detection import (
    detect_matched_filter_stimulus_events,
    fit_matched_filter_thresholds,
    fit_stimulus_event_templates,
)


def _stream(stream_id: str, *, subject: object, event_onset: float | None = None) -> pd.DataFrame:
    rows = []
    bump = {0.0: 0.20, 0.1: 0.42, 0.2: 0.48, 0.3: 0.28}
    for time in np.round(np.arange(-0.5, 2.01, 0.1), 1):
        class_a = 0.25
        if event_onset is not None:
            class_a += bump.get(round(float(time - event_onset), 1), 0.0)
        probabilities = (class_a, 1.0 - class_a)
        predicted_label = int(np.argmax(probabilities))
        rows.append(
            {
                "subject": subject,
                "stream_id": stream_id,
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "time": float(time),
                "window_start": float(time) - 0.05,
                "window_stop": float(time) + 0.05,
                "predicted_label": predicted_label,
                "predicted_class": ("A", "B")[predicted_label],
                "confidence": max(probabilities),
                "class_0": "A",
                "class_1": "B",
                "prob_class_0": probabilities[0],
                "prob_class_1": probabilities[1],
            }
        )
    return pd.DataFrame(rows)


def test_matched_filter_preserves_missing_group_values_end_to_end():
    train_observations = _stream("train", subject=np.nan, event_onset=0.0)
    template_annotations = pd.DataFrame(
        [{"subject": pd.NA, "stream_id": "train", "stimulus_class": "A", "onset_time": 0.0}]
    )
    templates = fit_stimulus_event_templates(
        train_observations,
        template_annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        target_classes=["A"],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert not templates.empty
    assert templates["subject"].isna().all()

    scan_observations = _stream("scan", subject=None, event_onset=1.0)
    thresholds = fit_matched_filter_thresholds(
        scan_observations,
        templates,
        threshold_window=(-0.5, 0.5),
        threshold_quantile=1.0,
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )
    events = detect_matched_filter_stimulus_events(
        scan_observations,
        templates=templates,
        thresholds=thresholds,
        stream_columns=("stream_id",),
        detection_window=(0.5, 1.5),
        refractory=0.4,
        min_template_coverage=1.0,
    )

    assert len(events) == 1
    assert pd.isna(events.iloc[0]["subject"])
    assert events.iloc[0]["stimulus_class"] == "A"
    assert events.iloc[0]["onset_time"] == 1.0
