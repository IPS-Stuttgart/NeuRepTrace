from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.matched_filter_detection import fit_stimulus_event_templates, score_stimulus_event_templates


CLASS_NAMES = ("A", "B")


def _row(stream_id: str, time: float, probabilities: tuple[float, float]) -> dict[str, object]:
    predicted_label = int(np.argmax(probabilities))
    return {
        "subject": "sub-01",
        "stream_id": stream_id,
        "decoder": "logistic",
        "emission_mode": "calibrated",
        "time": time,
        "window_start": time - 0.05,
        "window_stop": time + 0.05,
        "predicted_label": predicted_label,
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": max(probabilities),
        "class_0": "A",
        "class_1": "B",
        "prob_class_0": probabilities[0],
        "prob_class_1": probabilities[1],
    }


def _stream(stream_id: str, *, event_onset: float | None = None) -> pd.DataFrame:
    rows = []
    bump = {0.0: 0.20, 0.1: 0.42, 0.2: 0.48, 0.3: 0.28}
    for time in np.round(np.arange(-0.5, 2.01, 0.1), 1):
        class_a = 0.25
        if event_onset is not None:
            class_a += bump.get(round(float(time - event_onset), 1), 0.0)
        rows.append(_row(stream_id, float(time), (class_a, 1.0 - class_a)))
    return pd.DataFrame(rows)


def test_template_fitting_handles_duplicate_indices_across_streams():
    observations = pd.concat(
        [
            _stream("train-a", event_onset=0.0),
            _stream("train-b", event_onset=1.0),
        ],
        ignore_index=False,
    )
    assert observations.index.duplicated().any()
    annotations = pd.DataFrame(
        [
            {"stream_id": "train-a", "stimulus_class": "A", "onset_time": 0.0},
            {"stream_id": "train-b", "stimulus_class": "A", "onset_time": 1.0},
        ]
    )

    templates = fit_stimulus_event_templates(
        observations,
        annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        target_classes=["A"],
        stream_columns=("stream_id",),
        min_template_events=2,
        min_template_coverage=1.0,
    )

    assert not templates.empty
    assert templates["n_template_events"].eq(2).all()


def test_template_scoring_handles_duplicate_indices_across_streams():
    train_observations = _stream("train", event_onset=0.0)
    annotations = pd.DataFrame([{"stream_id": "train", "stimulus_class": "A", "onset_time": 0.0}])
    templates = fit_stimulus_event_templates(
        train_observations,
        annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        target_classes=["A"],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )
    observations = pd.concat(
        [
            _stream("scan-a", event_onset=1.0),
            _stream("scan-b", event_onset=0.5),
        ],
        ignore_index=False,
    )
    assert observations.index.duplicated().any()

    scores = score_stimulus_event_templates(
        observations,
        templates,
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert set(scores["stream_id"]) == {"scan-a", "scan-b"}
    peak_rows = scores.loc[scores.groupby("stream_id")["matched_filter_score"].idxmax()]
    peak_times = dict(zip(peak_rows["stream_id"], peak_rows["time"], strict=True))
    assert peak_times == {"scan-a": 1.0, "scan-b": 0.5}
